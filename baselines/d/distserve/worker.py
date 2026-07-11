"""
Adapted from https://github.com/vllm/worker/worker.py
"""
import copy
import time
from typing import List, Tuple, Optional
import socket

import ray
import torch
import torch.distributed

from distserve.config import ModelConfig, CacheConfig, ParallelConfig
from distserve.request import Request, BatchedRequests
from distserve.utils import set_random_seed, cudaMemoryIpcHandle, Stage, random_digits
from distserve.utils import get_gpu_memory, set_random_seed, GB, MB
from distserve.logger import init_logger
from pathlib import Path
from vllm.utils import init_cached_hf_modules
init_cached_hf_modules()
logger = init_logger(__name__)

@ray.remote(num_cpus=0, num_gpus=1)
class ParaWorker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache, the KV swap and executing the model on the GPU.
    In case of distributed inference, each worker is assigned a partition of
    the model.

    """

    # call remote workers as follows
    # fn calling -> ray.get(self.workers[0][0].ready.remote())
    # attribute getting -> ray.get(self.workers[0][0].self.remote("gpu_id"))
    def self(self, param_name, param_val=None):
        if param_val is None:
            return getattr(self, param_name)
        else:
            setattr(self, param_name, param_val)

    def __init__(
        self,
        worker_id: int,
        stage: Stage,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig = ParallelConfig(),
        tensor_parallel_id: List[int] = None,   # Although the type is list[int], it is actually a NCCL unique ID
        pipeline_parallel_id: List[int] = None, # Same as above
    ) -> None:
        self.worker_id = worker_id
        self.stage = stage
        self.model = None
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.tensor_parallel_id = tensor_parallel_id
        self.pipeline_parallel_id = pipeline_parallel_id
        self.gpu_id = ray.get_gpu_ids()[0]
        
        self.device = torch.device(f"cuda:0")
        torch.cuda.set_device(self.device)
        
        # K/V cache on GPU
        self.k_cache = None
        self.v_cache = None
        # K/V swap on CPU
        self.k_swap = None
        self.v_swap = None
        # CUDA streams for swapping in and out
        self.swap_in_stream = torch.cuda.Stream()
        self.swap_out_stream = torch.cuda.Stream()
        # The swap_event_table, refer to block_manager.py for more details
        self.swap_event_table = {}
        # The latest swap event in each stream
        # Used when we need to wait for all swap events to finish
        self.latest_swap_in_event = None
        self.latest_swap_out_event = None
        # Statistics
        self.execution_time = 0.0
        self.blocked_swapping_time = 0.0

    def ready(self):
        """
        Ray functions queue inside one single actor to be executed in order.
        If ready is called, the actor is ready.
        """
        logger.info(f"Worker {self.stage}.#{self.worker_id} created on host {socket.gethostname()} and gpu #{self.gpu_id}")
        pass

    def init_model(self):
        # Initialize the model.
        set_random_seed(self.model_config.seed)
        engine_config = self.model_config.vllm_engine_config
        from vllm.worker.worker import init_worker_distributed_environment
        # print(int(random_digits(4))+int(self.gpu_id))
        init_worker_distributed_environment(engine_config.parallel_config,
                                            rank = 0,
                                            distributed_init_method=f'tcp://localhost:{int(random_digits(4))+int(self.gpu_id)}',
                                            local_rank=0)
        from vllm.worker.model_runner import ModelRunner, GPUModelRunnerBase
        model_runner: GPUModelRunnerBase = ModelRunner(model_config=engine_config.model_config,
                                parallel_config=engine_config.parallel_config,
                                scheduler_config=engine_config.scheduler_config,
                                device_config=engine_config.device_config,
                                cache_config=engine_config.cache_config,
                                load_config=engine_config.load_config,
                                lora_config=engine_config.lora_config,
                                kv_cache_dtype=engine_config.cache_config.cache_dtype,
                                is_driver_worker=1,
                                prompt_adapter_config=None,
                                observability_config=None)
        model_runner.load_model()
        # print(model_runner)
        self.engine_config = engine_config
        self.model_runner = model_runner
        # print(f'{self.stage} Model Loaded')
        torch.cuda.synchronize()



    def init_kvcache_and_swap(self, num_gpu_blocks, num_cpu_blocks) -> (cudaMemoryIpcHandle, cudaMemoryIpcHandle):
        kv_cache_shape = (
            self.model_config.get_num_layers(self.parallel_config),
            2,
            num_gpu_blocks,
            self.cache_config.block_size,
            self.model_config.get_num_heads(self.parallel_config),
            self.model_config.get_head_size(),
        )

        self.kv_cache =  torch.empty(kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda")
        self.k_cache = self.kv_cache[:,0,...]
        self.v_cache = self.kv_cache[:,1,...]

        # print('\n\n\n')
        # print(f'kv_cache {self.kv_cache.shape}')
        # print(f'k_cache {self.k_cache.shape}')
        # print(f'v_cache {self.v_cache.shape}')
        # print('\n\n\n')

        kv_swap_shape = (
            self.model_config.get_num_layers(self.parallel_config),
            2,
            num_cpu_blocks,
            self.cache_config.block_size,
            self.model_config.get_num_heads(self.parallel_config),
            self.model_config.get_head_size(),
        )

        self.kv_swap =  torch.empty(kv_swap_shape, dtype=self.model_config.get_torch_dtype(), device="cpu", pin_memory=True)
        self.k_swap = self.kv_swap[:,0,...]
        self.v_swap = self.kv_swap[:,1,...]

        torch.cuda.synchronize()

        return torch.ops.block_migration_ops.get_ipc_mem_handle(self.kv_cache)


    def _get_block_size_in_bytes(
        self,
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        # the shape of one slot in k/v cache is [num_layers, num_local_heads, block_size, head_dim]
        num_layers = model_config.get_num_layers(parallel_config)
        num_heads = model_config.get_num_heads(parallel_config)
        head_dim = model_config.get_head_size()

        key_cache_size = num_layers * num_heads * block_size * head_dim
        total = key_cache_size * 2
        dtype_size = model_config.get_dtype_size()
        return total * dtype_size

    @torch.inference_mode()
    def _profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile memory usage with max_batch_size requests and the total number of tokens equal to max_tokens_per_batch.
        total_gpu_memory = get_gpu_memory()
        peak_runtime_memory = (total_gpu_memory * 0.01 + self.model_config.get_model_size_in_bytes(parallel_config=self.parallel_config))

        block_size_in_bytes = self._get_block_size_in_bytes(block_size, self.model_config, self.parallel_config)

        # logger.info(f"KV cache size for one token: {block_size_in_bytes / block_size / MB:.5f} MB")
        num_gpu_blocks = int((total_gpu_memory * gpu_memory_utilization - peak_runtime_memory) // block_size_in_bytes)

        num_cpu_blocks = int(cpu_swap_space // block_size_in_bytes)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        logger.info(f"Runtime peak memory: {peak_runtime_memory / GB:.3f} GB. Total GPU memory: {total_gpu_memory / GB:.3f} GB. Num GPU Blocks: {num_gpu_blocks} Num CPU Blocks: {num_cpu_blocks}")


        # Reset the seed to ensure that the random state is not affected by the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks


    @torch.inference_mode()
    def step_vllm(
        self,
        request_ids: List[int],
        seqs,
        vision_blocks=None
    ) -> List[int]:
        """Run one step of inference on the batch of requests."""

        start = time.perf_counter()
        # Check whether synchronization is necessary
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # We let the current stream wait for the swap event
                # This is non-blocking (It just stop the current stream instead
                # of chocking the CPU)
                self.swap_event_table[request_id].wait(torch.cuda.current_stream())
                self.swap_event_table.pop(request_id, None)
        self.blocked_swapping_time += time.perf_counter() - start
        kv_cache = self.kv_cache
        finished_requests_ids = []


        try:
            model_input = self.model_runner.prepare_model_input(seqs, finished_requests_ids=finished_requests_ids)
            seq_outs = self.model_runner.execute_model(model_input, kv_cache, None)
        except Exception as e:
            # print(f'\n\nException in worker: {e}  \n')
            raise e
        generated_tokens_ids = [output.samples[0].output_token for output in seq_outs[0]]
        self.execution_time += time.perf_counter() - start
        return generated_tokens_ids

    def register_kvcache_mem_handles_context(
        self,
        context_parallel_config: ParallelConfig,
        kvcache_ipc_mem_handles: List[List[cudaMemoryIpcHandle]]
    ):
        for pp_rank, stage_workers in enumerate(kvcache_ipc_mem_handles):
            for tp_rank, mem_handle in enumerate(stage_workers):
                tmp_parallel_config = copy.deepcopy(context_parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = pp_rank
                tmp_parallel_config.tensor_parallel_rank = tp_rank
                
                torch.ops.block_migration_ops.register_ipc_mem_handle_context(
                    kvcache_ipc_mem_handles[pp_rank][tp_rank],
                    self.model_config.get_num_layers(),
                    self.model_config.get_num_heads(),
                    tmp_parallel_config.to_list(),
                    self.parallel_config.to_list() # decoding worker's parallel config (as that worker calls the fn)
                )
                
        torch.cuda.synchronize()

    def migrate_blocks_context(
        self,
        context_block_indexes: List[int],
        context_parallel_config: ParallelConfig,
        decoding_block_indexes: List[int]
    ):
        torch.ops.block_migration_ops.migrate_blocks_context(
            context_parallel_config.pipeline_parallel_size,
            context_parallel_config.tensor_parallel_size,
            context_parallel_config.data_parallel_rank,
            context_block_indexes,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_rank,
            self.parallel_config.tensor_parallel_rank,
            decoding_block_indexes,
            self.kv_cache
        )

    def register_kvcache_mem_handles_encoding(
        self,
        encoding_parallel_config: ParallelConfig,
        vecache_ipc_mem_handles: List[List[cudaMemoryIpcHandle]]
    ):
        for pp_rank, stage_workers in enumerate(vecache_ipc_mem_handles):
            for tp_rank, mem_handle in enumerate(stage_workers):
                tmp_parallel_config = copy.deepcopy(encoding_parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = pp_rank
                tmp_parallel_config.tensor_parallel_rank = tp_rank
                
                torch.ops.block_migration_ops.register_ipc_mem_handle_encoding(
                    vecache_ipc_mem_handles[pp_rank][tp_rank],
                    tmp_parallel_config.to_list(),
                    self.parallel_config.to_list() # context worker's parallel config (as that worker calls the fn)
                )
        torch.cuda.synchronize()

    def migrate_blocks_encoding(
        self,
        encoding_block_indexes: List[int],
        encoding_parallel_config: ParallelConfig,
        context_block_indexes: List[int]
    ):
        torch.ops.block_migration_ops.migrate_blocks_encoding(
            encoding_parallel_config.pipeline_parallel_size,
            encoding_parallel_config.tensor_parallel_size,
            encoding_parallel_config.data_parallel_rank,
            encoding_block_indexes,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_rank,
            self.parallel_config.tensor_parallel_rank,
            context_block_indexes,
            self.ve_cache,
        )

    def swap_blocks(
        self,
        request_ids: List[int],
        source_block_ids: List[int],
        target_block_ids: List[int],
        is_swap_in: bool,
    ):
        """Swap some blocks between CPU and GPU
        If is_swap_in, then move blocks from CPU to GPU, i.e. CPU block
        #source_block_ids[0] will be copied to GPU block #target_block_ids[0]
        and so on. Similar for is_swap_in = False
        """

        # print(f"Swap {source_block_ids} ({'CPU' if is_swap_in else 'GPU'}) to {target_block_ids} ({'GPU' if is_swap_in else 'CPU'})")
        stream = self.swap_in_stream if is_swap_in else self.swap_out_stream

        # Record event
        event = torch.cuda.Event()
        event.record(stream)

        # Save that event
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # If we've issued another swapping operation before, we shall wait it
                # Pay attention to the difference between wait() and synchronize()
                self.swap_event_table[request_id].wait(stream)
            self.swap_event_table[request_id] = event
        if is_swap_in:
            self.latest_swap_in_event = event
        else:
            self.latest_swap_out_event = event

        # Swap
        with torch.cuda.stream(stream):
            torch.ops.swapping_ops.swap(
                source_block_ids,
                target_block_ids,
                is_swap_in,
                self.k_cache,
                self.v_cache,
                self.k_swap,
                self.v_swap,
            )

    def clear_request_resource(self, request_id: int):
        """Clear the resources associated with the request."""
        """This is called by LLMEngine when a request is finished or aborted"""
        # Clear the swap event table
        self.swap_event_table.pop(request_id, None)

    def clear_request_resource_batched(self, requests: List[Request]):
        """Clear the resources associated with the requests."""
        for request in requests:
            self.clear_request_resource(request.request_id)

    def wait_for_all_swap_in(self):
        """Wait for all swap in to finish"""
        if self.latest_swap_in_event is not None:
            self.latest_swap_in_event.synchronize()
            self.latest_swap_in_event = None

    def wait_for_all_swap_out(self):
        """Wait for all swap out to finish"""
        if self.latest_swap_out_event is not None:
            self.latest_swap_out_event.synchronize()
            self.latest_swap_out_event = None
