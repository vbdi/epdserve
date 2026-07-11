import time, copy
from typing import Callable, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import asyncio

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

import torch

from epdserve.logger import init_logger
from epdserve.config import (
    ModelConfig,
    ParallelConfig,
    CacheConfig,
)
from epdserve.request import (
    Request, 
    BatchedRequests,
    MigratingRequest
)
from epdserve.utils import Counter, cudaMemoryIpcHandle, EngineStage, EngineStatus
from epdserve.lifetime import LifetimeEvent, LifetimeEventType
from epdserve.tokenizer import get_tokenizer
from epdserve.block_manager import BlockManager, VisionBlockManager
from epdserve.worker import Worker
from epdserve.scheduler_context import ContextStageSchedConfig, ContextStageScheduler, get_context_stage_scheduler
from epdserve.scheduler_decoding import DecodingStageSchedConfig, DecodingStageScheduler, get_decoding_stage_scheduler

logger = init_logger(__name__)

from termcolor import colored

# Sleep for this many seconds when there is no request in ContextStageEngine.step()
# We need to sleep for a while because the whole program is a asyncio-based,
# event driven, single thread program. We save some CPU time for other coroutines.
SLEEP_WHEN_CONTEXT_NO_REQUEST = 0.003

# Sleep for this many seconds when there is no request in DecodingStageEngine.step()
SLEEP_WHEN_DECODING_NO_REQUEST = 0.003

# Sleep for this many seconds in each event loop, useful for debugging
SLEEP_IN_EACH_EVENT_LOOP = 0

# Print engine status every this many seconds
PRINT_STATUS_INTERVAL = 1


from vllm import SamplingParams
from vllm.sequence import (SequenceGroupMetadata, SequenceData, VLLM_TOKEN_ID_ARRAY_TYPE)
from array import array
from PIL import Image
import collections

class StepOutput:
    """The output of request in one step of inference.
    It contains the information of corresponding request and the generated tokens until this step.
    """

    def __init__(self, request: Request, new_token: str, new_token_id: int):
        self.request = request
        self.request_id = request.request_id
        self.prompt = request.prompt
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = request.is_finished
        self.timestamp = None

    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finished})"
        )


class StageCluster(ABC):
    @property
    def engines(self):
        return list(self.engine_map.values())

    @property
    def active_engines(self):
        return [engine for engine in self.engines if engine.operating_status==EngineStatus.ACTIVE]

    @ property
    def next_eid(self):
         last_dp, last_engine = next(reversed(self.engine_map.items()))
         return last_dp + 1

    def __init__(self, stage: EngineStage):
        self.stage = stage
        self.engine_map = collections.OrderedDict() # DP Rank: Engine

    async def wind_engine(self, index):
        # Step1: Stop taking in new requests
        last_dp, last_engine = next(reversed(self.engine_map.items()))
        last_engine.operating_status = EngineStatus.INACTIVE

        # Step 2: Transfer existing requests in waiting queue to siblings' waiting queues
        # TODO: implement migrated request distribution to siblings in GPU
        while len(last_engine.scheduler.waiting_queue) > 0 and self.stage==EngineStage.ENCODING:
            popped_req = last_engine.scheduler.waiting_queue.pop()
            self.add_request(popped_req)

        # Step 3: Wait for running and unaccepted requests to finish
        while True:
            num_waiting = len(last_engine.scheduler.waiting_queue)
            num_awaiting = len(last_engine.scheduler.unaccepted_queue)
            
            num_running = len(last_engine.scheduler.running[-1]) if len(last_engine.scheduler.running)>0 else 0
                           
            if num_waiting+num_awaiting+num_running==0:
                break
            else:
                await asyncio.sleep(1)

        # Step 4: Remove engine from this stage and update config of siblings
        popped_dp, popped_engine = self.engine_map.popitem(last=True)
        for engine in self.engines:
            engine.parallel_config.data_parallel_size -= 1

        # Step 5: Remove IPC cache handle connections in the next stage
        # self.context_cluster.unregister_kvcache_mem_handles(self.encoding_cluster.engines[E_index].parallel_config) # THIS NEEDS TO BE DONE TO BE CLEAN        

        # Step 6: Put it to sleep
        popped_engine.operating_stats = EngineStatus.SLEEP
        return popped_engine
       
    def append_engine(self, new_engine):
        # Step 1:  Update configs of new siblings
        for engine in self.engines:
            engine.parallel_config.data_parallel_size+=1

        # Step 2: Update configs of the new engine
        new_engine.parallel_config.data_parallel_size = len(self.engines) + 1
        new_engine.parallel_config.data_parallel_rank = self.next_eid

        # Step 2: Add thew new engine
        self.engine_map[new_engine.parallel_config.data_parallel_rank] = new_engine 

    async def initialize(self):
        await asyncio.gather(*[engine.initialize() for engine in self.engines])

    async def start_event_loop(self):
        await asyncio.gather(*[engine.start_event_loop() for engine in self.engines])
    
class StageEngine(ABC):
    """
    StageEngine: An LLMEngine that runs either the context stage or the decoding stage.
    
    This class is the base class for ContextStageEngine and DecodingStageEngine.
    """
    @abstractmethod
    def _get_scheduler(self) -> ContextStageScheduler | DecodingStageScheduler:
        raise NotImplementedError()
    

    def migrate_cache(self, new_inst, cache_type='kv'):
        if cache_type =='kv':
            new_inst.num_gpu_blocks = self.num_gpu_blocks 
            new_inst.num_cpu_blocks = self.num_cpu_blocks # cpu swap not be used 
            new_inst.kv_cache_mem_handles = self.kv_cache_mem_handles
            new_inst.block_manager = self.block_manager
        elif cache_type == 've':
            new_inst.num_vision_gpu_blocks = self.num_vision_gpu_blocks 
            new_inst.num_vision_cpu_blocks = self.num_vision_cpu_blocks # cpu swap not be used 
            new_inst.ve_cache_mem_handles = self.ve_cache_mem_handles
            new_inst.vision_block_manager = self.vision_block_manager
            # new_inst.vision_block_manager = VisionBlockManager(
            #     new_inst.stage,
            #     new_inst.num_vision_gpu_blocks,
            #     new_inst.num_vision_cpu_blocks,
            #     new_inst.model_config,
            #     new_inst.parallel_config,
            #     new_inst.cache_config,
            #     new_inst._remote_call_all_workers_async,
            # )
            # new_inst._init_vecache()


    def _free_request_resources(self, request_id: int) -> None:
        self.block_manager.free_blocks(request_id)
        self._remote_call_all_workers_async("clear_request_resource", request_id)
    
    def __init__(
        self,
        stage: EngineStage,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig | DecodingStageSchedConfig,
        placement_groups: List[PlacementGroup],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],   # The LLMEngine's callback function when a new StepOutput of a particular request is generated
        engine_on_new_lifetime_event_callback: Optional[Callable[[int, LifetimeEvent, bool], None]] = None,   # The LLMEngine's callback function when a new LifetimeEvent of a particular request is generated
    ):
        self.stage = stage
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.sched_config = sched_config
        self.engine_on_new_step_output_callback = engine_on_new_step_output_callback
        self.engine_on_new_lifetime_event_callback = engine_on_new_lifetime_event_callback

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )

        self.placement_groups = placement_groups
        
        # workers[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.workers = []
        self.operating_status = EngineStatus.ACTIVE
    
    async def initialize(self):
        """Initialize workers, load models and initialize k/v cache
        
        We seperate this function from __init__ because we want to run it in an async way
        to enable parallel initialization between Engines.
        """
        logger.info(f"{self.stage.name} Initializing workers")
        await self._init_workers()
        self.gpu_ids = ','.join([str(ray.get(tp_worker.self.remote("gpu_id"))) for pp_workers in self.workers for tp_worker in pp_workers])
        
        logger.info(f"{self.stage.name} Initializing models")
        await self._init_model()
        
        # if self.stage not in [EngineStage.DECODING]:
        #     logger.info(f"{self.stage.name} Initializing ve caches")
        #     await self._init_vecache()            

        # if self.stage not in [EngineStage.ENCODING]:
        #     logger.info(f"{self.stage.name} Initializing kv caches")
        #     await self._init_kvcache()
        # logger.info(f"{self.stage.name} Block manager: {self.block_manager}")

        # Always initialize all caches
        logger.info(f"{self.stage.name} Initializing ve caches")
        await self._init_vecache()            
        logger.info(f"{self.stage.name} Initializing kv caches")
        await self._init_kvcache()


    async def _init_workers(self):
        """
        for each pipeline stage, create tensor_parallel_size workers
        each worker will be assigned a GPU
        the worker will be placed in the corresponding placement group
        """

        layer_per_placement_group = self.model_config.get_num_layers() // len(self.placement_groups)
        layer_per_pp = self.model_config.get_num_layers(self.parallel_config)
        pp_per_placement_group = layer_per_placement_group // layer_per_pp
        
        pp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        
        init_handlers = []
        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []
            placement_group_index = i // pp_per_placement_group
            tp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            cur_placement_group = self.placement_groups[placement_group_index]
            for j in range(self.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(self.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j
                scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group=cur_placement_group)
                worker_id = i*self.parallel_config.tensor_parallel_size + j
                worker = Worker.options(scheduling_strategy=scheduling_strategy).remote(
                        worker_id=worker_id,
                        stage=self.stage,
                        model_config=self.model_config,
                        cache_config=self.cache_config,
                        parallel_config=tmp_parallel_config,
                        pipeline_parallel_id=pp_id,
                        tensor_parallel_id=tp_id,
                    )
                workers.append(worker)
                init_handlers.append(worker.ready.remote())
            self.workers.append(workers)
            
        await asyncio.wait(init_handlers)

    async def _init_model(self):
        """
        init model by call init_model() on all workers
        """
        handlers = self._remote_call_all_workers_async("init_model")
        await asyncio.wait(handlers)

    async def _init_vecache(self):
        """ Profile available blocks and initialize k/v cache on all workers """
        # logger.info("Profiling available blocks for VISION embeddings")
        num_gpu_blocks, num_cpu_blocks = await self.workers[0][0]._profile_num_available_blocks_encoding.remote(
            self.cache_config.block_size,
            self.cache_config.gpu_memory_utilization,
            self.cache_config.cpu_swap_space,
        )

        logger.info(f"Vision Profiling result: num_gpu_blocks: {num_gpu_blocks}, num_cpu_blocks: {num_cpu_blocks}")
        # if self.stage == EngineStage.CONTEXT:
        #     # Do not set to 0 to avoid division by 0
        #     logger.info(f"The engine performs context stage, setting num_cpu_blocks to 1")
        #     num_cpu_blocks = 1
        # logger.info("Allocating kv cache")
        ve_cache_mem_handles_1d = await asyncio.gather(*self._remote_call_all_workers_async(
            "init_vecache_and_swap", num_gpu_blocks, num_cpu_blocks
        ))
        
        # Gather the address of kv cache for block migration
        self.ve_cache_mem_handles = []
        for stage in self.workers:
            ve_cache_mem_handles = []
            for worker in stage:
                ve_cache_mem_handles.append(ve_cache_mem_handles_1d.pop(0))
            self.ve_cache_mem_handles.append(ve_cache_mem_handles)

        self.num_vision_gpu_blocks, self.num_vision_cpu_blocks = num_gpu_blocks, num_cpu_blocks
        self.vision_block_manager = VisionBlockManager(
            self.stage,
            self.num_vision_gpu_blocks,
            self.num_vision_cpu_blocks,
            self.model_config,
            self.parallel_config,
            self.cache_config,
            self._remote_call_all_workers_async,
        )

    async def _init_kvcache(self):
        """
        Profile available blocks and initialize k/v cache on all workers
        """
        # logger.info("Profiling available blocks")
        num_gpu_blocks, num_cpu_blocks = await self.workers[0][0]._profile_num_available_blocks.remote(
            self.cache_config.block_size,
            self.cache_config.gpu_memory_utilization,
            self.cache_config.cpu_swap_space,
        )

        logger.info(f"{self.stage.name} Profiling result: num_gpu_blocks: {num_gpu_blocks}, num_cpu_blocks: {num_cpu_blocks}")
        
        if self.stage == EngineStage.CONTEXT:
            logger.info(f"{self.stage.name} The engine performs context stage, setting num_cpu_blocks to 1")
            num_cpu_blocks = 1  # Do not set to 0 to avoid division by 0

        # logger.info(f"{self.stage.name} Allocating kv cache")
        kv_cache_mem_handles_1d = await asyncio.gather(*self._remote_call_all_workers_async("init_kvcache_and_swap", num_gpu_blocks, num_cpu_blocks))
        
        # Gather the address of kv cache for block migration
        self.kv_cache_mem_handles = []
        for stage in self.workers:
            kv_cache_mem_handles = []
            for worker in stage:
                kv_cache_mem_handles.append(kv_cache_mem_handles_1d.pop(0))
            self.kv_cache_mem_handles.append(kv_cache_mem_handles)

        self.num_gpu_blocks, self.num_cpu_blocks = num_gpu_blocks, num_cpu_blocks
        self.block_manager = BlockManager(
            self.stage,
            self.num_gpu_blocks,
            self.num_cpu_blocks,
            self.model_config,
            self.parallel_config,
            self.cache_config,
            self._remote_call_all_workers_async,
        )

    def _remote_call_all_workers_async(self, func_name: str, *args):
        """
        call func_name asynchronously on all workers, return the futures immediately
        """
        handlers = []
        for stage in self.workers:
            for worker in stage:
                handlers.append(getattr(worker, func_name).remote(*args))
        return handlers

    def abort_request(self, request_id: int):
        """
        abort_request: Abort one request and free its resources
        """
        # Currently there may be some race conditions here,
        # so we just do nothing
        # TODO. Implement request abortion
        logger.warn(f"Request abortion is not implemented yet")
        # return
        self.scheduler.abort_request(request_id)
        self._free_request_resources(request_id)
    
    @abstractmethod
    async def start_event_loop(self):
        raise NotImplementedError()
