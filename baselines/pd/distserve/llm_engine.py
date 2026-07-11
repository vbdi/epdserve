from typing import List, Optional, Dict, AsyncGenerator
import asyncio
import argparse
from collections import defaultdict

import ray
from ray.util.placement_group import PlacementGroup

from distserve.config import (
    ModelConfig, 
    DisaggParallelConfig,  
    CacheConfig, 
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.logger import init_logger
from distserve.request import (
    SamplingParams,
    create_request,
)
from distserve.tokenizer import get_tokenizer
from distserve.utils import Counter
from distserve.engine_common import (StepOutput)
from distserve.engine_context import ContextStageCluster
from distserve.engine_decoding import DecodingStageCluster
from distserve.lifetime import LifetimeEvent, LifetimeEventType
logger = init_logger(__name__)

from vllm.inputs import INPUT_REGISTRY
from distserve.engine_common import PRINT_STATUS_INTERVAL
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)  # For two decimal places

from nvitop import CudaDevice
from torch.utils.tensorboard import SummaryWriter
import time, os
import numpy as np


class LLMEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        context_sched_config: ContextStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig
    ):
        self.model_config = model_config
        self.disagg_parallel_config = disagg_parallel_config
        self.cache_config = cache_config
        self.context_sched_config = context_sched_config
        self.decoding_sched_config = decoding_sched_config

        # Creates input processor with multimedia support
        vllm_model_config = self.model_config.vllm_engine_config.model_config
        self.input_processor = INPUT_REGISTRY.create_input_processor(vllm_model_config)
        # Relevant files - vllm/multimodal/image.py
        # Relevant files - vllm/transformers_utils/image_processor.py

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )
        
        self.context_bridge_queue = asyncio.Queue()
        
        self.num_engines = self.disagg_parallel_config.context.data_parallel_size\
                           + self.disagg_parallel_config.decoding.data_parallel_size
        
        logger.info(f"Initializing placement group for total engines {self.num_engines}")
        placement_groups = self._init_placement_groups()

        self.context_cluster = ContextStageCluster(
            self.context_bridge_queue,
            model_config,
            disagg_parallel_config.context,
            cache_config,
            context_sched_config,
            placement_groups,
            self._on_new_step_output_callback,
            self._on_new_lifetime_event_callback
        )
        
        self.decoding_cluster = DecodingStageCluster(
            self.context_bridge_queue,
            model_config,
            disagg_parallel_config.decoding,
            cache_config,
            decoding_sched_config,
            placement_groups,
            self.context_cluster.clear_migrated_blocks_callback,
            self._on_new_step_output_callback,
            self._on_new_lifetime_event_callback
        )
        
        # request_id -> list of StepOutput
        self.request_outputs: Dict[int, asyncio.Queue[StepOutput]] = {}
        
        # request_id -> list of LifetimeEvent
        # self.request_lifetime_events: Dict[int, List[LifetimeEvent]] = {}
        self.request_lifetime_events = defaultdict(dict)
        self.engine_initialized = False
        # self.nvitop_devices = CudaDevice.all()
        self.tb_metrics = []

    def _init_placement_groups(self) -> Optional[List[PlacementGroup]]:
        # naive one GPU per engine strategy, assumes one worker per engine
        workers_per_placement_group = self.num_engines
        placement_group_list = [ { "GPU": 1 }] * workers_per_placement_group
        placement_group = ray.util.placement_group(placement_group_list, strategy="STRICT_PACK")
        ray.get(placement_group.ready(), timeout=1000)
        return [placement_group]

    async def initialize(self):
        await asyncio.gather(
            self.context_cluster.initialize(),
            self.decoding_cluster.initialize()
        )

        # TODO: I think this can be optimized by putting in a gather operation
        for engine in self.context_cluster.engines:
            await self.decoding_cluster.register_kvcache_mem_handles(engine.parallel_config, engine.kv_cache_mem_handles)

        self.engine_initialized = True

    async def start_all_event_loops(self):
        print("\n\n********* Starting LLMEngine's event loops ********** \n\n")
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before starting event loops."
        try:
            await asyncio.gather(
                self.context_cluster.start_event_loop(),
                self.decoding_cluster.start_event_loop(),
                self._start_my_event_loop()
            )
        except Exception as e:
            print(f'\n\n\Error!! Exception in {self.__class__.__name__} \n{e}\n EXITING THE PROGRAM')
            exit()

    def get_cluster_status(self):
        block_usage_index = []
        scheduler_usage_index = []

        context_scheduler_usage = []
        context_block_usage = []
        for dp_rank, engine in enumerate(self.context_cluster.engines):
            context_scheduler_usage.append(engine.get_scheduler_status())
            context_block_usage.append(engine.get_block_status())
            scheduler_usage_index.append(f'context(DP{dp_rank})')
            block_usage_index.append(f'context(DP{dp_rank})')

        decoding_scheduler_usage = []
        decoding_block_usage = []
        for dp_rank, engine in enumerate(self.decoding_cluster.engines):
            decoding_scheduler_usage.append(engine.get_scheduler_status())
            decoding_block_usage.append(engine.get_block_status())
            scheduler_usage_index.append(f'decoding(DP{dp_rank})')
            block_usage_index.append(f'decoding(DP{dp_rank})')
        return context_block_usage, decoding_block_usage, block_usage_index, \
               context_scheduler_usage, decoding_scheduler_usage, scheduler_usage_index

    def get_gpu_status(self, context_block_usage, decoding_block_usage):
        context_gpus = [int(block_usage['gpus']) for block_usage in context_block_usage]
        decoding_gpus = [int(block_usage['gpus']) for block_usage in decoding_block_usage]

        context_stats = [{
            '#gpu': f'context(#{device.nvml_index})',
            'gpu%': device.gpu_utilization(),
            'mem%': device.memory_percent()} for device in self.nvitop_devices if device.nvml_index in context_gpus]

        decoding_stats = [{
            '#gpu': f'decoding(#{device.nvml_index})',
            'gpu%': device.gpu_utilization(),
            'mem%': device.memory_percent()} for device in self.nvitop_devices if device.nvml_index in decoding_gpus]

        avg_stats = [{
            '#gpu': f'context(avg)',
            'gpu%': np.mean([stat['gpu%'] for stat in context_stats]),
            'mem%': np.mean([stat['mem%'] for stat in context_stats])

        },
        {
            '#gpu': f'decoding(avg)',
            'gpu%': np.mean([stat['gpu%'] for stat in decoding_stats]),
            'mem%': np.mean([stat['mem%'] for stat in decoding_stats])
        }]

        return context_stats, decoding_stats, avg_stats

    def log_tb_agg(self, context_block_usage, decoding_block_usage,  \
                   context_scheduler_usage, decoding_scheduler_usage, avg_gpu_stats=None):

        context_block_usage = [block_usage for (idx, block_usage) in enumerate(context_block_usage) if idx %2 ==0]
        metrics = {}
        
        metric_id = ord('A')
        context_blocks_percent = np.mean([float(block_usage['gpu'].split('%')[0]) for block_usage in context_block_usage])
        decoding_blocks_percent = np.mean([float(block_usage['gpu'].split('%')[0]) for block_usage in decoding_block_usage])
        metrics[f'{chr(metric_id)}.blocks%/B.context'] = context_blocks_percent
        metrics[f'{chr(metric_id)}.blocks%/C.decoding'] = decoding_blocks_percent

        # stat_keys = ['queuing', 'running', 'awaiting', 'exited', 'blocks']
        stat_keys = ['queuing', 'running', 'awaiting', 'exited']
        for stat in stat_keys:
            metric_id+=1
            context_stat = np.sum([float(scheduler_usage[stat]) for scheduler_usage in context_scheduler_usage])
            decoding_stat = np.sum([float(scheduler_usage[stat]) for scheduler_usage in decoding_scheduler_usage])
            metrics[f'{chr(metric_id)}.{stat}/B.context#'] = context_stat
            metrics[f'{chr(metric_id)}.{stat}/C.decoding#'] = decoding_stat

        if avg_gpu_stats is not None:
            metric_id+=1
            metrics[f'{chr(metric_id)}.gpu_util/B.context'] = avg_gpu_stats[0]['gpu%']
            metrics[f'{chr(metric_id)}.gpu_util/C.decoding'] = avg_gpu_stats[1]['gpu%']

        # Write data
        # if not hasattr(self, 'tb_writer'):
        #     self.tb_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), f'../runs/{self.model_config.run_name}'))
        #     self.global_step = -1
        #     self.tb_metrics = []

        # self.global_step += 1
        # for tag, scalar in metrics.items():
        #     self.tb_writer.add_scalar(tag, scalar, global_step=self.global_step)

        self.tb_metrics.append(metrics)

    async def _start_my_event_loop(self):
        while True:
            context_block_usage, decoding_block_usage, block_usage_index, \
            context_scheduler_usage, decoding_scheduler_usage, scheduler_usage_index = self.get_cluster_status()
            # context_gpu_stats, decoding_gpu_stats, avg_gpu_stats = self.get_gpu_status(context_block_usage, decoding_block_usage)

            # shell output
            print()
            # print(pd.DataFrame(context_gpu_stats+decoding_gpu_stats+avg_gpu_stats).set_index('#gpu').round(2))
            print(pd.DataFrame(context_block_usage + decoding_block_usage, index=block_usage_index))
            print(pd.DataFrame(context_scheduler_usage + decoding_scheduler_usage, index=scheduler_usage_index))

            ### TF metrics computation and logging ###
            self.log_tb_agg(context_block_usage, decoding_block_usage,\
                            context_scheduler_usage, decoding_scheduler_usage, avg_gpu_stats=None)

            if not hasattr(self, 'prev_time'):
                self.prev_time = time.perf_counter()
            else:
                elasped_time = time.perf_counter()-self.prev_time
                # print(f'Event loop called after {elasped_time}')
                if elasped_time > (1+0.20)*PRINT_STATUS_INTERVAL:
                    print(f'Potential chocking. Event loop called after {elasped_time}')
                # assert(elasped_time < (1+0.30)*PRINT_STATUS_INTERVAL)
                self.prev_time = time.perf_counter()
            await asyncio.sleep(PRINT_STATUS_INTERVAL)

    def _on_new_step_output_callback(self, request_id: int, step_output: StepOutput):
        """
        Called by self.context_engine or self.decoding_engine when a new output token
        is generated
        """
        step_output.timestamp = time.perf_counter()
        self.request_outputs[request_id].put_nowait(step_output)


    def _on_new_lifetime_event_callback(self, request_id: int, event: LifetimeEvent, dont_add_if_dup: bool = False):
        """
        Called by self.context_engine or self.decoding_engine when a new lifetime event
        is generated
        """
        # if dont_add_if_dup == True and self.request_lifetime_events[request_id][-1].event_type == event.event_type, don't add it
        if dont_add_if_dup and \
           len(self.request_lifetime_events[request_id]) > 0 and \
           event.event_type in self.request_lifetime_events[request_id]:
            return
        # self.request_lifetime_events[request_id].append(event)
        self.request_lifetime_events[request_id][event.event_type.value] = event
            

    def _remote_call_all_workers(self, func_name: str, *args):
        """
        call func_name on all workers, blocked until all workers finish, and return all the results
        """
        handlers = self._remote_call_all_workers_async(func_name, *args)
        return ray.get(handlers)

    def _remote_call_all_workers_async(self, func_name: str, *args):
        """
        call func_name asynchronously on all workers (context/decoding/both), return the futures immediately
        """
        handlers = self.context_engine._remote_call_all_workers_async(func_name, *args)
        handlers += self.decoding_engine._remote_call_all_workers_async(func_name, *args)
        return handlers

    async def generate(
        self,
        input: Optional[dict],
        sampling_params: SamplingParams,
        request_id: Optional[int],
        arrival_time: Optional[float] = None,
    ) -> AsyncGenerator[StepOutput, None]:
        """
        generate - Generate outputs for one request
        
        This function is intended to be used as an async generator, i.e., it can be
        used in a for loop. For example, `async for output in engine.generate(...)`
        """
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before generating."

        self._on_new_lifetime_event_callback(request_id, LifetimeEvent(LifetimeEventType.Issued))
        input['prompt_token_ids']  = self.tokenizer.encode(input['prompt'])
        model_input = self.input_processor(input)
        input['processed_prompt'] = model_input['prompt']
        input['processed_prompt_token_ids'] = model_input['prompt_token_ids']

        req = create_request(
            model_input=model_input,
            sampling_params=sampling_params,
            request_id=request_id,
            arrival_time=arrival_time if arrival_time is not None else time.perf_counter(),
        )

        self.request_outputs[req.request_id] = asyncio.Queue()
        self._on_new_lifetime_event_callback(req.request_id, LifetimeEvent(LifetimeEventType.Preprocessed))
        self.context_cluster.add_request(req)
        
        while True:
            try:
                step_output = await self.request_outputs[req.request_id].get()
            except asyncio.CancelledError:
                # The engine returns
                # Exception should be handled by the engine, not me
                return
            except GeneratorExit:
                return
            yield step_output
            if step_output.is_finished:
                break
                
        self._on_new_lifetime_event_callback(req.request_id, LifetimeEvent(LifetimeEventType.Finished))
        del self.request_outputs[req.request_id]


    def abort_request(self, request_id: int):
        self.context_engine.abort_request(request_id)
        self.decoding_engine.abort_request(request_id)
        
        
def add_engine_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-dummy-weights", action="store_true")
    
    parser.add_argument("--context-data-parallel-size", type=int, default=1)
    parser.add_argument("--context-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-tensor-parallel-size", type=int, default=1)

    parser.add_argument("--decoding-data-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-tensor-parallel-size", type=int, default=1)
    
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-blocks-per-req", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=int, default=16)
    
    parser.add_argument("--encoding-sched-policy", type=str, default="fcfs")
    parser.add_argument("--encoding-max-batch-size", type=int, default=2)
    
    parser.add_argument("--context-sched-policy", type=str, default="fcfs")
    parser.add_argument("--context-max-batch-size", type=int, default=256)
    parser.add_argument("--context-max-tokens-per-batch", type=int, default=4096)
    
    parser.add_argument("--decoding-sched-policy", type=str, default="fcfs")
    parser.add_argument("--decoding-max-batch-size", type=int, default=256)
    parser.add_argument("--decoding-max-tokens-per-batch", type=int, default=8192)
    
    parser.add_argument("--simulator-mode", action="store_true")
    parser.add_argument("--profiler-data-path", type=str, default=None)
    parser.add_argument("--gpu-mem-size-gb", type=float, default=None)
    parser.add_argument("--limit-mm-per-prompt", type=int, default=32)
    