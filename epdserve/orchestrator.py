from typing import List, Optional, Dict, DefaultDict, AsyncGenerator
import asyncio
import argparse
from collections import defaultdict, deque

import ray
from ray.util.placement_group import PlacementGroup

from epdserve.config import (
    ModelConfig, 
    DisaggParallelConfig,  
    CacheConfig, 
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from epdserve.utils import MyGeneralQueue
from epdserve.logger import init_logger
from epdserve.request import (
    SamplingParams,
    create_request,
)
from epdserve.tokenizer import get_tokenizer
from epdserve.engine_common import (StepOutput)
from epdserve.engine_context import ContextStageCluster, ContextStageEngine
from epdserve.engine_decoding import DecodingStageCluster, DecodingStageEngine
from epdserve.lifetime import LifetimeEvent, LifetimeEventType
logger = init_logger(__name__)

from epdserve.engine_encoding import EncodingStageCluster, EncodingStageEngine
from epdserve.engine_common import StageEngine
from epdserve.utils import EngineStage, EngineStatus

from epdserve.scheduler_encoding import EncodingStageScheduler
from vllm.inputs import INPUT_REGISTRY
from epdserve.engine_common import PRINT_STATUS_INTERVAL
import pandas as pd

from nvitop import CudaDevice
from torch.utils.tensorboard import SummaryWriter
import time, os
import numpy as np

from epdserve.resource_allocation import (LoadEstimator,
                                           ResourceAllocatorOnce,
                                           ResourceAllocatorRuntime,
                                           ResourceAllocatorSimulator)

class EPDOrchestrator:
    def __init__(
        self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        encoding_sched_config: EncodingStageScheduler,
        context_sched_config: ContextStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.encoding_sched_config = encoding_sched_config
        self.context_sched_config = context_sched_config
        self.decoding_sched_config = decoding_sched_config

        # Creates input processor with multimedia support
        # Relevant files - vllm/multimodal/image.py
        # Relevant files - vllm/transformers_utils/image_processor.py
        vllm_model_config = self.model_config.vllm_engine_config.model_config
        self.input_processor = INPUT_REGISTRY.create_input_processor(vllm_model_config)

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )
        
        self.encode_context_bridge_queue = MyGeneralQueue() 
        self.context_decode_bridge_queue = MyGeneralQueue()
        # self.encode_context_bridge_queue = asyncio.Queue()
        # self.context_decode_bridge_queue = asyncio.Queue()
        
        self.num_engines = disagg_parallel_config.encoding.data_parallel_size\
                           + disagg_parallel_config.context.data_parallel_size\
                           + disagg_parallel_config.decoding.data_parallel_size
        
        logger.info(f"Initializing placement group for total engines {self.num_engines}")
        placement_groups = self._init_placement_groups()

        self.encoding_cluster = EncodingStageCluster(
            self.encode_context_bridge_queue,
            model_config,
            disagg_parallel_config.encoding,
            cache_config,
            encoding_sched_config,
            placement_groups,
            self._on_new_step_output_callback,
            self._on_new_lifetime_event_callback
        )

        self.context_cluster = ContextStageCluster(
            self.encode_context_bridge_queue,
            self.context_decode_bridge_queue,
            model_config,
            disagg_parallel_config.context,
            cache_config,
            context_sched_config,
            placement_groups,
            self.encoding_cluster.clear_migrated_blocks_callback,
            self._on_new_step_output_callback,
            self._on_new_lifetime_event_callback
        )
        
        self.decoding_cluster = DecodingStageCluster(
            self.context_decode_bridge_queue,
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
        self.load_estimator = LoadEstimator(self.encoding_cluster,
                                            self.context_cluster,
                                            self.decoding_cluster)

    def _init_placement_groups(self) -> Optional[List[PlacementGroup]]:
        # naive one GPU per engine strategy, assumes one worker per engine
        workers_per_placement_group = self.num_engines
        placement_group_list = [ { "GPU": 1 }] * workers_per_placement_group
        placement_group = ray.util.placement_group(placement_group_list, strategy="STRICT_PACK")
        ray.get(placement_group.ready(), timeout=1000)
        return [placement_group]

    async def initialize(self):
        await asyncio.gather(
            self.encoding_cluster.initialize(),
            self.context_cluster.initialize(),
            self.decoding_cluster.initialize()
        )

        # TODO: I think this can be optimized by putting in a gather operation
        for engine in self.encoding_cluster.engines:
            await self.context_cluster.register_kvcache_mem_handles(engine.parallel_config, engine.ve_cache_mem_handles)

        for engine in self.context_cluster.engines:
            await self.decoding_cluster.register_kvcache_mem_handles(engine.parallel_config, engine.kv_cache_mem_handles)

        self.engine_initialized = True


    async def start_all_event_loops(self):
        print("\n\n********* Starting EPDOrchestrator's event loops ********** \n\n")
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before starting event loops."
        try:
            await asyncio.gather(
                self.encoding_cluster.start_event_loop(),
                self.context_cluster.start_event_loop(),
                self.decoding_cluster.start_event_loop(),
                self.load_estimator.start_event_loop(),
                self.start_event_loop()
            )
        except Exception as e:
            print(f'\n\n\Error!! Exception in {self.__class__.__name__} \n{e}\n EXITING THE PROGRAM')
            exit()


    async def start_event_loop(self):
        while True:
            if not hasattr(self, 'prev_time'):
                self.prev_time = time.perf_counter()
            else:
                elasped_time = time.perf_counter()-self.prev_time
                # print(f'Event loop called after {elasped_time}')
                if elasped_time > (1+0.20)*PRINT_STATUS_INTERVAL:
                    print(f'Potential chocking. Event loop called after {round(elasped_time,2)}s')
                # assert(elasped_time < (1+0.30)*PRINT_STATUS_INTERVAL)
                self.prev_time = time.perf_counter()
            await asyncio.sleep(PRINT_STATUS_INTERVAL)


    async def generate(
        self,
        input: Optional[dict],
        sampling_params: SamplingParams,
        request_id: Optional[int],
        arrival_time: Optional[float] = None,
        **kwargs        
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
            **kwargs              
        )

        self.request_outputs[req.request_id] = asyncio.Queue()
        self._on_new_lifetime_event_callback(req.request_id, LifetimeEvent(LifetimeEventType.Preprocessed))
        self.encoding_cluster.add_request(req)
        
        while True:
            try:
                step_output = await self.request_outputs[req.request_id].get()
            except asyncio.CancelledError:
                return
            except GeneratorExit:
                return
            yield step_output
            if step_output.is_finished:
                break
                
        self._on_new_lifetime_event_callback(req.request_id, LifetimeEvent(LifetimeEventType.Finished))
        self.load_estimator.add_timing_data(self.request_lifetime_events[req.request_id]) 


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

        if isinstance(request_id, str):
            unsharded_req_id, sharded_req_idx, sharded_total_reqs = [int(val) for val in request_id.split('_')]
            self.request_lifetime_events[unsharded_req_id][f'{event.event_type.value}_{sharded_req_idx}'] = event
        else:
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

    def abort_request(self, request_id: int):
        self.context_engine.abort_request(request_id)
        self.decoding_engine.abort_request(request_id)


class DynamicEPDOrchestrator(EPDOrchestrator):
    def __init__(self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        encoding_sched_config: EncodingStageScheduler,
        context_sched_config: ContextStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig):
        super().__init__(model_config,
                         disagg_parallel_config,
                         cache_config,
                         encoding_sched_config,
                         context_sched_config,
                         decoding_sched_config)
        self.migration_done = False
        # self.resource_allocator = ResourceAllocatorOnce(self.load_estimator)
        self.resource_allocator = ResourceAllocatorRuntime(self.load_estimator)

    async def start_all_event_loops(self):
        print("\n\n********* Starting Dynamic EPDOrchestrator's event loops ********** \n\n")
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before starting event loops."
        try:
            await asyncio.gather(
                self.encoding_cluster.start_event_loop(),
                self.context_cluster.start_event_loop(),
                self.decoding_cluster.start_event_loop(),
                self.load_estimator.start_event_loop(),
                self.resource_allocator.start_event_loop(),
                self.start_event_loop(),
                self.engine_migration_loop()
            )

        except Exception as e:
            print(f'\n\n\Error!! Exception in {self.__class__.__name__} \n{e}\n EXITING THE PROGRAM')
            exit()

    async def engine_migration_loop(self):
        while(True):
            migration_request = self.resource_allocator.dequeue()
            if migration_request is not None:
                await self.handle_migration_request(migration_request)
            await asyncio.sleep(2)

    async def handle_migration_request(self, migration_request):
        if not self.should_migrate(migration_request[0]):
            print(f"\n\nMigration {migration_request} termnated as {migration_request[0]} stage has only 1 engines!!\n\n")
            return

        print(f'\n\nEngine Migration Started {migration_request}. Status -> E:{len(self.encoding_cluster.engines)} | P:{len(self.context_cluster.engines)} | D:{len(self.decoding_cluster.engines)}\n\n')
        start_time = time.time()
        if migration_request == 'E->P':
            await self.migrate_E_to_P(-1)
        elif migration_request == 'E->D':
            await self.migrate_E_to_D(-1)
        elif migration_request == 'P->D':
            await self.migrate_P_to_D(-1)
        elif migration_request == 'P->E':
            await self.migrate_P_to_E(-1) 
        elif migration_request == 'D->P':
            await self.migrate_D_to_P(-1)
        elif migration_request == 'D->E':
            await self.migrate_D_to_E(-1)
        else:
            return "Invalid migration request"        

        print(f'\n\nEngine Migration Complete {migration_request}. Status -> E:{len(self.encoding_cluster.engines)} | P:{len(self.context_cluster.engines)} | D:{len(self.decoding_cluster.engines)}. Time Taken {round(time.time()-start_time,3)}s\n\n')
        # print(f'\n\n\n####Transfering Engine .Requests done {np.sum([len(engine.scheduler.transferred) for engine in self.encoding_cluster.engines])}')

    def should_migrate(self, from_stage):
        if from_stage == 'E':
            return len(self.encoding_cluster.engines)>1
        elif from_stage == 'P':
            return len(self.context_cluster.engines)>1
        elif from_stage == 'D':
            return len(self.decoding_cluster.engines)>1
        else:
            return "Invalid stage" 

    async def migrate_D_to_P(self, index):
        # Step 1: Remove engine from the from_stage
        old_eng = await self.decoding_cluster.wind_engine(index)
        # for P_engine in self.context_cluster.engines:
        #     self.decoding_cluster.engines[D_index].unregister_kvcache_mem_handles(P_engine.parallel_config)
        
        # Step 2: Cast to the new the new stage
        new_eng = await old_eng.cast_to_context(ContextStageEngine, self.context_cluster)

        # Step 3: Append the new engine to the to cluster
        self.context_cluster.append_engine(new_eng)

        # Step 4: Register kv cache memory handles for the new engine. 1. connect E <- Ps 2. connect P <- Ds)
        await self.decoding_cluster.register_kvcache_mem_handles(new_eng.parallel_config, new_eng.kv_cache_mem_handles)
        for engine in self.encoding_cluster.engines:
            await new_eng.register_kvcache_mem_handles(engine.parallel_config, engine.ve_cache_mem_handles)

        # Step 5: Start the new engine to take new requests
        new_eng.operating_stats = EngineStatus.ACTIVE
        task = asyncio.create_task(new_eng.start_event_loop())

        return task


    async def migrate_E_to_P(self, index):
        # Step 1: Remove engine from the from_stage
        old_eng = await self.encoding_cluster.wind_engine(index)

        # Step 2: Cast to the new the new stage
        new_eng = await old_eng.cast_to_context(ContextStageEngine, self.context_cluster)

        # Step 3: Append the new engine to the to cluster
        self.context_cluster.append_engine(new_eng) 
        print('Engine created')

        # Step 4: Register kv cache memory handles for the new engine. 1. connect P <- Ds and connect Es -< P
        await self.decoding_cluster.register_kvcache_mem_handles(new_eng.parallel_config, new_eng.kv_cache_mem_handles)
        for engine in self.encoding_cluster.engines:
            await new_eng.register_kvcache_mem_handles(engine.parallel_config, engine.ve_cache_mem_handles)

        # Step 5: # Start the new engine to take new requests
        new_eng.operating_stats = EngineStatus.ACTIVE
        task = asyncio.create_task(new_eng.start_event_loop())
        return task


    async def migrate_E_to_D(self, index):
        # Step 1: Remove engine from the from_stage
        old_eng = await self.encoding_cluster.wind_engine(index) 

        # Step 2: Cast to the new the new stage
        new_eng = await old_eng.cast_to_decoding(DecodingStageEngine, self.decoding_cluster) # TODO what if 0 D engine? 
            
        # Step 3: Append the new engine to the to cluster
        self.decoding_cluster.append_engine(new_eng)

        # Step 4: Register kv cache memory handles for the new engine (connect Ps <- D)
        for idx, P_engine in enumerate(self.context_cluster.engines): 
            await new_eng.register_kvcache_mem_handles(P_engine.parallel_config, P_engine.kv_cache_mem_handles)

        # Step 5: Start the new engine to take new requests
        new_eng.operating_stats = EngineStatus.ACTIVE
        task = asyncio.create_task(new_eng.start_event_loop())
        # await new_D.start_event_loop()
        return task 


    async def migrate_P_to_D(self, index): 
        # Step 1: Remove engine from the from_stage
        old_eng = await self.context_cluster.wind_engine(index) 
        # Optional TODO: Better to clean it up
        # self.decoding_cluster.unregister_kvcache_mem_handles(old_eng.parallel_config)

        # Step 2: Cast to the new the new stage
        new_eng = await old_eng.cast_to_decoding(DecodingStageEngine, self.decoding_cluster) # TODO what if 0 D engine? 

        # Step 3: Append the new engine to the to cluster
        self.decoding_cluster.append_engine(new_eng)

        # Step 4: Register kv cache memory handles for the new engine (connect Ps <- D)
        for idx, P_engine in enumerate(self.context_cluster.engines): 
            await new_eng.register_kvcache_mem_handles(P_engine.parallel_config, P_engine.kv_cache_mem_handles)

        # Step 5: Start the new engine to take new requests
        new_eng.operating_stats = EngineStatus.ACTIVE
        task = asyncio.create_task(new_eng.start_event_loop())

        return task


    async def migrate_P_to_E(self, index): 
        # Step 1: Remove engine from the from_stage
        old_eng = await self.context_cluster.wind_engine(index) 
        # Optional TODO: Better to clean it up
        # self.decoding_cluster.unregister_kvcache_mem_handles(old_eng.parallel_config)

        # Step 2: Cast to the new the new stage
        new_eng = await old_eng.cast_to_encoding(EncodingStageEngine, self.encoding_cluster)
            
        # Step 3: Append the new engine to the to cluster
        self.encoding_cluster.append_engine(new_eng)

        # Step 4: Register kv cache memory handles for the new engine (connect E <- Ps )
        await self.context_cluster.register_kvcache_mem_handles(new_eng.parallel_config, new_eng.ve_cache_mem_handles)        

        # Step 5: Start the new engine to take new requests
        new_eng.operating_stats = EngineStatus.ACTIVE
        task = asyncio.create_task(new_eng.start_event_loop())

        # Step 6: Redistribute requests so that new engine can get work
        self.encoding_cluster.redistribute_requests()
        return task


    async def migrate_D_to_E(self, index): 
        # Step 1: Remove engine from the from_stage
        old_eng = await self.decoding_cluster.wind_engine(index) 
        # Optional TODO: Better to clean it up
        # for P_engine in self.context_cluster.engines:
        #     self.decoding_cluster.engines[D_index].unregister_kvcache_mem_handles(P_engine.parallel_config)

        # Step 2: Cast to the new the new stage
        new_eng = await old_eng.cast_to_encoding(EncodingStageEngine, self.encoding_cluster)

        # Step 3: Append the new engine to the to cluster
        self.encoding_cluster.append_engine(new_eng)

        # Step 4: Register kv cache memory handles for the new engine (connect E <- Ps )
        await self.context_cluster.register_kvcache_mem_handles(new_eng.parallel_config, new_eng.ve_cache_mem_handles)        

        # Step 5: Start the new engine to take new requests
        new_eng.operating_stats = EngineStatus.ACTIVE
        task = asyncio.create_task(new_eng.start_event_loop())

        # Step 6: Redistribute requests so that new engine can get work
        self.encoding_cluster.redistribute_requests()
        return task

def add_orchestrator_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-dummy-weights", action="store_true")
    
    parser.add_argument("--encoding-data-parallel-size", type=int, default=1)
    parser.add_argument("--encoding-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--encoding-tensor-parallel-size", type=int, default=1)

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

    parser.add_argument("--intra-request-dp", type=int, default=0)
    parser.add_argument("--limit-mm-per-prompt", type=int, default=32)
    