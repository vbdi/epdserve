
import time
import copy
from typing import List
import asyncio
import ray
from ray.util.placement_group import PlacementGroup

from epdserve.config import ModelConfig, ClusterParallelConfig, CacheConfig, ContextStageSchedConfig
from epdserve.logger import init_logger
from epdserve.request import Request
from epdserve.engine_common import StepOutput
from epdserve.lifetime import LifetimeEvent, LifetimeEventType
logger = init_logger(__name__)

from epdserve.worker import Worker
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from epdserve.scheduler_encoding import EncodingStageScheduler, get_encoding_stage_scheduler
from abc import ABC
from epdserve.utils import EngineStage, EngineStatus, random_digits
from typing import Callable, List
from epdserve.request import (BatchedRequests, MigratingRequest)
from epdserve.engine_common import SLEEP_IN_EACH_EVENT_LOOP, PRINT_STATUS_INTERVAL, SLEEP_WHEN_CONTEXT_NO_REQUEST
from epdserve.utils import EngineStage
import torch
from epdserve.block_manager import VisionBlockManager
from epdserve.engine_common import *

from vllm.multimodal import MultiModalRegistry
from termcolor import colored


class EncodingStageCluster(StageCluster):
    @property
    def request_allocator(self):
        # Safeguard with modulo on the current size
        if len(self.active_engines) == 0:
            raise ValueError("active_engines is empty.")
        self._request_allocator = (self._request_allocator + 1) % len(self.active_engines)
        return self._request_allocator

    def __init__(
        self,
        encode_context_bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ClusterParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig,
        placement_groups: List[PlacementGroup],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        super().__init__(EngineStage.ENCODING)
        self.encode_context_bridge_queue=encode_context_bridge_queue
        for dp_rank in range(parallel_config.data_parallel_size): 
            logger.info(f"Initializing Encoding (DP{dp_rank}) Engine")
            tmp_parallel_config = copy.deepcopy(parallel_config)
            tmp_parallel_config.data_parallel_rank = dp_rank
            engine = EncodingStageEngine(
                encode_context_bridge_queue,
                model_config,
                tmp_parallel_config,
                cache_config,
                sched_config,
                placement_groups,
                engine_on_new_step_output_callback,
                engine_on_new_lifetime_event_callback
            )
            self.engine_map[dp_rank] = engine 
        self._request_allocator = 0

    async def initialize(self):
        await asyncio.gather(*[engine.initialize() for engine in self.engines])        

    def shard_requests(self, request: Request):
        sharded_requests = []
        images = request.vllm_seq_metadata.multi_modal_data['image']['image']
        for idx, image in enumerate(images):
            new_req = copy.copy(request)
            new_req.request_id = f'{request.request_id}_{idx+1}_{len(images)}'
            new_req.vllm_seq_metadata.multi_modal_data['image']['image'] = [image]
            new_req.is_sharded=True
            sharded_requests.append(new_req)
        return sharded_requests

    def get_next_engine(self):
        if len(self.active_engines) == 0:
            raise ValueError("No active engines available.")
        engine = self.active_engines[self.request_allocator]
        return engine

    def add_request(self, request: Request):
        if request.enable_irp==True and not request.is_sharded:
            all_requests = self.shard_requests(request)
        else:
            all_requests = [request]

        for request in all_requests:
            engine = self.get_next_engine()
            engine.scheduler.add_request(request)

    def redistribute_requests(self):
        all_requests = []
        for engine in self.engines:
            all_requests += engine.scheduler.waiting_queue
            engine.scheduler.waiting_queue = []
        for request in all_requests:
            self.add_request(request)

    async def start_event_loop(self):
        await asyncio.gather(*[engine.start_event_loop() for engine in self.engines])        

    def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest):
        """ Called when the context engine finishes migrating the blocks of the request."""
        engine = self.engine_map[migrated_request.source_parallel_config.data_parallel_rank]
        engine.clear_migrated_blocks_callback(migrated_request)

class EncodingStageEngine(StageEngine):
    def _get_scheduler(self) -> EncodingStageScheduler:
        return get_encoding_stage_scheduler(
            self.sched_config,
            self.parallel_config,
            self.vision_block_manager
        )

    def __init__(
        self,
        encode_context_bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ClusterParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig,
        placement_groups: List[PlacementGroup],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        super().__init__(
            EngineStage.ENCODING,
            model_config,
            parallel_config,
            cache_config,
            sched_config,
            placement_groups,
            engine_on_new_step_output_callback,
            engine_on_new_lifetime_event_callback
        )
        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []
        
        self.encode_context_bridge_queue = encode_context_bridge_queue

        vllm_model_config = model_config.vllm_engine_config.model_config
        mm_registry = MultiModalRegistry()
        mm_registry.init_mm_limits_per_prompt(vllm_model_config)
        self.mm_registry = mm_registry
        self.vllm_model_config = vllm_model_config
        self.gpu_ids: str = ''
        self.pls_stop_loop = asyncio.Event()
        self.is_loop_stopped = asyncio.Event()


    async def cast_to_decoding(self, to_cls, to_cluster):
        # Step 1: Remove unecessary cache 
        # await asyncio.wait(self._remote_call_all_workers_async("destruct", ['ve'])) # TODO can be async with following stuffs

        # Step 2: Initialize the new engine
        inst_to_mimic = to_cluster.engines[-1]
        new_engine = to_cls(
            context_decode_bridge_queue=inst_to_mimic.context_decode_bridge_queue, 
            model_config=self.model_config, # model config always same
            parallel_config=copy.deepcopy(inst_to_mimic.parallel_config), # TODO decoding engines may be heterogenous 
            cache_config=self.cache_config, # always same
            sched_config=inst_to_mimic.sched_config, 
            placement_groups=self.placement_groups,
            clear_migrated_blocks_callback_context=inst_to_mimic.clear_migrated_blocks_callback_context, # TODO double check? 
            engine_on_new_step_output_callback=self.engine_on_new_step_output_callback,
            engine_on_new_lifetime_event_callback=self.engine_on_new_lifetime_event_callback
        )
        new_engine.workers = self.workers
        
        handlers=[]
        for wks_in_one_PP in new_engine.workers:
            for wk in wks_in_one_PP:
                hd=wk.self.remote('stage', EngineStage.DECODING)
                handlers.append(hd) 
        await asyncio.wait(handlers)
        
        new_engine.gpu_ids = self.gpu_ids 

        # Step3: Handle cache migration
        # initialize new kv cache
        # await new_D._init_kvcache()

        ## ALL CACHE CASE
        self.migrate_cache(new_engine, 'kv')
        self.migrate_cache(new_engine, 've')

        # Step 4: Intialize the new scheduler 
        new_engine.scheduler = new_engine._get_scheduler()
        
        return new_engine


    async def cast_to_context(self, to_cls, to_cluster):
        # Step 1: Remove unecessary cache 
        # await asyncio.wait(self._remote_call_all_workers_async("destruct", ['ve'])) # TODO can be async with following stuffs

        # Step 2: Initialize the new engine
        inst_to_mimic = to_cluster.engines[-1]
        new_engine = to_cls(
            encode_context_bridge_queue=inst_to_mimic.encode_context_bridge_queue, 
            context_decode_bridge_queue=inst_to_mimic.context_decode_bridge_queue, 
            model_config=self.model_config, 
            parallel_config=copy.deepcopy(inst_to_mimic.parallel_config), # TODO prefill and decoding may have different parall config? 
            cache_config=self.cache_config, # always same
            sched_config=inst_to_mimic.sched_config,
            placement_groups=self.placement_groups, 
            clear_migrated_blocks_callback_encoding=inst_to_mimic.clear_migrated_blocks_callback_encoding,
            engine_on_new_step_output_callback=self.engine_on_new_step_output_callback,
            engine_on_new_lifetime_event_callback=self.engine_on_new_lifetime_event_callback
        )
        new_engine.workers = self.workers
        
        handlers=[]
        for wks_in_one_PP in new_engine.workers:
            for wk in wks_in_one_PP:
                hd=wk.self.remote('stage', EngineStage.CONTEXT)
                handlers.append(hd) 
        await asyncio.wait(handlers)

        new_engine.gpu_ids = self.gpu_ids         

        # Step3: Handle cache migration
        # VE cache
        # new_P.num_vision_gpu_blocks = self.num_vision_gpu_blocks 
        # new_P.num_vision_cpu_blocks = self.num_vision_cpu_blocks # cpu swap not be used 
        # new_P.ve_cache_mem_handles = self.ve_cache_mem_handles
        # new_P.vision_block_manager = self.vision_block_manager
        # # await new_P._init_vecache()

        # KV cache
        # await new_P._init_kvcache()

        ## ALL CACHE CASE
        self.migrate_cache(new_engine, 'kv')
        self.migrate_cache(new_engine, 've')

        # Step 4: Intialize the new scheduler 
        new_engine.scheduler = new_engine._get_scheduler()
        
        return new_engine

    async def initialize(self):
        await super().initialize()
        self.scheduler: EncodingStageScheduler = self._get_scheduler()

    def add_request(self, request: Request):
        self.scheduler.add_request(request)

    async def start_event_loop(self):
        async def event_loop1():
            while self.operating_status!=EngineStatus.SLEEP:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        await asyncio.gather(event_loop1())
        
    # async def stop_event_loops(self):
    #     # Sets the stop event to terminate the loops
    #     self.pls_stop_loop.set()
    #     # await self.bridge of P.join() ?
    #     await self.is_loop_stopped.wait()

    def get_scheduler_status(self):
        scheduler_satus = self.scheduler.get_status()
        return scheduler_satus

    def get_block_status(self):
        block_status = self.vision_block_manager.get_block_usage()
        block_status['gpus'] = self.gpu_ids
        return block_status

    async def _step(self):
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        
        Note2. Pipeline parallel is not tested yet
        """

        # pick next batch from scheduler
        batched_requests = self.scheduler.get_next_batch_and_pop()
        if len(batched_requests) == 0:
            # Two cases may cause len(batched_requests) == 0:
            # 1. No request in the waiting queue
            # 2. No enough free blocks (e.g. the decoding stage is too slow)
            self.batches_in_pipeline.append(batched_requests)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_CONTEXT_NO_REQUEST)
        else:
            # allocate blocks as needed
            self.vision_block_manager.allocate_blocks_batched(batched_requests)

            # Log down the lifetime event
            for request in batched_requests.requests:
                if isinstance(request.request_id, str): # Handle the IRP case
                    # TODO: fix this to make sure that it should take a max of all shards.
                    unsharded_req_id, sharded_req_idx, sharded_total_reqs = [int(val) for val in request.request_id.split('_')]
                    request_id = unsharded_req_id
                else:
                    request_id = request.request_id
                self.engine_on_new_lifetime_event_callback(
                    request_id,
                    LifetimeEvent(LifetimeEventType.EncodingBegin)
                )
                
            # push the batch into pipeline
            batched_requests.start_one_iteration(time.perf_counter())
            self.batches_in_pipeline.append(batched_requests)

            block_tables = self.vision_block_manager.get_partial_block_table(batched_requests.get_request_ids())
            remote_calls = self._remote_call_all_workers_async(
                "step_encoding",
                batched_requests.get_request_ids(),
                batched_requests,
                block_tables
            )
            
            pp_size = self.parallel_config.pipeline_parallel_size
            tp_size = self.parallel_config.tensor_parallel_size
            # only the leader of the last stage return valid output, i.e., generated tokens ids
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                # No request in the batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                vision_embeddings_lens = await self.batches_ret_futures[0]

                end_time = time.perf_counter()

                finished_batch = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(None, None, end_time)
                
                self.scheduler.on_finish_requests(finished_batch)
                
                for request in finished_batch.requests:
                    if isinstance(request.request_id, str): # Handle the IRP case
                        # TODO: fix this to make sure that it should take a max of all shards.
                        unsharded_req_id, sharded_req_idx, sharded_total_reqs = [int(val) for val in request.request_id.split('_')]
                        request_id = unsharded_req_id
                    else:
                        request_id = request.request_id
                    self.engine_on_new_lifetime_event_callback(
                        request_id,
                        LifetimeEvent(LifetimeEventType.EncodingEnd)
                    )
                # Cannot free blocks now! The decoding stage may still need them!

                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
                
                # Inform the user that the request has finished the context stage
                for request in finished_batch.requests:
                    if not request.is_finished:
                        # Push the request into the bridge queue if it is not finished
                        migrating_req = MigratingRequest(req = request, 
                                                         block_indexes = self.vision_block_manager.get_block_table(request.request_id), 
                                                         source_parallel_config = self.parallel_config)

                        self.encode_context_bridge_queue.put_nowait(migrating_req) # This won't panic because the queue is unbounded
                    else:
                        self._free_request_resources(request.request_id)
                # logger.info(colored(f"(DONE) ENCODING batch of length {self.encode_context_bridge_queue.qsize()}. Time taken {time.perf_counter() - start_time}s", 'red'))
    
    def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest) -> None:
        """ Called when the context engine finishes migrating the blocks of the request. """
        self._free_request_resources(migrated_request.req.request_id)
        self.scheduler.on_request_migrated(migrated_request)

    def _free_request_resources(self, request_id: int):
        self.vision_block_manager.free_blocks(request_id) # clear vision cache blocks
