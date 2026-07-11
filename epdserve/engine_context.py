from epdserve.engine_common import *
from epdserve.config import ClusterParallelConfig


class ContextStageCluster(StageCluster):
    def __init__(
        self,
        encode_context_bridge_queue: asyncio.Queue[MigratingRequest],
        context_decode_bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ClusterParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig,
        placement_groups: List[PlacementGroup],
        clear_migrated_blocks_callback_encoding: Callable[[Request], None],        
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        super().__init__(EngineStage.CONTEXT)
        self.encode_context_bridge_queue=encode_context_bridge_queue
        self.context_decode_bridge_queue=context_decode_bridge_queue
        for dp_rank in range(parallel_config.data_parallel_size):  
            logger.info(f"Initializing Context (DP{dp_rank}) Engine")
            tmp_parallel_config = copy.deepcopy(parallel_config)
            tmp_parallel_config.data_parallel_rank = dp_rank
            engine = ContextStageEngine(
                encode_context_bridge_queue,
                context_decode_bridge_queue,
                model_config,
                tmp_parallel_config,
                cache_config,
                sched_config,
                placement_groups,
                clear_migrated_blocks_callback_encoding,
                engine_on_new_step_output_callback,
                engine_on_new_lifetime_event_callback
            )
            self.engine_map[dp_rank] = engine 

    async def initialize(self):
        await asyncio.gather(*[engine.initialize() for engine in self.engines])

    async def register_kvcache_mem_handles(
        self,
        encoding_parallel_config: ClusterParallelConfig,
        ve_cache_ipc_mem_handles: List[List[cudaMemoryIpcHandle]]
    ):
        await asyncio.gather(*[engine.register_kvcache_mem_handles(encoding_parallel_config, ve_cache_ipc_mem_handles) for engine in self.engines]) 

    async def start_event_loop(self):
        await asyncio.gather(*[engine.start_event_loop() for engine in self.engines])        

    async def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest):
        """ Called when the context engine finishes migrating the blocks of the request."""
        engine:ContextStageEngine = self.engine_map[migrated_request.source_parallel_config.data_parallel_rank] 
        await engine.clear_migrated_blocks_callback(migrated_request)



class ContextStageEngine(StageEngine):
    def _get_scheduler(self) -> ContextStageScheduler:
        return get_context_stage_scheduler(
            self.sched_config,
            self.parallel_config,
            self.block_manager,
            self.vision_block_manager,
            self._migrate_blocks_encoding,
            self.encode_context_bridge_queue
        )
    
    def __init__(
        self,
        encode_context_bridge_queue: asyncio.Queue[MigratingRequest],
        context_decode_bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig,
        placement_groups: List[PlacementGroup],
        clear_migrated_blocks_callback_encoding: Callable[[Request], None],        
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        super().__init__(
            EngineStage.CONTEXT,
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
        self.context_decode_bridge_queue = context_decode_bridge_queue
        # this is  for the encoding engine to clear the vision embeddings that has been migrated
        self.clear_migrated_blocks_callback_encoding = clear_migrated_blocks_callback_encoding
        self.pls_stop_loop = asyncio.Event()
        self.is_loop_stopped = asyncio.Event()


    async def initialize(self):
        await super().initialize()
        self.scheduler: ContextStageScheduler = self._get_scheduler()
        # logger.info(f"{self.stage.name} Scheduler: {self.scheduler}")

    async def cast_to_encoding(self, to_cls, to_cluster):
        # Step 1: Remove unecessary cache 
        # await asyncio.wait(self._remote_call_all_workers_async("destruct", ['kv']))

        # Step 2: Initialize the new engine
        inst_to_mimic = to_cluster.engines[-1]
        new_engine = to_cls(
            encode_context_bridge_queue=inst_to_mimic.encode_context_bridge_queue, 
            model_config=self.model_config, # model config always same
            parallel_config=copy.deepcopy(inst_to_mimic.parallel_config), # TODO decoding engines may be heterogenous 
            cache_config=self.cache_config, # always same
            sched_config=inst_to_mimic.sched_config, 
            placement_groups=self.placement_groups,
            engine_on_new_step_output_callback=self.engine_on_new_step_output_callback,
            engine_on_new_lifetime_event_callback=self.engine_on_new_lifetime_event_callback
        )
        new_engine.workers = self.workers

        handlers=[]
        for wks_in_one_PP in new_engine.workers:
            for wk in wks_in_one_PP:
                hd=wk.self.remote('stage', EngineStage.ENCODING)
                handlers.append(hd) 
        await asyncio.wait(handlers)
    
        new_engine.gpu_ids = self.gpu_ids 

        # Step3: Handle cache migration 
        # await new_engine._init_vecache()
        # new_engine.num_vision_gpu_blocks = self.num_vision_gpu_blocks 
        # new_engine.num_vision_cpu_blocks = self.num_vision_cpu_blocks # cpu swap not be used 
        # new_engine.ve_cache_mem_handles = self.ve_cache_mem_handles
        # new_engine.vision_block_manager = self.vision_block_manager
        
        ## ALL CACHE CASE.
        self.migrate_cache(new_engine, 'kv')
        self.migrate_cache(new_engine, 've')

        # Step 4: Intialize the new scheduler 
        new_engine.scheduler = new_engine._get_scheduler()
        
        return new_engine


    async def cast_to_decoding(self, to_cls, to_cluster):
        # Step 1: Remove unecessary cache 
        pass # None

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
        # # handle kv cache migration
        # new_D.num_gpu_blocks = self.num_gpu_blocks 
        # new_D.num_cpu_blocks = self.num_cpu_blocks # cpu swap not be used 
        # new_D.kv_cache_mem_handles = self.kv_cache_mem_handles
        # new_D.block_manager = self.block_manager

        ## ALL CACHE CASE
        self.migrate_cache(new_engine, 'kv')
        self.migrate_cache(new_engine, 've')  

        # Step 4: Intialize the new scheduler 
        new_engine.scheduler = new_engine._get_scheduler()
        
        return new_engine

    async def register_kvcache_mem_handles(
        self,
        encoding_parallel_config: ClusterParallelConfig,
        ve_cache_ipc_mem_handles: List[List[cudaMemoryIpcHandle]]
    ):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """
        await asyncio.wait(self._remote_call_all_workers_async(
            "register_kvcache_mem_handles_encoding",
            encoding_parallel_config,
            ve_cache_ipc_mem_handles
        ))

    async def start_event_loop(self):
        async def event_loop2():
            while self.operating_status!=EngineStatus.SLEEP:
                await self._step()

                # proactive request migraion
                if self.operating_status == EngineStatus.ACTIVE:
                    await self.scheduler.post_process()

                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        await asyncio.gather(event_loop2())

    def get_scheduler_status(self):
        scheduler_satus = self.scheduler.get_status()
        return scheduler_satus

    def get_block_status(self, vision=False):
        if vision:
            block_status = self.vision_block_manager.get_block_usage()
        else:
            block_status = self.block_manager.get_block_usage()
        block_status['gpus'] = self.gpu_ids
        return block_status

    def add_request(self, request: Request):
        self.scheduler.add_request(request)
    
    def _free_request_resources(self, request_id: int):
        self.vision_block_manager.free_blocks(request_id) # clear vision cache blocks
        super()._free_request_resources(request_id)
        
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
            self.block_manager.allocate_blocks_batched(batched_requests)
            
            # Log down the lifetime event
            for request in batched_requests.requests:
                self.engine_on_new_lifetime_event_callback(request.request_id,
                                                           LifetimeEvent(LifetimeEventType.ContextBegin))
                
            # push the batch into pipeline
            batched_requests.start_one_iteration(time.perf_counter())
            self.batches_in_pipeline.append(batched_requests)


            block_tables_all = self.block_manager.get_partial_block_table(batched_requests.get_request_ids())

            seq_group_metadata_list: List[SequenceGroupMetadata] = []

            for req_idx, request in enumerate(batched_requests.requests):
                seq_group_metadata = request.vllm_seq_metadata
                req_id = request.request_id
                seq_group_metadata.block_tables = {req_id:block_tables_all[req_idx]}
                seq_group_metadata.multi_modal_data = None
                seq_group_metadata_list.append(seq_group_metadata)

            vision_embeddings = self.vision_block_manager.get_partial_block_table(batched_requests.get_request_ids())

            remote_calls = self._remote_call_all_workers_async(
                "step_llm",
                batched_requests.get_request_ids(),
                seq_group_metadata_list,
                vision_embeddings
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
                generated_tokens_ids = await self.batches_ret_futures[0]
                    
                end_time = time.perf_counter()
                generated_tokens = []
                for gen_token_id in generated_tokens_ids:
                    try:
                        token = self.tokenizer.decode(gen_token_id)
                    except Exception as e:
                        print(f"(context) Warning: Cannot decode token with id {gen_token_id}. Error: {e}")
                        token = ""
                    generated_tokens.append(token)

                finished_batch = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(
                    generated_tokens, generated_tokens_ids, end_time
                )
                
                self.scheduler.on_finish_requests(finished_batch)
                
                for request, new_token, new_token_id in zip(
                    finished_batch.requests, generated_tokens, generated_tokens_ids
                ):
                    step_output = StepOutput(request, new_token, new_token_id)
                    self.engine_on_new_lifetime_event_callback(
                        request.request_id,
                        LifetimeEvent(LifetimeEventType.ContextEnd)
                    )
                    self.engine_on_new_step_output_callback(
                        request.request_id,
                        step_output
                    )

                # Cannot free blocks now! The decoding stage may still need them!

                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
                
                # Inform the user that the request has finished the context stage
                for request in finished_batch.requests:
                    if not request.is_finished:
                        # Push the request into the bridge queue if it is not finished
                        migrating_req = MigratingRequest(
                            req=request,
                            block_indexes=self.block_manager.get_block_table(request.request_id),
                            source_parallel_config=self.parallel_config,
                        )
                        self.context_decode_bridge_queue.put_nowait(migrating_req) # This won't panic because the queue is unbounded
                    else:
                        self._free_request_resources(request.request_id)
                # logger.info(colored(f"(DONE) PREFILLING batch of length {len(batched_requests)}. Time taken {time.perf_counter() - start_time}s", 'blue'))

    async def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest):
        """ Called when the decoding engine finishes migrating the blocks of the request. """
        self._free_request_resources(migrated_request.req.request_id)
        self.scheduler.on_request_migrated(migrated_request)

    async def _migrate_blocks_encoding(self, migrating_req: MigratingRequest) -> None:
        self.vision_block_manager.allocate_blocks(migrating_req.req)
        target_block_indexes = self.vision_block_manager.get_block_table(migrating_req.req.request_id)
        assert len(target_block_indexes) == len(migrating_req.block_indexes)

        if isinstance(migrating_req.req.request_id, str): # Handle the IRP case
            # TODO: fix this to make sure that it should take a max of all shards.
            unsharded_req_id, sharded_req_idx, sharded_total_reqs = [int(val) for val in migrating_req.req.request_id.split('_')]
            request_id = unsharded_req_id
        else:
            request_id = migrating_req.req.request_id

        self.engine_on_new_lifetime_event_callback(
            request_id,
            LifetimeEvent(LifetimeEventType.EncodingMigrationBegin)
        )
        
        await asyncio.wait(self._remote_call_all_workers_async(
            "migrate_blocks_encoding",
            migrating_req.block_indexes, # encoding block indices
            migrating_req.source_parallel_config,
            target_block_indexes # context block indices
        ))
        self.engine_on_new_lifetime_event_callback(
            request_id,
            LifetimeEvent(LifetimeEventType.EncodingMigrationEnd)
        )
    
        # Clear the blocks on the encoding engine's side
        self.clear_migrated_blocks_callback_encoding(migrating_req)

            

