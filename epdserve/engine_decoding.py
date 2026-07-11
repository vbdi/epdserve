from epdserve.engine_common import *
from epdserve.config import ClusterParallelConfig

class DecodingStageCluster(StageCluster):
    def __init__(
        self,
        context_decode_bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: DecodingStageSchedConfig,
        placement_groups: List[PlacementGroup],
        clear_migrated_blocks_callback_context: Callable[[Request], None],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        super().__init__(EngineStage.DECODING)
        for dp_rank in range(parallel_config.data_parallel_size):  
            logger.info(f"Initializing Decoding (DP{dp_rank}) Engine")
            tmp_parallel_config = copy.deepcopy(parallel_config)
            tmp_parallel_config.data_parallel_rank = dp_rank
            engine = DecodingStageEngine(
                context_decode_bridge_queue,
                model_config,
                tmp_parallel_config,
                cache_config,
                sched_config,
                placement_groups,
                clear_migrated_blocks_callback_context,
                engine_on_new_step_output_callback,
                engine_on_new_lifetime_event_callback
            )
            self.engine_map[dp_rank]=engine           

    async def initialize(self):
        await asyncio.gather(*[engine.initialize() for engine in self.engines])

    async def register_kvcache_mem_handles(
        self,
        context_parallel_config: ClusterParallelConfig,
        kv_cache_ipc_mem_handles: List[List[cudaMemoryIpcHandle]]
    ):
        await asyncio.gather(*[engine.register_kvcache_mem_handles(context_parallel_config, kv_cache_ipc_mem_handles) for engine in self.engines]) 

    async def start_event_loop(self):
        await asyncio.gather(*[engine.start_event_loop() for engine in self.engines])        

    async def unregister_kvcache_mem_handles(
        self, 
        context_parallel_config: ClusterParallelConfig,
    ):
        # cluster.unregister_kvcache_mem_handles: dis-connect [D1 D2 ... ] <- P
        # D_engine.unregister_kvcache_mem_handles: dis-connect a D <- P 
        await asyncio.gather(*[engine.unregister_kvcache_mem_handles(context_parallel_config) for engine in self.engines]) 
    

class DecodingStageEngine(StageEngine):
    def _get_scheduler(self) -> DecodingStageScheduler:
        return get_decoding_stage_scheduler(
            self.sched_config,
            self.parallel_config,
            self.block_manager,
            self._migrate_blocks_context,
            self.context_decode_bridge_queue
        )
        
    def __init__(
        self,
        context_decode_bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: DecodingStageSchedConfig,
        placement_groups: List[PlacementGroup],
        clear_migrated_blocks_callback_context: Callable[[Request], None],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        super().__init__(
            EngineStage.DECODING,
            model_config,
            parallel_config,
            cache_config,
            sched_config,
            placement_groups,
            engine_on_new_step_output_callback,
            engine_on_new_lifetime_event_callback
        )
        
        self.context_decode_bridge_queue = context_decode_bridge_queue
        self.clear_migrated_blocks_callback_context = clear_migrated_blocks_callback_context
        
        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline = []
        self.batches_ret_futures = []
        self.pls_stop_loop = asyncio.Event()
        self.is_loop_stopped = asyncio.Event()

    async def initialize(self):
        await super().initialize()
        self.scheduler: DecodingStageScheduler = self._get_scheduler()
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


    async def cast_to_context(self, to_cls, to_cluster):
        # Step 1: Remove unecessary cache 
        pass # None

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
        # # handle KV cache migration
        # new_P.num_gpu_blocks = self.num_gpu_blocks 
        # new_P.num_cpu_blocks = self.num_cpu_blocks # cpu swap not be used 
        # new_P.kv_cache_mem_handles = self.kv_cache_mem_handles
        # new_P.block_manager = self.block_manager

        # # intiiatlize new VE cache
        # await new_P._init_vecache()

        ## ALL CACHE CASE
        self.migrate_cache(new_engine, 'kv')
        self.migrate_cache(new_engine, 've')

        # Step 4: Intialize the new scheduler 
        new_engine.scheduler = new_engine._get_scheduler()
        
        return new_engine


    async def unregister_kvcache_mem_handles(self,  context_parallel_config: ClusterParallelConfig):
        await asyncio.wait(self._remote_call_all_workers_async("unregister_kvcache_mem_handles_context",  context_parallel_config))

    async def register_kvcache_mem_handles(
        self,
        context_parallel_config: ParallelConfig,
        kv_cache_mem_handles: List[List[cudaMemoryIpcHandle]]
    ):
        await asyncio.wait(self._remote_call_all_workers_async(
            "register_kvcache_mem_handles_context",
            context_parallel_config,
            kv_cache_mem_handles
        ))
    
    def _free_request_resources(self, request_id: int):
        super()._free_request_resources(request_id)
        self.request_events.pop(request_id)
        self.request_outputs.pop(request_id)

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

    def get_block_status(self):
        block_status = self.block_manager.get_block_usage()
        block_status['gpus'] = self.gpu_ids
        return block_status

    async def _migrate_blocks_context(
        self,
        migrating_req: MigratingRequest
    ) -> None:
        # Allocate blocks on the decoding engine's side        
        # Here we temporarily backup the generated tokens and generated token ids
        # since we are going to overwrite them later when allocating blocks
        generated_token_bkup = migrating_req.req.generated_tokens
        generated_token_ids_bkup = migrating_req.req.generated_token_ids
        migrating_req.req.generated_tokens = []
        migrating_req.req.generated_token_ids = []
        self.block_manager.allocate_blocks(migrating_req.req)
        migrating_req.req.generated_tokens = generated_token_bkup
        migrating_req.req.generated_token_ids = generated_token_ids_bkup
        
        target_block_indexes = self.block_manager.get_block_table(migrating_req.req.request_id)
        assert len(target_block_indexes) == len(migrating_req.block_indexes)
        
        # Transfer the blocks
        self.engine_on_new_lifetime_event_callback(
            migrating_req.req.request_id,
            LifetimeEvent(LifetimeEventType.MigrationBegin)
        )
        await asyncio.wait(self._remote_call_all_workers_async(
            "migrate_blocks_context",
            migrating_req.block_indexes,
            migrating_req.source_parallel_config,
            target_block_indexes
        ))
        self.engine_on_new_lifetime_event_callback(
            migrating_req.req.request_id,
            LifetimeEvent(LifetimeEventType.MigrationEnd)
        )
    
        # Clear the blocks on the context engine's side
        await self.clear_migrated_blocks_callback_context(migrating_req)
            

    async def _step(self) -> None:
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """

        pp_size = self.parallel_config.pipeline_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size

        # pick next batch from scheduler
        # this may trigger migration if some requests are still at context stage
        # this may trigger swap_in if some requests have been swapped out to CPU
        # this may also trigger swap_out if GPU blocks are not enough
        batched_requests = self.scheduler.get_next_batch()

        if len(batched_requests) == 0:
            self.batches_in_pipeline.append(batched_requests)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_DECODING_NO_REQUEST)
        else:
            # logger.info(colored(f"Decoding batch of length {len(batched_requests.requests)}", 'green'))
            # start_time = time.perf_counter()
            # Log down the lifetime event

            for request in batched_requests.requests:
                self.engine_on_new_lifetime_event_callback(
                    request.request_id,
                    LifetimeEvent(LifetimeEventType.DecodingBegin),
                    True
                )

            # Allocate blocks as needed
            self.block_manager.allocate_blocks_batched(batched_requests)

            # Check if all requests are on GPU (i.e. not swapped out)
            assert self.block_manager.is_all_requests_on_gpu(
                batched_requests
            ), "Some requests are currently swapped out to CPU"

            # push the batch into pipeline
            batched_requests.start_one_iteration(time.perf_counter())
            self.batches_in_pipeline.append(batched_requests)
            
            block_tables_all = self.block_manager.get_partial_block_table(batched_requests.get_request_ids())
            seq_group_metadata_list: List[SequenceGroupMetadata] = []
            for req_idx, request in enumerate(batched_requests.requests):
                req_id = request.request_id
                try:
                    seq_md = request.vllm_seq_metadata
                    seq_md.is_prompt = False
                    block_tables={req_id:block_tables_all[req_idx]}
                except:
                    print('ERROR')
                    print(block_tables_all)
                    print(batched_requests.requests)
                    print('ERROR')
                    # breakpoint()
                seq_md.block_tables = block_tables


                gen_token = request.generated_token_ids[-1]
                seq_md.seq_data[req_id].output_token_ids = seq_md.seq_data[req_id].output_token_ids + tuple([gen_token])
                seq_md.seq_data[req_id]._num_computed_tokens = len(seq_md.seq_data[req_id].prompt_token_ids) + len(seq_md.seq_data[req_id].output_token_ids) - 1
                seq_md.token_chunk_size=1
                seq_md.multi_modal_data = None
                seq_group_metadata_list.append(seq_md)

            remote_calls = self._remote_call_all_workers_async(
                "step_llm",
                batched_requests.get_request_ids(),
                seq_group_metadata_list
            )

            # only the leader of the last stage return valid output, i.e., generated tokens ids
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        # output buffer
        finished_reqs = []

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
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
                        print(f"(decoding) Warning: Cannot decode token with id {gen_token_id}. Error: {e}")
                        token = ""
                    generated_tokens.append(token)

                finished_batch = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(
                    generated_tokens, generated_tokens_ids, end_time
                )

                for request, new_token, new_token_id in zip(
                    finished_batch.requests, generated_tokens, generated_tokens_ids
                ):
                    self.engine_on_new_step_output_callback(
                        request.request_id,
                        StepOutput(request, new_token, new_token_id)
                    )
                    if request.is_finished:
                        self.engine_on_new_lifetime_event_callback(
                            request.request_id,
                            LifetimeEvent(LifetimeEventType.DecodingEnd)
                        )
                        self.scheduler.on_finish_request(request)
                finished_reqs = self.scheduler.pop_finished_requests()

                # free blocks for finished requests
                self.block_manager.free_blocks_batched(finished_reqs)

                # COMMENTED OUT BY EPD AUTHoR AS IT CAUSES CHOCKING IN LOOP AND NOT NEEDED AS IT SEEMS TO BE FOR SWAPPING STUFF.
                # self._remote_call_all_workers_async("clear_request_resource_batched", finished_reqs)

                # pop the finished batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
                # logger.info(colored(f"DONE Decoding batch of length {len(batched_requests.requests)}. Time taken {time.perf_counter() - start_time}", 'green'))


    