
from distserve.engine_common import *
from distserve.config import ClusterParallelConfig


class ContextStageCluster(ABC):
    def __init__(
        self,
        bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ClusterParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig,
        placement_groups: List[PlacementGroup],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        self.request_allocator = 0
        self.engines: List[ContextStageLLMEngine] = []

        for dp_rank in range(parallel_config.data_parallel_size):  
            logger.info(f"Initializing Context (DP{dp_rank}) Engine")
            tmp_parallel_config = copy.deepcopy(parallel_config)
            tmp_parallel_config.data_parallel_rank = dp_rank
            self.engines.append(ContextStageLLMEngine(
                bridge_queue,
                model_config,
                tmp_parallel_config,
                cache_config,
                sched_config,
                placement_groups,
                engine_on_new_step_output_callback,
                engine_on_new_lifetime_event_callback
            ))

    async def initialize(self):
        await asyncio.gather(*[engine.initialize() for engine in self.engines])

    def add_request(self, request: Request):
        engine = self.engines[self.request_allocator]
        engine.scheduler.add_request(request)
        self.request_allocator = (self.request_allocator +1) % len(self.engines)

    async def start_event_loop(self):
        await asyncio.gather(*[engine.start_event_loop() for engine in self.engines])        

    def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest):
        """ Called when the context engine finishes migrating the blocks of the request."""
        engine = self.engines[migrated_request.source_parallel_config.data_parallel_rank]
        engine.clear_migrated_blocks_callback(migrated_request)


class ContextStageLLMEngine(SingleStageLLMEngine):
    def _get_scheduler(self) -> ContextStageScheduler:
        return get_context_stage_scheduler(
            self.sched_config,
            self.parallel_config,
            self.block_manager,
        )
    
    def __init__(
        self,
        bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig,
        placement_groups: List[PlacementGroup],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        super().__init__(
            Stage.CONTEXT,
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
        
        self.bridge_queue = bridge_queue


    async def initialize(self):
        await super().initialize()


    async def start_event_loop(self):        
        async def event_loop2():
            # Event loop 2. Run step()
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        
        await asyncio.gather(event_loop2())

    def get_scheduler_status(self):
        scheduler_satus = self.scheduler.get_status()
        return scheduler_satus

    def get_block_status(self):
        block_status = self.block_manager.get_block_usage()
        block_status['gpus'] = self.gpu_ids
        return block_status

    def add_request(self, request: Request):
        self.scheduler.add_request(request)
    
    def _free_request_resources(self, request_id: int):
        # self.vision_block_manager.free_blocks(request_id) # clear vision cache blocks
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
            # logger.info(colored(f"(context) Forwarding with lengths {[len(request.prompt_token_ids) for request in batched_requests.requests]}", 'blue'))
            # logger.info(colored(f"PREFILLING batch of length {len(batched_requests.requests)}", 'blue'))
            # start_time = time.perf_counter()
            # print(f"(context) Forwarding with lengths {batched_requests}", flush=True)
            # allocate blocks as needed
            self.block_manager.allocate_blocks_batched(batched_requests)
            
            # Log down the lifetime event
            for request in batched_requests.requests:
                self.engine_on_new_lifetime_event_callback(request.request_id,
                                                           LifetimeEvent(LifetimeEventType.ContextBegin))
                
            # push the batch into pipeline
            batched_requests.start_one_iteration(time.perf_counter())
            self.batches_in_pipeline.append(batched_requests)

            # print(f"(context) Forwarding with lengths {[len(request.prompt_token_ids) for request in batched_requests.requests]}")

            block_tables_all = self.block_manager.get_partial_block_table(batched_requests.get_request_ids())
            # print(f'(context) block_tables_all {block_tables_all}')

            seq_group_metadata_list: List[SequenceGroupMetadata] = []

            # vision_embeddings = []
            for req_idx, request in enumerate(batched_requests.requests):
                seq_group_metadata = request.vllm_seq_metadata
                req_id = request.request_id
                seq_group_metadata.block_tables = {req_id:block_tables_all[req_idx]}
                seq_group_metadata_list.append(seq_group_metadata)

            remote_calls = self._remote_call_all_workers_async(
                "step_vllm",
                batched_requests.get_request_ids(),
                seq_group_metadata_list,
                None
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
                            request,
                            self.block_manager.get_block_table(request.request_id),
                            self.parallel_config,
                        )
                        self.bridge_queue.put_nowait(migrating_req) # This won't panic because the queue is unbounded
                    else:
                        self._free_request_resources(request.request_id)
                # logger.info(colored(f"(DONE) PREFILLING batch of length {len(batched_requests)}. Time taken {time.perf_counter() - start_time}s", 'blue'))
    
    def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest):
        """ Called when the decoding engine finishes migrating the blocks of the request. """
        self._free_request_resources(migrated_request.req.request_id)
        self.scheduler.on_request_migrated(migrated_request)
