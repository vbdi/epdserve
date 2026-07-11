from abc import ABC, abstractmethod
import copy
from typing import List, Callable, Tuple

from epdserve.config import ContextStageSchedConfig, ParallelConfig
from epdserve.logger import init_logger
from epdserve.request import Request, BatchedRequests, MigratingRequest
from epdserve.block_manager import BlockManager, VisionBlockManager
from collections import defaultdict

logger = init_logger(__name__)

class ContextStageScheduler():
    """
    A first-come-first-serve scheduler.
    """

    def __init__(
        self,
        sched_config: ContextStageSchedConfig, 
        parallel_config: ParallelConfig,
        block_manager: BlockManager,
        vision_block_manager: VisionBlockManager,
        engine_encoding_migrate_block_callback: Callable,
        encode_context_bridge_queue):
        self.encode_context_bridge_queue = encode_context_bridge_queue
        assert (sched_config.policy == "fcfs"), f"can not initialize a FCFS scheduler with policy {sched_config.policy}"
        self.sched_config = sched_config
        # If the current batch is full, the requests will be put into the waiting queue.
        self.waiting_queue = []
        self.parallel_config: List[Request] = copy.deepcopy(parallel_config)
        self.block_manager = block_manager
        # Requests that finished the context stage but are not accepted by the decoding stage.
        self.unaccepted_queue: List[Request] = []

        # If the request has not been accepted (i.e. it still resides in the "bridge" queu
        # and its vision block are still on the encoding stage engine's side), then it will be put
        # into the encoding unaccepted queue.
        self.encoding_unaccepted_queue: List[MigratingRequest] = []

        # The number of on-the-fly (i.e. processing) request blocks
        # Adds when calling get_next_batch_and_pop(); Subtracts when calling on_finish_requests()
        self.num_on_fly_request_block = 0
        self.engine_encoding_migrate_block_callback = engine_encoding_migrate_block_callback

        # these two are not queues
        self.running: List[List[Request]] = [] # to track batch stats
        self.transferred: List[Request] = [] # all reqeusts that are exited from this stage

        self.sharded_requests = defaultdict(dict)
        self.vision_block_manager = vision_block_manager


    async def add_request(self, migrating_req: MigratingRequest) -> None:
        """
        Add a request to the scheduler.
        """
        self.encoding_unaccepted_queue.append(migrating_req)

    def abort_request(self, request_id: int) -> None:
        """
        Cancel a request from the scheduler.
        """
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return

    def _get_block_needed(self, length: int):
        block_size = self.block_manager.cache_config.block_size
        return (length + block_size - 1) // block_size
            
    def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        Get the next batch for the context stage in a FCFS-like manner, and pop them
        """
        next_batch = BatchedRequests()

        def _check_add_to_cur_batch(request: Request) -> bool:
            """
            Check whether the request can be added to the current batch.
            """
            return (
                # Limit 1. batch size
                len(next_batch) < self.sched_config.max_batch_size
            ) and (
                # Limit 2. tokens per batch
                next_batch.get_num_input_tokens() + request.get_num_input_tokens() <= self.sched_config.max_tokens_per_batch
            ) and (
                # Limit 3. GPU blocks
                sum([self._get_block_needed(len(req.prompt_token_ids)) for req in next_batch.requests + [request]]) +
                sum([self._get_block_needed(len(req.prompt_token_ids)) for req in self.unaccepted_queue]) +
                self.num_on_fly_request_block 
                <= self.block_manager.max_num_gpu_blocks
            )
    
        while len(self.waiting_queue) > 0:
            request = self.waiting_queue[0]
            if _check_add_to_cur_batch(request):
                next_batch.add_request(request)
                self.waiting_queue.pop(0)
            else:
                break
        
        self.running.append(next_batch.requests)
        self.num_on_fly_request_block += sum([self._get_block_needed(req.get_input_len()) for req in next_batch.requests])

        return next_batch

    def on_finish_requests(self, batch: BatchedRequests):
        for request in batch.requests:
            if not request.is_finished:
                self.unaccepted_queue.append(request)
        
        self.num_on_fly_request_block -= sum([self._get_block_needed(req.get_input_len()) for req in batch.requests])
    
    def on_request_migrated(self, migrated_request: MigratingRequest):
        for i, request in enumerate(self.unaccepted_queue):
            if request.request_id == migrated_request.req.request_id:
                del self.unaccepted_queue[i]
                self.transferred.append(migrated_request.req)
                return
            
    def get_num_waiting_requests(self) -> int:
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        return (
            f"FCFS(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )
    
    def print_status(self):
        logger.info(f"(context) {len(self.waiting_queue)} waiting, {len(self.unaccepted_queue)} finished but unaccepted, {len(self.encoding_unaccepted_queue)} finished but unaccepted (encoding), {self.num_on_fly_request_block} blocks occupied by on-the-fly requests")


    def get_status(self):
        status = {'queuing': len(self.waiting_queue),         # If the current batch is full, the requests will be put into the waiting queue.                        
                  'running': len(self.running[-1]) if len(self.running)>0 else 0,
                  'awaiting': len(self.unaccepted_queue),    # Requests that finished the context stage but are not accepted by the decoding stage.                
                  'exited': len(self.transferred),
                  'blocks': self.num_on_fly_request_block
                  }
        return status


    async def post_process(self) -> None:
        def should_accept(migrating_req: MigratingRequest) -> bool:
            return sum([self._get_block_needed(len(req.prompt_token_ids))
                        for req in self.waiting_queue
                    ]) < self.block_manager.max_num_gpu_blocks * self.sched_config.waiting_block_prop_threshold \
                    and self._get_block_needed(len(migrating_req.req.prompt_token_ids)) <= self.block_manager.get_num_avail_gpu_blocks()


        while len(self.encode_context_bridge_queue) > 0:
            migrating_req = self.encode_context_bridge_queue[0]
            if should_accept(migrating_req):
                self.encode_context_bridge_queue.pop(0)
                await self.engine_encoding_migrate_block_callback(migrating_req)

                # assumption: sharded reqs have string request_id
                if isinstance(migrating_req.req.request_id, str):
                    unsharded_req_id, sharded_req_idx, sharded_total_reqs = [int(val) for val in migrating_req.req.request_id.split('_')]
                    self.sharded_requests[unsharded_req_id][sharded_req_idx] = migrating_req.req
                    if len(self.sharded_requests[unsharded_req_id]) == sharded_total_reqs:
                        base_req = self.sharded_requests[unsharded_req_id][1]
                        multi_modal_data_list = [self.sharded_requests[unsharded_req_id][req_idx].vllm_seq_metadata.multi_modal_data['image']['image'] for req_idx in range(1, sharded_total_reqs+1)]
                        multi_modal_data_list = [item for sublist in multi_modal_data_list for item in sublist]
                        sharded_req_ids = [self.sharded_requests[unsharded_req_id][req_idx].request_id for req_idx in range(1, sharded_total_reqs+1)]
                        block_table_list = [self.vision_block_manager.block_table[shar_req_id] for shar_req_id in sharded_req_ids]
                        block_table_list = [item for sublist in block_table_list for item in sublist]
                        base_req_location = self.vision_block_manager.request_location[sharded_req_ids[0]]
                        for shar_req_id in sharded_req_ids:
                            del self.vision_block_manager.block_table[shar_req_id]
                            del self.vision_block_manager.request_location[shar_req_id]
                        self.vision_block_manager.block_table[unsharded_req_id] = block_table_list
                        self.vision_block_manager.request_location[unsharded_req_id] = base_req_location
                        base_req.request_id = unsharded_req_id
                        base_req.vllm_seq_metadata.multi_modal_data['image']['image'] = multi_modal_data_list
                        self.waiting_queue.append(base_req)
                else: # handle unsharded reqs as well
                    self.waiting_queue.append(migrating_req.req)
            else:
                break

        

def get_context_stage_scheduler(
    sched_config: ContextStageSchedConfig,
    parallel_config: ParallelConfig,
    block_manager: BlockManager,
    vision_block_manager: VisionBlockManager,
    engine_migrate_block_callback: Callable,
    encode_context_bridge_queue
) -> ContextStageScheduler:
    if sched_config.policy == "fcfs":
        return ContextStageScheduler(sched_config, parallel_config, block_manager, vision_block_manager, engine_migrate_block_callback, encode_context_bridge_queue)
    else:
        raise NotImplementedError(f"Unknown context scheduler policy {sched_config.policy}")
    