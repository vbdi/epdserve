from abc import ABC, abstractmethod
import copy
from typing import List

from epdserve.config import EncodingStageSchedConfig, ParallelConfig
from epdserve.logger import init_logger
from epdserve.request import Request, BatchedRequests, MigratingRequest
from epdserve.block_manager import VisionBlockManager
logger = init_logger(__name__)

class EncodingStageScheduler():
    """ A first-come-first-serve scheduler. """

    def __init__(
        self,
        sched_config: EncodingStageSchedConfig, 
        parallel_config: ParallelConfig,
        block_manager: VisionBlockManager):
        assert (sched_config.policy == "fcfs"), f"can not initialize a FCFS scheduler with policy {sched_config.policy}"
        self.sched_config = sched_config
        # If the current batch is full, the requests will be put into the waiting queue.
        self.waiting_queue = []
        self.parallel_config: List[Request] = copy.deepcopy(parallel_config)

        # Requests that finished the encoding stage but are not accepted by the context stage.
        self.unaccepted_queue: List[Request] = []
    
        self.block_manager = block_manager
        # The number of on-the-fly (i.e. processing) request blocks
        # Adds when calling get_next_batch_and_pop()
        # Subtracts when calling on_finish_requests()
        self.num_on_fly_request_block = 0

        # these two are not queues
        self.running: List[List[Request]] = [] # to track batch stats
        self.transferred: List[Request] = [] # all reqeusts that are exited from this stage

    def add_request(self, request: Request) -> None:
        """ Add a request to the scheduler. """
        self.waiting_queue.append(request)

    def abort_request(self, request_id: int) -> None:
        """ Cancel a request from the scheduler. """
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return

    def get_next_batch_and_pop(self) -> BatchedRequests:
        """ Get the next batch for the encoding stage in a FCFS-like manner, and pop them """
        next_batch = BatchedRequests()

        def _check_add_to_cur_batch(request: Request) -> bool:
            """ Check whether the request can be added to the current batch. """
            return (
                # Limit 1. batch size
                len(next_batch) < self.sched_config.max_batch_size
                and (
                # Limit 3. GPU blocks
                sum([self.block_manager.get_num_blocks_needed(req) for req in next_batch.requests + [request]]) +
                sum([self.block_manager.get_num_blocks_needed(req) for req in self.unaccepted_queue]) +
                self.num_on_fly_request_block 
                <= self.block_manager.max_num_gpu_blocks)
            )

    
        while len(self.waiting_queue) > 0:
            request = self.waiting_queue[0]
            if _check_add_to_cur_batch(request):
                next_batch.add_request(request)
                self.waiting_queue.pop(0)
            else:
                break
        
        self.running.append(next_batch.requests)
        self.num_on_fly_request_block += sum([self.block_manager.get_num_blocks_needed(req) for req in next_batch.requests])
        return next_batch

    def on_finish_requests(self, batch: BatchedRequests):
        for request in batch.requests:
            if not request.is_finished:
                self.unaccepted_queue.append(request)
        self.num_on_fly_request_block -= sum([self.block_manager.get_num_blocks_needed(req) for req in batch.requests])
    
    def on_request_migrated(self, migrated_request: MigratingRequest):
        for i, request in enumerate(self.unaccepted_queue):
            if request.request_id == migrated_request.req.request_id:
                self.transferred.append(migrated_request.req)
                del self.unaccepted_queue[i]
                return

    def get_num_waiting_requests(self) -> int:
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        return (
            f"FCFS(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )
    
    def print_status(self):
        logger.info(f"(encoding) {len(self.waiting_queue)} waiting, {len(self.unaccepted_queue)} finished but unaccepted, {self.num_on_fly_request_block} blocks occupied by on-the-fly requests")

    def get_status(self):
        status = {'queuing': len(self.waiting_queue),         # If the current batch is full, the requests will be put into the waiting queue.                        
                  'running': len(self.running[-1]) if len(self.running)>0 else 0,
                  'awaiting': len(self.unaccepted_queue),   # Requests that finished the encoding stage but are not accepted by the context stage.                  
                  'exited': len(self.transferred),
                  'blocks': self.num_on_fly_request_block
                  }
        return status
 
def get_encoding_stage_scheduler(
    sched_config: EncodingStageSchedConfig,
    parallel_config: ParallelConfig,
    block_manager: VisionBlockManager
) -> EncodingStageScheduler:
    if sched_config.policy == "fcfs":
        return EncodingStageScheduler(sched_config, parallel_config, block_manager)
    else:
        raise NotImplementedError(f"Unknown encoding scheduler policy {sched_config.policy}")
    