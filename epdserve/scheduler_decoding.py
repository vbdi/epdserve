from abc import ABC, abstractmethod
import copy
from typing import List, Callable, Tuple
import warnings
import torch

from epdserve.config import ParallelConfig, DecodingStageSchedConfig
from epdserve.logger import init_logger
from epdserve.request import Request, BatchedRequests, MigratingRequest
from epdserve.block_manager import BlockManager, BlockLocation

logger = init_logger(__name__)


class DecodingStageScheduler():
    """A first-come-first-serve scheduler.
    Note: It supports pipeline parallelism. It maintains #pp disjoint batches which are in the pipeline under execution.
    Note: The requests are in waiting_queue or the batch_queues, and one request can only be in one queue at a time.
    """

    def __init__(
        self,
        sched_config: DecodingStageSchedConfig,
        parallel_config: ParallelConfig,
        block_manager: BlockManager,
        engine_migrate_block_callback: Callable,
        context_decode_bridge_queue
    ):
        self.context_decode_bridge_queue = context_decode_bridge_queue
        assert (
            sched_config.policy == "fcfs"
        ), f"can not initialize a FCFS scheduler with policy {sched_config.policy}"
        self.sched_config = sched_config
        # If the request has not been accepted (i.e. it still resides in the "bridge" queu
        # and its block are still on the context stage engine's side), then it will be put
        # into the unaccepted queue.
        self.unaccepted_queue: List[MigratingRequest] = []
        # If the current batch is full, the requests will be put into the waiting queue.
        self.waiting_queue: List[Request] = []
        # If one request was in batch_queues before, but swapped out, it will be put into the swapped queue.
        self.swapped_queue: List[Request] = []
        # Since pipeline parallelism is used, there are multiple batches in the system.
        self.cur_index = -1
        self.batch_queues = [
            BatchedRequests() for i in range(parallel_config.pipeline_parallel_size)
        ]
        self.parallel_config = copy.deepcopy(parallel_config)
        self.block_manager = block_manager
        self.engine_migrate_block_callback = engine_migrate_block_callback
        
        self.processed: List[Request] = [] # to track all done and finished requests

    # Requests-related methods
    async def add_request(self, migrating_req: MigratingRequest) -> None:
        # We take a simple approach here: Accept any request that comes in.
        self.unaccepted_queue.append(migrating_req)

    def abort_request(self, request_id: int) -> None:
        # scan the current batch
        for queue in self.batch_queues:
            for _, request in enumerate(queue.requests):
                if request.request_id == request_id:
                    # This request may be under processed by the model currently,
                    # so it is not safe to delete it from current batch directly.
                    # Mark it as finished will release the resources it holds finally.
                    request.is_finished = True
                    return

        # scan the waiting queue
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return

    def _get_last_stage_batch(self) -> BatchedRequests:
        last_stage_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size
        return self.batch_queues[last_stage_index]

    def pop_finished_requests(self) -> List[Request]:
        return self._get_last_stage_batch().pop_finished_requests()


    def _get_block_needed(self, length: int):
        block_size = self.block_manager.cache_config.block_size
        return (length + block_size - 1) // block_size

        
    def _check_add_to_cur_batch(self, request: Request) -> bool:
        return (
            # Limit 1. batch size
            len(self.batch_queues[self.cur_index]) < self.sched_config.max_batch_size
        ) and (
            # Limit 2. tokens per batch
            self.batch_queues[self.cur_index].get_num_input_tokens() + request.get_num_input_tokens() <= self.sched_config.max_tokens_per_batch
            
        ) and (
            # Limit 3. GPU blocks

            # blocks for requests already being processed
            sum([sum([self._get_block_needed(len(req.prompt_token_ids) + req.get_actual_output_len()) for req in self.batch_queues[index].requests]) for index in range(self.parallel_config.pipeline_parallel_size)]) 
            # blocks for requests that has been transferred and are in waiting queue
            + sum([self._get_block_needed(len(req.prompt_token_ids)) for req in self.waiting_queue]) 
            # blocks requred for the current requests
            + self._get_block_needed(request.get_input_len() + request.get_actual_output_len()) \
                <= self.block_manager.max_num_gpu_blocks
        )

    def get_next_batch(self) -> BatchedRequests:
        self.cur_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size

        # Check whether the blocks on GPU is enough for the next batch. If not, swap out the last request
        while sum([sum([self._get_block_needed(req.get_input_len() + req.get_output_len()) for req in self.batch_queues[index].requests]) for index in range(self.parallel_config.pipeline_parallel_size)]) \
            + sum([self._get_block_needed(req.get_input_len()) for req in self.waiting_queue]) \
            > self.block_manager.max_num_gpu_blocks:
            
            logger.info("No enough GPU blocks. Swap-out triggered in Decoding")
            raise ValueError("for temporarily getting around, no swap should happens")
            request = self.batch_queues[self.cur_index].requests.pop(-1)
            self.swapped_queue.append(request)
            self.block_manager.swap_out_requests([request])

        # Try to add in some new requests.
        while len(self.swapped_queue) > 0 or len(self.waiting_queue) > 0:
            # Consider requests in the swapped queue first.
            if len(self.swapped_queue) > 0:
                request = self.swapped_queue[0]
                if self._check_add_to_cur_batch(request):
                    logger.info("Swap-in triggered")
                    self.block_manager.swap_in_requests([request])
                    self.batch_queues[self.cur_index].add_request(request)
                    self.swapped_queue.pop(0)
                else:
                    break
            # Otherwise check if one request from the waiting queue can be incorporated.
            else:
                request = self.waiting_queue[0]
                if self._check_add_to_cur_batch(request):
                    self.batch_queues[self.cur_index].add_request(request)
                    self.waiting_queue.pop(0)
                else:
                    break
        return self.batch_queues[self.cur_index]

    # Getter functions
    def get_total_num_requests(self) -> int:
        return self.get_processing_num_requests() + self.get_waiting_num_requests()

    def get_processing_num_requests(self) -> int:
        num = 0
        for batch in self.batch_queues:
            num = num + len(batch.requests)
        return num

    @property
    def running(self):
        return [[1 for i in range(self.get_processing_num_requests())]]

    def get_waiting_num_requests(self) -> int:
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        return (
            f"FCFS(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )
    
    def print_status(self) -> None:
        logger.info(f"(decoding) {len(self.unaccepted_queue)} unaccepted, {len(self.waiting_queue)} waiting, {self.get_processing_num_requests()} processing")

 
    def on_finish_request(self, request: Request):
        self.processed.append(request)
 
 
    def get_status(self):
        status = {'queuing': len(self.waiting_queue),         # If the current batch is full, the requests will be put into the waiting queue.                        
                  'running': self.get_processing_num_requests(),
                  'awaiting': -1,    # Requests that finished the context stage but are not accepted by the decoding stage.
                  'exited': len(self.processed),
                  'blocks': -1
                #   'unaccepted': len(self.unaccepted_queue),   # If the request has not been accepted (i.e. it still resides in the "bridge" queu
                                                              # and its block are still on the context stage engine's side), then it will be put
                                                              # into the unaccepted queue.                   
                  }
        return status

    async def post_process(self) -> None:
        def should_accept(migrating_req: MigratingRequest) -> bool:
            # Condition1: check blocks already taken by transferred reqs < suggested allocated blocks in KV cache
            # Condition2: blocks are available for this request
            # Condition3: forsee #tokens will be used in worst case. # TODO this is for temporarily getting around swap-out issue 
            decision1 = sum([self._get_block_needed(len(req.prompt_token_ids)) for req in self.waiting_queue]) < self.block_manager.max_num_gpu_blocks * self.sched_config.waiting_block_prop_threshold \
                        and self._get_block_needed(len(migrating_req.req.prompt_token_ids)) <= self.block_manager.get_num_avail_gpu_blocks() \
                        and sum([self._get_block_needed(req.get_input_len()) for req in self.waiting_queue]) + sum([self._get_block_needed(req.get_input_len() + req.get_actual_output_len()) for req in self.batch_queues[0].requests]) + self._get_block_needed(len(migrating_req.req.prompt_token_ids)) < self.block_manager.max_num_gpu_blocks
            assert self.parallel_config.pipeline_parallel_size==1, "Condition 3 assume this"
            decision2 = len(self.waiting_queue)<=self.sched_config.max_batch_size*2 # 2 batches are allowed to be stored
            return decision1 and decision2

        while len(self.context_decode_bridge_queue) > 0:
            migrating_req = self.context_decode_bridge_queue[0]
            if should_accept(migrating_req):
                self.context_decode_bridge_queue.pop(0)
                await self.engine_migrate_block_callback(migrating_req)
                self.waiting_queue.append(migrating_req.req)
            else:
                break


def get_decoding_stage_scheduler(
    sched_config: DecodingStageSchedConfig,
    parallel_config: ParallelConfig,
    block_manager: BlockManager,
    engine_migrate_block_callback: Callable,
    context_decode_bridge_queue
) -> DecodingStageScheduler:
    if sched_config.policy == "fcfs":
        return DecodingStageScheduler(sched_config, parallel_config, block_manager, engine_migrate_block_callback, context_decode_bridge_queue)
    else:
        raise NotImplementedError(
            f"scheduler policy {sched_config.policy} is not supported"
        )
        