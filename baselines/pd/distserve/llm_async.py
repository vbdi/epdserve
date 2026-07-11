import time
from typing import List, Union, Optional, AsyncGenerator

import asyncio
from tqdm import tqdm
import argparse
import pandas as pd

from distserve.config import (
    ModelConfig,
    ClusterParallelConfig,
    CacheConfig,
    DisaggParallelConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.engine_common import StepOutput
from distserve.llm_engine import LLMEngine
from distserve.logger import init_logger
from distserve.request import Request, SamplingParams
from distserve.lifetime import json_encode_lifetime_events
from distserve.utils import Counter
import json, os
import numpy as np
from distserve.utils import check_create_dir
from vllm import EngineArgs

logger = init_logger(__name__)

class AsyncLLM:
    """A Large Language Model (LLM) for online inference."""

    def __init__(
        self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        context_sched_config: ContextStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig
    ):
        self.request_counter = Counter()
        self.engine = LLMEngine(
            model_config,
            disagg_parallel_config,
            cache_config,
            context_sched_config,
            decoding_sched_config
        )
        
        asyncio.run(self.engine.initialize())
        
    async def generate(
        self,
        request_id: int,
        input_dict: dict,
        arrival_time: float,
        sampling_params: SamplingParams = SamplingParams(),
        **kwargs,
    ) -> AsyncGenerator[StepOutput, None]:
        """Generate outputs for a single request.

        This method is a coroutine. It adds the request into the engine, and
        yields the StepOutput objects from the LLMEngine for the request.

        Args:
            request_id: The unique id of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            sampling_params: The sampling parameters of the request.

        Yields:
            The output `StepOutput` objects from the LLMEngine for the
            request.
        """
        if input_dict is None:
            raise ValueError("input_dict must be provided")

        async for step_output in  self.engine.generate(
                            input=input_dict,
                            sampling_params=sampling_params,
                            request_id=next(self.request_counter) if  request_id is None else request_id,
                            arrival_time=arrival_time if arrival_time is not None else time.perf_counter(),
        ):
            yield step_output
            
        # Here the engine has all the request's lifetime events in engine.request_lifetime_events[request_id]
        # But unfortunately I don't know how to return it and when to clear it...
        # TODO Find a reliable way to return and clear the lifetime events

    async def start_event_loop(self):
        await self.engine.start_all_event_loops()

    def from_engine_args(
        args: argparse.Namespace
    ):
        vllm_engine_args = EngineArgs(model=args.model,
                                    max_num_seqs=8,
                                    tensor_parallel_size=1,
                                    dtype="float16",
                                    enforce_eager=True,
                                    quantization=None,
                                    trust_remote_code=True,
                                    limit_mm_per_prompt={"image": args.limit_mm_per_prompt})
        vllm_engine_config = vllm_engine_args.create_engine_config()

        return AsyncLLM(
            model_config=ModelConfig(
                model=args.model,
                dtype="fp16",
                # tokenizer=args.tokenizer,
                tokenizer=None if "llava" in args.model else args.model,
                trust_remote_code=True,
                # seed=args.seed,
                # use_dummy_weights=args.use_dummy_weights,
                vllm_config=vllm_engine_config
            ),
            disagg_parallel_config=DisaggParallelConfig(
                context=ClusterParallelConfig(
                    data_parallel_size=args.context_data_parallel_size,
                    tensor_parallel_size=args.context_tensor_parallel_size,
                    pipeline_parallel_size=args.context_pipeline_parallel_size
                ),
                decoding=ClusterParallelConfig(
                    data_parallel_size=args.decoding_data_parallel_size,
                    tensor_parallel_size=args.decoding_tensor_parallel_size,
                    pipeline_parallel_size=args.decoding_pipeline_parallel_size
                )
            ),
            cache_config=CacheConfig(
                block_size=args.block_size,
                max_num_blocks_per_req=args.max_num_blocks_per_req,
                gpu_memory_utilization=args.gpu_memory_utilization,
                cpu_swap_space=args.swap_space
            ),
            context_sched_config=ContextStageSchedConfig(
                policy=args.context_sched_policy,
                max_batch_size=args.context_max_batch_size,
                max_tokens_per_batch=args.context_max_tokens_per_batch
            ),
            decoding_sched_config=DecodingStageSchedConfig(
                policy=args.decoding_sched_policy,
                max_batch_size=args.decoding_max_batch_size,
                max_tokens_per_batch=args.decoding_max_tokens_per_batch,
                model_name=args.model,
                waiting_block_prop_threshold=0.05
            )
        )

    def get_and_pop_request_lifetime_events(self, request_id: str):
        return self.engine.request_lifetime_events.pop(request_id)
    
    async def abort(self, request_id: str) -> None:
        logger.info(f"Aborted request {request_id}.")
        self.engine.abort_request(request_id)

