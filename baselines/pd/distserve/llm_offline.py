import time
from typing import List, Union, Optional

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

class OfflineLLM:
    """A Large Language Model (LLM) for offline inference.
    It wraps around the LLMEngine and provides the **generate** interface to do
    offline inference on a list of prompts, which only return when all the prompts
    finish generation. If you want to do online inference where each user can asynchronously
    get the generation results in a streaming fashion, please refer to the **AsyncLLM** class.
    """

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
        self.run_stats = {}
        self.outputs = []
        self.inputs = []

    def generate(
        self,
        inputs: Optional[Union[List[dict], dict]] = None,
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        use_tqdm: bool = True,
    ) -> List[List[StepOutput]]:
        num_requests = len(inputs)
        if sampling_params is None:
            sampling_params = [SamplingParams()] * num_requests
        elif isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * num_requests
        else:
            assert (len(sampling_params) == num_requests), f"prompts should pair with the list of sampling parameters, but got {num_requests} prompts and {len(sampling_params)} sampling parameters"

        async def deal_with_request_coroutine(req_index: int) -> List[StepOutput]:
            input = inputs[req_index] if inputs is not None else None
            step_outputs = []
            request_id = next(self.request_counter)
            arrival_time = time.perf_counter()
            async for step_output in self.engine.generate(input, sampling_params[req_index], request_id, arrival_time):
                step_outputs.append(step_output)
            return step_outputs
        
        async def generate_main() -> List[List[StepOutput]]:
            request_tasks = []
            event_loop_task = asyncio.create_task(self.engine.start_all_event_loops())

            for i in range(num_requests):
                request_tasks.append(asyncio.create_task(deal_with_request_coroutine(i)))
            
            result = await asyncio.gather(*request_tasks)
            event_loop_task.cancel()
            return result

        start_time = time.perf_counter()
        try:
            outputs = asyncio.run(generate_main())
        except Exception as e:
            print('\n\n PROCESS STOPPED BY THE USER')
            print(f'Exception {e}')
            exit()
        time_taken = time.perf_counter() - start_time
        self.run_stats['running_time'] = time_taken

        # print(self.run_stats)
        self.outputs.extend(outputs)
        self.inputs.extend(inputs)
        return outputs

    def get_request_lifetime_events(self, request_id: str):
        return self.engine.request_lifetime_events.get(request_id)

    def print_outputs(self):
        for pid, (input, step_outputs) in enumerate(zip(self.inputs, self.outputs)):
            # new_token_ids = [step_output.new_token_id for step_output in step_outputs]
            # output_text = llm.tokenizer.decode(new_token_ids)
            print(f"{pid}. Prompt: {input['prompt']!r}\n Generated text: {' '.join([step_output.new_token for step_output in step_outputs])} ({len(step_outputs)} tokens generated).\n")

    def dump_lifetime_events(self):
        json_data = [json_encode_lifetime_events(event_dict.values()) for req_id, event_dict in self.engine.request_lifetime_events.items()]
        file_name = 'lifetime_events.json'
        file_path = os.path.join(os.path.dirname(__file__), f'../runs/{self.engine.model_config.run_name}/{file_name}')
        with open(check_create_dir(file_path), 'w') as file:
            json.dump(json_data, file, indent=4)

    def dump_tb_metrics(self):
        json_data = self.engine.tb_metrics
        file_name = 'tb_metrics.json'
        file_path = os.path.join(os.path.dirname(__file__), f'../runs/{self.engine.model_config.run_name}/{file_name}')
        with open(check_create_dir(file_path), 'w') as file:
            json.dump(json_data, file, indent=4)

    def dump_run_stats(self):
        self.run_stats['gpu_time'] = self.run_stats['running_time']*self.engine.num_engines
        if 'F.gpu_util/C.decoding' in self.engine.tb_metrics[0]:
            self.run_stats['context_gpu_util'] = np.mean([metrics['F.gpu_util/B.context'] for metrics in self.engine.tb_metrics])
            self.run_stats['decoding_gpu_util'] = np.mean([metrics['F.gpu_util/C.decoding'] for metrics in self.engine.tb_metrics])
            self.run_stats['overall_gpu_util'] = np.mean([self.run_stats['encoding_gpu_util'], self.run_stats['context_gpu_util'], self.run_stats['decoding_gpu_util']])
        self.run_stats['context_blocks'] = np.mean([metrics['A.blocks%/B.context'] for metrics in self.engine.tb_metrics])
        self.run_stats['decoding_blocks'] = np.mean([metrics['A.blocks%/C.decoding'] for metrics in self.engine.tb_metrics])

        # issue_ts = np.array([self.engine.request_lifetime_events[req_id][0].timestamp for req_id in range(len(self.outputs))])
        issue_ts = np.array([self.engine.request_lifetime_events[req_id]['issued'].timestamp for req_id in range(len(self.outputs))])
        token_ts = np.array([[tok.timestamp for tok in req] for req in self.outputs])
        ttft = token_ts[:,0]-issue_ts
        tpot = token_ts[:, 1:] - token_ts[:, :-1]
        token_times = np.hstack([ttft.reshape(-1, 1) , tpot])
        self.run_stats['ttft'] = ttft.mean()
        self.run_stats['tpot'] = tpot.mean()


        ''' only make sense in offline scenario '''
        # print([event.event_type.value for event in self.engine.request_lifetime_events[0]])
        # ['issued', 'encoding_begin', 'encoding_end', 'encoding_migration_begin', 'encoding_migration_end', 'context_begin', 'context_end', 'migration_begin', 'migration_end', 'decoding_begin', 'decoding_end']

        # encoding_stage_started = np.min([np.min([self.engine.request_lifetime_events[req_id][key].timestamp for key in self.engine.request_lifetime_events[req_id].keys() if key.startswith('encoding_begin')])  for req_id in range(len(self.outputs))])
        # encoding_stage_finished = np.max([np.max([self.engine.request_lifetime_events[req_id][key].timestamp for key in self.engine.request_lifetime_events[req_id].keys() if key.startswith('encoding_end')])  for req_id in range(len(self.outputs))])
        context_stage_started = np.min([self.engine.request_lifetime_events[req_id]['context_begin'].timestamp for req_id in range(len(self.outputs))])
        context_stage_finished = np.max([self.engine.request_lifetime_events[req_id]['context_end'].timestamp for req_id in range(len(self.outputs))])
        total_encoding_images_processed = np.sum([self.inputs[req_id]['multi_modal_data']['image'].__len__() for req_id in range(len(self.outputs))])
        total_prefill_tokens = np.sum([self.inputs[req_id]['prompt_token_ids'].__len__() for req_id in range(len(self.inputs))])
        total_prefill_processed_tokens = np.sum([self.inputs[req_id]['processed_prompt_token_ids'].__len__() for req_id in range(len(self.inputs))])        
        # self.run_stats['throughput_encoding'] = total_encoding_images_processed/(encoding_stage_finished-encoding_stage_started)
        self.run_stats['throughput_prefill'] = total_prefill_tokens/(context_stage_finished-context_stage_started)
        self.run_stats['throughput_prefill_processed'] = total_prefill_processed_tokens/(context_stage_finished-context_stage_started)

        self.run_stats['#images/req'] = np.mean([self.inputs[req_id]['multi_modal_data']['image'].__len__() for req_id in range(len(self.outputs))])
        self.run_stats['#prefill_tokens'] = np.mean([self.inputs[req_id]['prompt_token_ids'].__len__() for req_id in range(len(self.inputs))])
        self.run_stats['#prefill_tokens_processed'] = np.mean([self.inputs[req_id]['processed_prompt_token_ids'].__len__() for req_id in range(len(self.inputs))])

        try:
            decoding_stage_started = np.min([self.engine.request_lifetime_events[req_id][9].timestamp for req_id in range(len(self.outputs))])
            decoding_stage_finished = np.max([self.engine.request_lifetime_events[req_id][10].timestamp for req_id in range(len(self.outputs))])
            total_decoding_generated_tokens = np.sum([self.outputs[req_id].__len__()-1 for req_id in range(len(self.outputs))])
            self.run_stats['throughput_decoding'] = total_decoding_generated_tokens/(decoding_stage_finished-decoding_stage_started)
        except:
            print('Decoding was not performed (num_output_tokens must be equal to 1)')
        
        print(pd.DataFrame([self.run_stats]).T)

        self.run_stats['token_times'] = token_times.tolist()

        file_name = 'run_stats.json'
        file_path = os.path.join(os.path.dirname(__file__), f'../runs/{self.engine.model_config.run_name}/{file_name}')
        file_path = check_create_dir(file_path)
        with open(file_path, 'w') as file:
            json.dump(self.run_stats, file, indent=4)

