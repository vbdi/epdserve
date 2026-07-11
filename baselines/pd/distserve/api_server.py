"""
Usage example:

python -m distserve.api_server.distserve_api_server \\
    --host 0.0.0.0 \\
    --port {port} \\
    --model {args.model} \\
    --tokenizer {args.model} \\
    \\
    --context-data-parallel-size {context_dp} \\
    --context-tensor-parallel-size {context_tp} \\
    --context-pipeline-parallel-size {context_pp} \\
    --decoding-data-parallel-size {decoding_dp} \\
    --decoding-tensor-parallel-size {decoding_tp} \\
    --decoding-pipeline-parallel-size {decoding_pp} \\
    \\
    --block-size 16 \\
    --max-num-blocks-per-req 128 \\
    --gpu-memory-utilization 0.95 \\
    --swap-space 16 \\
    \\
    --context-sched-policy fcfs \\
    --context-max-batch-size 128 \\
    --context-max-tokens-per-batch 8192 \\
    \\
    --decoding-sched-policy fcfs \\
    --decoding-max-batch-size 1024 \\
    --decoding-max-tokens-per-batch 65536
"""
import sys, os
import argparse
import json
from typing import AsyncGenerator, List, Tuple
import asyncio
import time
import traceback
import signal

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
import distserve
import distserve.llm_engine
from distserve import utils
from distserve.llm_async import AsyncLLM
from distserve.request import SamplingParams
from distserve.utils import random_uuid, set_random_seed
from distserve.logger import init_logger
from distserve.engine_common import StepOutput
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.lifetime import json_encode_lifetime_events
import ray
from PIL import Image 
from distserve import utils

logger = init_logger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
counter = utils.Counter()

image_cache = {}
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '../../../assets')
image_path = os.path.join(ASSETS_DIR, 'images/image2_4032_3024.jpg')
image_name = os.path.basename(image_path)
image_cache[image_name] = Image.open(image_path).convert("RGB")

def load_image_cache(image_path):
    image_name = os.path.basename(image_path)
    if image_name in image_cache:
        return image_cache[image_name]
    else:
        # print (f'LOADING IMAGE {image_name}')
        # image_cache[image_path] = Image.open(image_path).convert("RGB")
        image_cache[image_name] = utils.load_images_parallel([image_path])[0]
        return image_cache[image_name]


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    logger.info("Received a request.")
    request_dict = await request.json()
    request_dict.pop("stream")
    input_dict = request_dict.pop("input") 
    prompt = input_dict["prompt"]
    mm_data = input_dict["multi_modal_data"]

    # start_time = time.perf_counter()
    ps = mm_data.pop("images_paths")
    # mm_data["image"] = utils.load_images_parallel(ps)
    # mm_data["image"]= [Image.open(p).convert("RGB") for p in ps ]
    # mm_data["image"] = [Image.open(ps[0]).convert("RGB")] * len(ps)
    # mm_data["image"] = [load_image_cache(p) for p in ps ]
    mm_data["image"] = [load_image_cache(ps[0])] * len(ps)

    # import pdb; pdb.set_trace()
    # images = [utils.base64_to_image(image) for image in mm_data['images']]
    # mm_data["image"] = images

    # print(f'Time taken in reading images {time.perf_counter() - start_time}')
    
    sampling_params = SamplingParams(**request_dict)
    request_id = counter.__next__()
    # intentionnally not added the image loading in arrival time (cons. with offline and realistic scenario)
    arrival_time = time.perf_counter() 

    results_generator = engine.generate(request_id=request_id,
                                        input_dict=input_dict, 
                                        sampling_params=sampling_params,
                                        arrival_time=arrival_time)

    final_outputs: List[Tuple[StepOutput, float]] = [] # (step_output, timestamp)
    async for step_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_outputs.append((step_output, time.perf_counter()))

    request_events = engine.get_and_pop_request_lifetime_events(request_id)
    
    # json_data = [json_encode_lifetime_events(event_dict.values()) for req_id, event_dict in self.engine.request_lifetime_events.items()]
    # import pdb; pdb.set_trace()

    text_output = prompt + ''.join([step_output[0].new_token for step_output in final_outputs])
    ret = {
        # "text": text_output,
        "prompt": prompt,
        "output": ''.join([step_output[0].new_token for step_output in final_outputs]),
        'prompt_token_len': len(input_dict['prompt_token_ids']),
        'processed_prompt_token_len': len(input_dict['processed_prompt_token_ids']),
        "timestamps": [step_output[1] for step_output in final_outputs],
        "lifetime_events": json_encode_lifetime_events(request_events.values())
    }
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    
    distserve.llm_engine.add_engine_cli_args(parser)

    PD_CONFIG = [7, 1]
    BS_CONFIG = [1, 1]
    MODEL_PATH=os.path.join(os.path.dirname(__file__), '../../assets/models/MiniCPM-V-2_6')

    parser.set_defaults(
        host='localhost',
        port=8400,
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        context_data_parallel_size=PD_CONFIG[0],
        decoding_data_parallel_size=PD_CONFIG[1],
        block_size=16,
        max_num_blocks_per_req=2048,
        gpu_memory_utilization=0.8,
        swap_space=16,
        context_max_batch_size=BS_CONFIG[0],
        decoding_max_batch_size=BS_CONFIG[1],
        context_max_tokens_per_batch=49152,
        decoding_max_tokens_per_batch=81920,
        context_sched_policy='fcfs',
        decoding_sched_policy='fcfs',
        limit_mm_per_prompt=32
    )

    args = parser.parse_args()
    print(f'Args: {args}')
    
    set_random_seed(args.seed)
    ray.init()
    
    engine = AsyncLLM.from_engine_args(args)

    uvicorn_config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    
    async def main_coroutine():
        task2 = asyncio.create_task(uvicorn_server.serve())
        
        async def start_event_loop_wrapper():
            try:
                task = asyncio.create_task(engine.start_event_loop())
                await task
            except Exception as e:
                traceback.print_exc()
                task2.cancel()
                os._exit(1) # Kill myself, or it will print tons of errors. Don't know why.
        
        task1 = asyncio.create_task(start_event_loop_wrapper())
        
        try:
            await task2
        except:
            # This is a workaround
            # When task1 exited for some reason (e.g. error in the engine),
            # task2 will raise many exceptions, which is annoying and I do 
            # not know why
            pass
    
    asyncio.run(main_coroutine())
    