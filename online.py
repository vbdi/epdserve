import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Optional, Any
import os
import sys
import aiohttp
import numpy as np
from tqdm import tqdm
import epdserve
from epdserve import utils
from epdserve.utils import TestRequest, Dataset, RequestResult
from PIL import Image

BACKEND_TO_PORTS = {
    "vllm": 8100,
    "lightllm": 8200,
    "deepspeed": 8300,
    "distserve": 8500
}

pbar: Optional[tqdm] = None

def set_seed(seed: int = 6):
    random.seed(seed) # Python's built-in random
    np.random.seed(seed) # NumPy

def sample_requests(num_prompts: int, model: str, prompt:str, nimg=32, image_path="", output_len=1, seed=6) -> List[TestRequest]:
    set_seed(seed)
    cnt=0

    for idx in range(num_prompts): 
        if cnt>=num_prompts: break 
        
        if "llava" in model:
            image_token = "<image>"
        elif "InternVL2" in model:
            image_token = "<image>"
        elif "MiniCPM" in model:
            image_token = "(<image>./</image>)"
        else:
            raise NotImplementedError
        
        req = TestRequest(prompt=image_token * nimg + prompt,
                          prompt_len=-1,
                          output_len=output_len,
                        #   images=[Image.open(image_path).convert("RGB")] * nimg)        
                          images_paths=[image_path] * nimg)        

        cnt+=1 
        yield req 

async def get_request(
    input_requests: List[TestRequest],
    process_name: str = "possion",
    request_rate: float = 1.0,
    cv: float = 1.0,
) -> AsyncGenerator[TestRequest, None]:
    interval_lens = len(input_requests)
    input_requests = iter(input_requests)

    if request_rate not in [float("inf"), 0.0]:
        if process_name == "uniform":
            intervals = [1.0 / request_rate for _ in range(interval_lens)]
        elif process_name == "gamma":
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        elif process_name == "possion":
            cv = 1
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        elif process_name=='right_away':
            intervals = np.zeros(interval_lens)
        else:
            raise ValueError(
                f"Unsupported prosess name: {process_name}, we currently support uniform, gamma and possion."
            )
    for idx, request in enumerate(input_requests):
        yield request
        if request_rate == float("inf") or request_rate == 0.0:
            continue

        interval = intervals[idx]
        await asyncio.sleep(interval) # The next request will be sent after the interval.


async def send_request(
    backend: str,
    api_url: str,
    request,
    best_of: int,
    use_beam_search: bool,
    verbose: bool,
) -> RequestResult:
    global pbara

    prompt: str = request.prompt
    prompt_len: int = request.prompt_len
    output_len: int = request.output_len
    images: List[Any] = request.images
    images_paths: List[str] = request.images_paths

    headers = {"User-Agent": "Benchmark Client"}
    aux = {"n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
            }
    
    if backend in ["distserve","vllm"]:
        pload = {"input": {"prompt": prompt, 
                           "multi_modal_data":{"images_paths":images_paths}}, 
                        #    "multi_modal_data":{"images": [utils.image_to_base64(image) for image in images]}}, 
                           **aux}
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # llava-1.5 The maximum length of the input is 4096, limited by the embedding
    # table size. (max_position_embeddings)
    # assert prompt_len+output_len < 4096
    
    request_start_time = time.perf_counter()
    request_output = None

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            try:
                output = json.loads(output)
            except:
                print("Failed to parse the response:")
                print(output)
                continue
            if verbose:
                print(f"Prompt: {prompt}\n\nOutput: {output['text']}")

            # Re-send the request if it failed.
            if "error" not in output:
                request_output = output
                break
            else:
                print(f"Failed to process the request: {output['error']}")
                print(f"Resending the request: {pload}")

    request_end_time = time.perf_counter()
    pbar.update(1)
    request_result = RequestResult(
        # prompt_len,
        # request_output['prompt_token_len'],
        request_output['processed_prompt_token_len'],
        output_len,
        request_start_time,
        request_end_time,
        token_timestamps=request_output["timestamps"],
        lifetime_events=request_output.get("lifetime_events", None)
    )
    return request_result

async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[TestRequest],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    request_cv: float = 1.0,
    process_name: str = "possion",
    verbose: bool = False
) -> List[RequestResult]:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, process_name, request_rate, request_cv):
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                request,
                best_of,
                use_beam_search,
                verbose
            )
        )
        tasks.append(task)
    request_results = await asyncio.gather(*tasks)
    return request_results

def main(args: argparse.Namespace):
    print(args)
    set_seed(args.seed)

    # input_requests = list(sample_requests(args.dataset, args.num_prompts, args.model))
    input_requests = list(sample_requests(num_prompts=args.num_prompts, 
                                          model=args.model,
                                          prompt=args.prompt,
                                          nimg=args.num_imgs,
                                          image_path=args.image_path,
                                          output_len=args.output_len))

    input_requests[0].__dataclass_fields__.keys()

    print("Sampling done. Start benchmarking...")

    global pbar
    pbar = tqdm(total=args.num_prompts)
    benchmark_start_time = time.perf_counter()
    api_url = f"http://{args.host}:{args.port}/generate"
    request_results = asyncio.run(
        benchmark(args.backend, api_url, input_requests, args.best_of, args.use_beam_search, 
                  args.request_rate, args.request_cv, args.process_name, args.verbose)
    )
    benchmark_end_time = time.perf_counter()
    pbar.close()
    
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"System: Total time: {benchmark_time:.2f} s")
    print(f"System: Request throughput: {args.num_prompts / benchmark_time:.2f} requests/s")
    print(f"System: Overall Throughput: {sum([req.prompt_len + req.output_len for req in request_results]) / benchmark_time:.2f} tokens/s")
    print(f"System: Prefill Throughput: {sum([req.prompt_len for req in request_results]) / benchmark_time:.2f} tokens/s")
    print(f"System: Decode Throughput: {sum([req.output_len-1 for req in request_results]) / benchmark_time:.2f} output tokens/s")
    
    print(f"Client: Mean Client Latency {np.mean([req_res.latency for req_res in request_results])}")
    
    print(f"Server: Mean Server Latency {np.mean([req_res.server_latency for req_res in request_results])}")
    print(f"Server: Mean TTFT {np.mean([req_res.ttft for req_res in request_results])}")
    print(f"Server: Mean TPOT {np.mean([req_res.tpot for req_res in request_results])}")

    with open(args.output, "w") as f:
        json.dump(request_results, f, default=vars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="distserve", choices=["distserve", "vllm", "deepspeed"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--dataset", type=str, help="Path to the (preprocessed) dataset.")
    parser.add_argument("--best-of", type=int, default=1, help="Generates `best_of` sequences per prompt and " "returns the best one.",)
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts-req-rates", type=str, help="[(num_prompts, request_rate), ...] where num_prompts is the number of prompts to process and request_rate is the number of requests per second.",)
    parser.add_argument("--request-cv", type=float, default=1.0, help="the coefficient of variation of the gap between" "the requests.",)
    parser.add_argument("--process-name", type=str, default="possion", choices=["possion", "gamma", "uniform"],)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--trust-remote-code", action="store_true", help="trust remote code from huggingface",)
    parser.add_argument("--exp-result-root", type=str, default=None, help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: env var EXP_RESULT_ROOT)")
    parser.add_argument("--exp-result-dir", type=str, help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: <model_name>-<dataset.name>)")
    parser.add_argument("--exp-result-prefix", type=str, default=None, help="Exp result file will be named as <exp-result-prefix>-<num-prompts>-<req-rate>.exp (default: <backend>)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs (prompts and outputs).")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--num_imgs", type=int, help="Number of images")
    parser.add_argument("--image_path", type=str, help="Path of image")
    parser.add_argument("--prompt", type=str, help="Prompt")
    parser.add_argument("--output_len", type=int, help="output len/ max num tokens generated")
    
    parser.set_defaults(
        model='MiniCPM', #'MiniCPM', 'llava', "InternVL2"
        backend='distserve', # "vllm", # 
        num_prompts_req_rates=[(20,2)], #(num_prompts, request_rate)
        process_name='possion', #right_away
        exp_result_dir='test',
        exp_result_root=f'{os.path.dirname(__file__)}/experiments/outputs/',
        port=8400,
        num_imgs=1,
        image_path=f'{os.path.dirname(__file__)}/assets/images/image2_4032_3024.jpg',
        prompt=f"Describe the image.",
        output_len=500
    )
    args = parser.parse_args()
    print(f'\nRunning online script for args: {args}\n')
        
    if args.exp_result_prefix == None:
        args.exp_result_prefix = args.backend
        
    if args.port == None:
        args.port = BACKEND_TO_PORTS[args.backend]
    
    if isinstance(args.num_prompts_req_rates, str):
        num_prompts_request_rates = eval(args.num_prompts_req_rates)
    else:
        num_prompts_request_rates = args.num_prompts_req_rates

    for (num_prompts, request_rate) in num_prompts_request_rates:
        num_prompts=int(num_prompts)
        request_rate=float(request_rate)
        print("===================================================================")
        print(f"Running with num_prompts={num_prompts}, request_rate={request_rate}")
        args.num_prompts = num_prompts
        args.request_rate = request_rate
        output_dir = os.path.join(args.exp_result_root, args.exp_result_dir)
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{args.exp_result_prefix}-{num_prompts}-{request_rate}.exp")
        main(args) # actually runs the benchmark
        time.sleep(1)
        