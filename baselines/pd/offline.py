import os, sys, time
import sys
sys.path.append("..") 
os.environ["RAY_DEDUP_LOGS"] = "0"

from nvitop import select_devices
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu_id) for gpu_id in sorted(select_devices(format='index', min_count=3, min_free_memory='75GiB')))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(f'\n################# CUDA_VISIBLE_DEVICES [{os.environ["CUDA_VISIBLE_DEVICES"]}] ###################\n')

import distserve
print(f'DISTSERVE CODE LOCATION {distserve.__path__}')
from distserve import OfflineLLM, SamplingParams
from distserve.config import (ModelConfig, DisaggParallelConfig, ClusterParallelConfig, CacheConfig, ContextStageSchedConfig, DecodingStageSchedConfig)
from vllm import EngineArgs
from PIL import Image
from transformers import AutoTokenizer
import argparse

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '../../assets')
MODEL_NAME = os.path.join(ASSETS_DIR, 'models/MiniCPM-V-2_6')
DP_CONFIG = '7,1'
BS_CONFIG = '1,128'
NUM_REQS = 100
NUM_IMGS = 2

# question = f"I have provided multiple pictures in the input. I want you to decribe the details of the picture. While describing, please make sure you do not miss any important details. Please keep the output concise yet it should also be meaningful and have a lot of insight. Further, make sure you explain things that can be missed by an average individual."
question = f"Describe the image."
question = question
image = Image.open(f"{ASSETS_DIR}/images/image2_4032_3024.jpg").convert("RGB")
print(f"image resolution: {image.size}")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='The model to use', default=MODEL_NAME)
parser.add_argument('--num_reqs', type=int, help='No of reqs', default=NUM_REQS)
parser.add_argument('--num_imgs', type=int, help='No of imgs', default=NUM_IMGS)
parser.add_argument('--dp_config', type=str, help='DP config', default=DP_CONFIG)
parser.add_argument('--bs_config', type=str, help='DP config', default=BS_CONFIG)
args = parser.parse_args()
print(f'Running EPD with args {args}')


vllm_engine_args = EngineArgs(model=args.model_name,
                              max_num_seqs=8,
                              tensor_parallel_size=1,
                              dtype="float16",
                              enforce_eager=True,
                              quantization=None,
                              trust_remote_code=True,
                              limit_mm_per_prompt={"image": args.num_imgs})
vllm_engine_config = vllm_engine_args.create_engine_config()

exp_suffix=''
run_name = f'{os.path.basename(args.model_name).split("-")[0]}|R{args.num_reqs}|O{args.num_imgs}|{args.dp_config}|{args.bs_config}|{exp_suffix}'

args.dp_config = [int(val) for val in args.dp_config.split(',')]
args.bs_config = [int(val) for val in args.bs_config.split(',')]

llm = OfflineLLM(
    model_config=ModelConfig(
        model=args.model_name,
        dtype="fp16",
        tokenizer=None if "llava" in args.model_name else args.model_name,
        trust_remote_code=True,
        vllm_config=vllm_engine_config),

    disagg_parallel_config=DisaggParallelConfig(
        context=ClusterParallelConfig(
            data_parallel_size=args.dp_config[0],
            tensor_parallel_size=1,
            pipeline_parallel_size=1),

        decoding=ClusterParallelConfig(
            data_parallel_size=args.dp_config[1],
            tensor_parallel_size=1,
            pipeline_parallel_size=1),
        ),

    cache_config=CacheConfig(
        block_size=16,
        max_num_blocks_per_req=2048,
        gpu_memory_utilization=0.8,
        cpu_swap_space=1
    ),

    context_sched_config=ContextStageSchedConfig(
        policy="fcfs",
        max_batch_size=args.bs_config[0],
        max_tokens_per_batch=16384*3
    ),

    decoding_sched_config=DecodingStageSchedConfig(
        policy="fcfs",
        max_batch_size=args.bs_config[1],
        max_tokens_per_batch=16384*5,
        waiting_block_prop_threshold=0.05
    )
)

sampling_params = SamplingParams(temperature=0, max_tokens=1)
content_string = '(<image>./</image>)' * args.num_imgs
messages = [{
    'role': 'user',
    'content': f'{content_string}\n{question}'
}]
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = [{
        "prompt": prompt,
        "multi_modal_data": {
            "image": [image] * args.num_imgs
        },
    } for _ in range(args.num_reqs)]

outputs = llm.generate(inputs, sampling_params=sampling_params)
llm.print_outputs()
llm.dump_lifetime_events()
llm.dump_tb_metrics()
llm.dump_run_stats()
