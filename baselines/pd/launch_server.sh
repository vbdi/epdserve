#!bin/bash

# cd EPD-Disaggregation #(run from porject root)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DEDUP_LOGS=0
# export PYTHONLOGGING=DEBUG

PD_CONFIG="7,0"
BS_CONFIG="1,1"
MODEL_PATH=../../assets/models/MiniCPM-V-2_6

IFS=',' read -ra BS_ARRAY <<< "$BS_CONFIG"
IFS=',' read -ra PD_ARRAY <<< "$PD_CONFIG"

PYTHONPATH=$PYTHONPATH:./ python -u distserve/api_server.py \
                          --host localhost \
                          --port 8400 \
                          --model $MODEL_PATH \
                          --tokenizer $MODEL_PATH \
                          --context-data-parallel-size ${PD_ARRAY[0]} \
                          --decoding-data-parallel-size ${PD_ARRAY[1]} \
                          --block-size 16 \
                          --max-num-blocks-per-req 2048 \
                          --gpu-memory-utilization 0.5 \
                          --swap-space 16 \
                          --context-max-batch-size ${BS_ARRAY[0]} \
                          --decoding-max-batch-size ${BS_ARRAY[1]} \
                          --context-max-tokens-per-batch 49152 \
                          --decoding-max-tokens-per-batch 81920 \
                          --context-sched-policy fcfs \
                          --decoding-sched-policy fcfs \
                          --limit-mm-per-prompt 32
                          

# --distributed_executor_backend ray
# python internvl_online.py
# --max-num-blocks-per-req 128 \q