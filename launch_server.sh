#!bin/bash
# cd EPD-Disaggregation #(run from porject root)
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export RAY_DEDUP_LOGS=0

EPD_CONFIG="2,2,2"
BS_CONFIG="1,1,4"
INTRA_REQ_DP=0
MODEL_PATH=./assets/models/MiniCPM-V-2_6

IFS=',' read -ra BS_ARRAY <<< "$BS_CONFIG"
IFS=',' read -ra EPD_ARRAY <<< "$EPD_CONFIG"

PYTHONPATH=$PYTHONPATH:./ python -u epdserve/api_server.py \
                          --host localhost \
                          --port 8400 \
                          --model $MODEL_PATH \
                          --tokenizer $MODEL_PATH \
                          --encoding-data-parallel-size ${EPD_ARRAY[0]} \
                          --intra-request-dp $INTRA_REQ_DP \
                          --context-data-parallel-size ${EPD_ARRAY[1]} \
                          --decoding-data-parallel-size ${EPD_ARRAY[2]} \
                          --block-size 16 \
                          --max-num-blocks-per-req 2048 \
                          --gpu-memory-utilization 0.8 \
                          --swap-space 16 \
                          --encoding-max-batch-size ${BS_ARRAY[0]} \
                          --context-max-batch-size ${BS_ARRAY[1]} \
                          --decoding-max-batch-size ${BS_ARRAY[2]} \
                          --context-max-tokens-per-batch 49152 \
                          --decoding-max-tokens-per-batch 81920 \
                          --encoding-sched-policy fcfs \
                          --context-sched-policy fcfs \
                          --decoding-sched-policy fcfs \
                          --limit-mm-per-prompt 32

# python internvl_online.py
# --max-num-blocks-per-req 128 \q