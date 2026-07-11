import subprocess
import json
import glob
import psutil
import os
import time

TIMEOUT = 60*60*4 # 4 hours

def delete_tmp_files():
    # First, run the shell command to remove files
    files_to_remove = glob.glob('/tmp/err_execute_model_input_*')
    for file_path in files_to_remove:
        try:
            subprocess.run(['rm', '-rf', file_path], check=True)
            print(f"File {file_path} removed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove files: {e}")

def run_online(server_script_path, server_args, logs_dir, client_script_path, client_args, conda_env = 'distserve'):
    delete_tmp_files()
    os.makedirs(logs_dir, exist_ok=True) 

    additional_python_path = os.path.dirname(os.path.dirname(server_script_path))
    new_python_path = f"{additional_python_path}:{os.environ.get('PYTHONPATH', '')}"

    server_command = [
        "bash", "-c",
        f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {conda_env} && PYTHONPATH={additional_python_path}:$PYHTONPATH python3 -u {server_script_path} {' '.join(server_args)}"
    ]

    print(f'\n\nStarting server: {" ".join(server_command)}')

    server_stdout_log = open(f'{logs_dir}/server_stdout.log', 'w')
    server_stderr_log = open(f'{logs_dir}/server_stderr.log', 'w')

    # Start the server in the background
    server_process = subprocess.Popen(
        server_command,
        stdout=server_stdout_log,
        stderr=server_stderr_log,    
        text=True,
        env={**os.environ, "PYTHONPATH": new_python_path, "PYTHONUNBUFFERED": "1"}
    )

    # Wait for the server to start (optional, to ensure it's ready before the client runs)
    time.sleep(180)

    client_command = ['conda', 'run', '-n', conda_env, 'python3', '-u', client_script_path] + client_args
    print(f'\n\nStarting Client: {" ".join(client_command)}')

    try:
        # Run the client in blocking mode
        client_result = subprocess.run(
            client_command,

            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,            
            text=True,
            timeout=TIMEOUT
        )

        print(f"Client Output:\n{client_result.stdout}")
        # print(f"Client Error (if any):\n{client_result.stderr}")

        with open(f'{logs_dir}/client_stdout.log', 'w') as file:
            file.write(client_result.stdout)
        with open(f'{logs_dir}/client_stderr.log', 'w') as file:
            file.write(client_result.stderr)        

    except subprocess.TimeoutExpired:
        print("Client process timed out.")

    finally:
        print("Shutting down the server...")
        server_stdout_log.flush()
        server_stderr_log.flush()
        server_stdout_log.close()
        server_stderr_log.close()

        try:
            parent_pid = server_process.pid
            parent = psutil.Process(parent_pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            print("Server terminated successfully.")
        except Exception as e:
            print(f"Error terminating server: {e}")
    print('DONE')

start_time = time.perf_counter()
base_folder = os.path.dirname(__file__)
MODEL_PATH=f'{base_folder}/assets/models/MiniCPM-V-2_6'
REQ_FREQ=f'np.array([[100,0.25],[100,0.25],[100,0.5],[100,1.0],[100,1.5],[100,2.0]])'

IMAGE_PATH=f'{base_folder}/assets/images/image2_4032_3024.jpg'
PROMPT="Describe the images"
OUTPUT_LEN=10
KV_CACHE_UTIL=0.5

for IMAGES_PER_REQ in [2]:
    RESULTS_ROOT_DIR=f'{base_folder}/experiments/intra_req_dp_e2e/imgs{IMAGES_PER_REQ}'



    ### EPD
    SERVER_SCRIPT_PATH = f'{base_folder}/epdserve/api_server.py'
    CLIENT_SCRIPT_PATH = f'{base_folder}/online.py'
    EPD_CONFIG=[6,1,1]
    BS_CONFIG=[1,1,1]

    INTRA_REQ_DP=1
    RESULTS_DIR='epd'

    server_args = f''' \
--host localhost \
--port 8400 \
--model {MODEL_PATH} \
--tokenizer {MODEL_PATH} \
--encoding-data-parallel-size {EPD_CONFIG[0]} \
--intra-request-dp {INTRA_REQ_DP} \
--context-data-parallel-size {EPD_CONFIG[1]} \
--decoding-data-parallel-size {EPD_CONFIG[2]} \
--block-size 16 \
--max-num-blocks-per-req 2048 \
--gpu-memory-utilization {KV_CACHE_UTIL} \
--swap-space 16 \
--encoding-max-batch-size {BS_CONFIG[0]} \
--context-max-batch-size {BS_CONFIG[1]} \
--decoding-max-batch-size {BS_CONFIG[2]} \
--context-max-tokens-per-batch 49152 \
--decoding-max-tokens-per-batch 81920 \
--encoding-sched-policy fcfs \
--context-sched-policy fcfs \
--decoding-sched-policy fcfs \
--limit-mm-per-prompt 32 \
'''.strip().split(' ')


    client_args = f'''\
--num-prompts-req-rates {REQ_FREQ} \
--exp-result-root {RESULTS_ROOT_DIR} \
--exp-result-dir {RESULTS_DIR} \
--num_imgs {IMAGES_PER_REQ} \
--image_path {IMAGE_PATH} \
--output_len {OUTPUT_LEN}
'''.strip().split(' ') + ['--prompt', f'"{PROMPT}"']


    run_online(server_script_path=SERVER_SCRIPT_PATH,
                server_args=server_args, 
                logs_dir=f'{RESULTS_ROOT_DIR}/{RESULTS_DIR}', 
                client_script_path=CLIENT_SCRIPT_PATH, 
                client_args=client_args)


    ### PD
    SERVER_SCRIPT_PATH = f'{base_folder}/baselines/pd/distserve/api_server.py'
    CLIENT_SCRIPT_PATH = f'{base_folder}/baselines/pd/online.py'
    EPD_CONFIG=[7,1]
    BS_CONFIG=[1,1]

    RESULTS_DIR='pd'

    server_args = f''' \
--host localhost \
--port 8400 \
--model {MODEL_PATH} \
--tokenizer {MODEL_PATH} \
--context-data-parallel-size {EPD_CONFIG[0]} \
--decoding-data-parallel-size {EPD_CONFIG[1]} \
--block-size 16 \
--max-num-blocks-per-req 2048 \
--gpu-memory-utilization {KV_CACHE_UTIL} \
--swap-space 16 \
--context-max-batch-size {BS_CONFIG[0]} \
--decoding-max-batch-size {BS_CONFIG[1]} \
--context-max-tokens-per-batch 49152 \
--decoding-max-tokens-per-batch 81920 \
--context-sched-policy fcfs \
--decoding-sched-policy fcfs \
--limit-mm-per-prompt 32 \
'''.strip().split(' ')

    client_args = f'''\
--num-prompts-req-rates {REQ_FREQ} \
--exp-result-root {RESULTS_ROOT_DIR} \
--exp-result-dir {RESULTS_DIR} \
--num_imgs {IMAGES_PER_REQ} \
--image_path {IMAGE_PATH}
'''.strip().split(' ') + ['--prompt', f'"{PROMPT}"']

    run_online(server_script_path=SERVER_SCRIPT_PATH,
                server_args=server_args, 
                logs_dir=f'{RESULTS_ROOT_DIR}/{RESULTS_DIR}', 
                client_script_path=CLIENT_SCRIPT_PATH, 
                client_args=client_args)



    ### D
    SERVER_SCRIPT_PATH = f'{base_folder}/baselines/d/distserve/api_server.py'
    CLIENT_SCRIPT_PATH = f'{base_folder}/baselines/d/online.py'
    EPD_CONFIG=[8]
    BS_CONFIG=[1]

    RESULTS_DIR='d'
    server_args = f''' \
--host localhost \
--port 8400 \
--model {MODEL_PATH} \
--tokenizer {MODEL_PATH} \
--decoding-data-parallel-size {EPD_CONFIG[0]} \
--block-size 16 \
--max-num-blocks-per-req 2048 \
--gpu-memory-utilization {KV_CACHE_UTIL} \
--swap-space 16 \
--decoding-max-batch-size {BS_CONFIG[0]} \
--decoding-max-tokens-per-batch 81920 \
--decoding-sched-policy fcfs \
--limit-mm-per-prompt 32 \
'''.strip().split(' ')

    client_args = f'''\
--num-prompts-req-rates {REQ_FREQ} \
--exp-result-root {RESULTS_ROOT_DIR} \
--exp-result-dir {RESULTS_DIR} \
--num_imgs {IMAGES_PER_REQ} \
--image_path {IMAGE_PATH} \
--output_len {OUTPUT_LEN}
'''.strip().split(' ') + ['--prompt', f'"{PROMPT}"']

    run_online(server_script_path=SERVER_SCRIPT_PATH,
                server_args=server_args, 
                logs_dir=f'{RESULTS_ROOT_DIR}/{RESULTS_DIR}', 
                client_script_path=CLIENT_SCRIPT_PATH, 
                client_args=client_args)

print(f'\n\nAll DONE. Total time taken - {time.perf_counter()-start_time}')
