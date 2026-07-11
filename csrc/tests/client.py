import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
import torch
import base64
import requests
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


base_dir = os.path.dirname(__file__)
LIB_PATH = f'{base_dir}/../build/libcuda_ipc_utils.so'
torch.ops.load_library(LIB_PATH)

# Initialize the model with the configuration on CPU (WEIGHTS NOT LOADED)
MODEL_NAME = "llama-2-7b-chat-hf/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_config(config)

print('MODEL LAODED ON CPU WITHOUT WEIGHTS.')
print('NOT FETCHING WEIGHTS USING ZERO COPY FROM SERVER.')

def get_handle_data(input_tensor):
    # print(f'Tensor -> {input_tensor} \nSum-> {tensor.sum()} \nShape -> {input_tensor.shape} \nDtype-> {input_tensor.dtype} \nNumel-> {input_tensor.numel()} \nELSize-> {input_tensor.element_size()} \nContinguous-> {input_tensor.is_contiguous()} \nStride-> {input_tensor.stride()} \nStorage Offset-> {input_tensor.storage_offset()}')
    # tensor = torch.randn(10, dtype=torch.float16, device='cuda')
    # tensor = torch.zeros_like(tensor)
    data_ptr = input_tensor.data_ptr()
    ret_code, ipc_tensor = torch.ops.zero_copy_ops.export_tensor_ipc_handle(data_ptr)
    if ret_code != 0:
        raise RuntimeError("Export IPC handle failed")
    handle_b64 = base64.b64encode(ipc_tensor.numpy().tobytes()).decode()
    # print(f"Server tensor ptr = {tensor.data_ptr()}")
    # print(f"Server handle bytes: {ipc_tensor.numpy().tobytes().hex()}")
    handle_data = {
            "handle": handle_b64,
            "shape": list(input_tensor.shape),
            "dtype": str(input_tensor.dtype),
            "size": input_tensor.numel() * input_tensor.element_size(),
            "sum": float(input_tensor.sum().item())
        }
    return handle_data

for name, param in model.named_parameters():
    resp = requests.get(f"http://localhost:5000/get_model_weights?param_name={name}")
    info = resp.json()

    handle_bytes = base64.b64decode(info["handle"])
    shape = tuple(info["shape"])
    dtype_str = info["dtype"].replace("torch.", "")

    ipc_tensor = torch.tensor(list(handle_bytes), dtype=torch.uint8)

    ret, ptr = torch.ops.zero_copy_ops.open_ipc_handle(ipc_tensor)
    if ret != 0:
        raise RuntimeError("Failed to open IPC handle")

    tensor = torch.ops.zero_copy_ops.tensor_from_cuda_ptr(ptr, shape, dtype_str)
    # print(f"{name}[{param.sum()}, {param.shape}, {param.device}] -> {name}[{tensor.sum()}, {tensor.shape}, {tensor.device}]")

    if tensor.sum() - info['sum'] < 0.1:
        param.data = tensor
    else:
        url = "http://localhost:5000/copy_manually"
        print(f'Failed for {name}. Copying manually')
        tensor = torch.rand_like(param.data, device='cuda')                        
        tensor = torch.ops.zero_copy_ops.allocate_ipc_safe_tensor(param.data.shape, "float16") 

        # tensor = torch.zeros(param.data.shape*2, device='cuda')
        handle_data = get_handle_data(tensor)
        handle_data['param_name'] = name
        # print(f"Tensor ptr = {tensor.data_ptr()}")
        # print(handle_data)
        response = requests.post(url, json=handle_data)
        print(f"{response} {tensor.sum()} {info['sum']}")
        if tensor.sum() - info['sum'] < 0.1:
            param.data = tensor
        else:
            print('Model copying failed even after copying')
            exit()

print('Validating')
for name, param in model.named_parameters():
    print(f'{name}: {param.sum()} [{param.shape}]')
