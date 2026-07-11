import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
import torch
import base64
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir = os.path.dirname(__file__)
LIB_PATH = f'{base_dir}/../build/libcuda_ipc_utils.so'
torch.ops.load_library(LIB_PATH)

MODEL_NAME = f'llama-2-7b-chat-hf/'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()

named_tensors = list(model.named_parameters())
named_tensors_dict = dict(named_tensors)


app = Flask(__name__)

print('Validating')
for name, param in model.named_parameters():
    print(f'{name}: {param.sum()} [{param.shape}]')


def get_handle_data(tensor):
    # print(f'Tensor -> {tensor} \nSum-> {tensor.sum()} \nShape -> {tensor.shape} \nDtype-> {tensor.dtype} \nNumel-> {tensor.numel()} \nELSize-> {tensor.element_size()} \nContinguous-> {tensor.is_contiguous()} \nStride-> {tensor.stride()} \nStorage Offset-> {tensor.storage_offset()}')
    data_ptr = tensor.data_ptr()
    ret_code, ipc_tensor = torch.ops.zero_copy_ops.export_tensor_ipc_handle(data_ptr)
    if ret_code != 0:
        raise RuntimeError("Export IPC handle failed")
    handle_b64 = base64.b64encode(ipc_tensor.numpy().tobytes()).decode()
    # print(f"Server tensor ptr = {tensor.data_ptr()}")
    # print(f"Server handle bytes: {ipc_tensor.numpy().tobytes().hex()}")
    handle_data = {
            "handle": handle_b64,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "size": tensor.numel() * tensor.element_size(),
            "sum": float(tensor.sum().item())
        }
    return handle_data

def get_gpu_tensor(handle_data):
    handle_bytes = base64.b64decode(handle_data["handle"])
    shape = tuple(handle_data["shape"])
    dtype_str = handle_data["dtype"].replace("torch.", "")
    ipc_tensor = torch.tensor(list(handle_bytes), dtype=torch.uint8)
    ret, ptr = torch.ops.zero_copy_ops.open_ipc_handle(ipc_tensor)
    if ret != 0:
        raise RuntimeError("Failed to open IPC handle")
    # import ctypes
    # print(f"Client IPC handle opened, ptr = {ptr}")
    # print(f"Raw handle bytes: {handle_bytes.hex()}")
    tensor = torch.ops.zero_copy_ops.tensor_from_cuda_ptr(ptr, shape, dtype_str)
    return tensor

@app.route('/get_model_weights')
def get_model_weights():
    param_name = request.args.get('param_name')
    tensor = named_tensors_dict[param_name]
    handle_data = get_handle_data(tensor)
    return jsonify(handle_data)

@app.route('/copy_manually', methods=['POST'])
def copy_manually():
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No JSON data provided"}), 400
    param_name = json_data['param_name']
    tensor = named_tensors_dict[param_name]
    client_tensor = get_gpu_tensor(handle_data=json_data)
    # print(f'\n\nClient Tensor {client_tensor.sum()}, Server tensor {tensor.sum()}')
    # client_tensor.copy_(tensor)
    client_tensor.copy_(tensor, non_blocking=True)
    # print(f'Client Tensor {client_tensor.sum()}, Server tensor {tensor.sum()}')
    return jsonify({"client_sum": client_tensor.sum().item(), 'server_sum': tensor.sum().item()})

if __name__ == '__main__':
    app.run(port=5000)