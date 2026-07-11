import torch 
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import torch

base_folder = os.path.dirname(__file__)
LIB_PATH = f'{base_folder}/build/lib.linux-x86_64-cpython-310/cuda_ipc_utils.cpython-310-x86_64-linux-gnu.so'
torch.ops.load_library(LIB_PATH)
print(f"Loaded library from: {LIB_PATH}")

print()
# print(dir(torch.ops.cuda_ipc_utils))

tensor = torch.randn(50, device='cuda')

# print(torch.ops.block_migration_ops.dummy_function(10))

ops = ["get_ipc_mem_handle", 
"register_ipc_mem_handle_context",
"unregister_ipc_mem_handle_context",
"migrate_blocks_context",
"register_ipc_mem_handle_encoding",
"migrate_blocks_encoding",]

for op in ops:
    result = hasattr(torch.ops.block_migration_ops, op)
    if result:
        print(f'Operator exists {op}')

op = 'generate_nccl_id'
result = hasattr(torch.ops.nccl_ops, op)
if result:
    print(f'Operator exists {op}')

print(torch.ops.nccl_ops.generate_nccl_id())


exit()





# tensor = torch.randn(10, device='cuda')
# data_ptr = tensor.data_ptr()
# ret_code, ipc_tensor = torch.ops.cuda_ipc_utils.export_tensor_ipc_handle(data_ptr)
# if ret_code != 0:
#     raise RuntimeError("Export IPC handle failed")
# print(ipc_tensor)


# torch.ops.block_migration_ops.get_ipc_mem_handle

ops = ["get_ipc_mem_handle", 
"register_ipc_mem_handle_context",
"unregister_ipc_mem_handle_context",
"migrate_blocks_context",
"register_ipc_mem_handle_encoding",
"migrate_blocks_encoding",]

for op in ops:
    result = hasattr(torch.ops.block_migration_ops, op)
    if result:
        print(f'Operator exists {op}')

op = 'generate_nccl_id'
result = hasattr(torch.ops.nccl_ops, op)
if result:
    print(f'Operator exists {op}')


# nccl_id = torch.ops.nccl_ops.generate_nccl_id()

# nccl_id = torch.ops.nccl_ops.generate_nccl_id()

# nccl_id = torch.ops.nccl_ops.generate_nccl_id()



# nccl_id = torch.ops.nccl_ops.generate_nccl_id()
# print(nccl_id)
