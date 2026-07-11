#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

int64_t export_tensor_ipc_handle_c(void* tensor_ptr, char* ipc_handle_out) {
    cudaIpcMemHandle_t handle;
    cudaError_t err = cudaIpcGetMemHandle(&handle, tensor_ptr);
    if (err != cudaSuccess) return -1;
    std::memcpy(ipc_handle_out, &handle, sizeof(handle));
    return 0;
}

int64_t open_ipc_handle_c(char* ipc_handle_in, void** d_ptr_out) {
    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, ipc_handle_in, sizeof(handle));
    cudaError_t err = cudaIpcOpenMemHandle(d_ptr_out, handle, cudaIpcMemLazyEnablePeerAccess);
    return err == cudaSuccess ? 0 : -2;
}

int64_t close_ipc_handle_c(void* d_ptr) {
    cudaError_t err = cudaIpcCloseMemHandle(d_ptr);
    return err == cudaSuccess ? 0 : -3;
}

std::tuple<int64_t, at::Tensor> export_tensor_ipc_handle(int64_t ptr) {
    std::vector<uint8_t> handle_buf(64);
    int64_t ret = export_tensor_ipc_handle_c(reinterpret_cast<void*>(ptr), reinterpret_cast<char*>(handle_buf.data()));
    auto tensor = torch::tensor(handle_buf, torch::kUInt8);
    return {ret, tensor};
}

std::tuple<int64_t, int64_t> open_ipc_handle(at::Tensor handle_tensor) {
    if (handle_tensor.numel() != 64) throw std::runtime_error("Invalid IPC handle tensor size");
    void* ptr;
    int64_t ret = open_ipc_handle_c(reinterpret_cast<char*>(handle_tensor.data_ptr()), &ptr);
    return {ret, reinterpret_cast<int64_t>(ptr)};
}

int64_t close_ipc_handle(int64_t ptr) {
    return close_ipc_handle_c(reinterpret_cast<void*>(ptr));
}

at::Tensor tensor_from_cuda_ptr(int64_t ptr, std::vector<int64_t> shape, std::string dtype_str) {
    void* data_ptr = reinterpret_cast<void*>(ptr);
    c10::ScalarType dtype;

    if (dtype_str == "float32" || dtype_str == "float" || dtype_str == "torch.float32")
        dtype = torch::kFloat32;
    else if (dtype_str == "float16" || dtype_str == "torch.float16" || dtype_str == "half")
        dtype = torch::kFloat16;
    else if (dtype_str == "float64" || dtype_str == "torch.float64")
        dtype = torch::kFloat64;
    else if (dtype_str == "int32" || dtype_str == "torch.int32")
        dtype = torch::kInt32;
    else if (dtype_str == "int64" || dtype_str == "torch.int64")
        dtype = torch::kInt64;
    else
        throw std::runtime_error("Unsupported dtype: " + dtype_str);

    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    return torch::from_blob(data_ptr, shape, options);
    // return torch::from_blob(data_ptr, shape, options).clone();
}

at::Tensor allocate_ipc_safe_tensor(std::vector<int64_t> shape, std::string dtype_str) {
    c10::ScalarType dtype;
    if (dtype_str == "float16") dtype = torch::kFloat16;
    else throw std::runtime_error("Only float16 implemented");

    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);

    size_t numel = 1;
    for (auto s : shape) numel *= s;
    size_t size_bytes = numel * 2; // float16

    void* ptr;
    cudaMalloc(&ptr, size_bytes);  // <-- true cudaMalloc

    auto deleter = [](void* ptr) { cudaFree(ptr); };
    auto tensor = torch::from_blob(ptr, shape, deleter, options);
    return tensor;
}

TORCH_LIBRARY(zero_copy_ops, m) {
    m.def("export_tensor_ipc_handle(int ptr) -> (int, Tensor)");
    m.impl("export_tensor_ipc_handle", export_tensor_ipc_handle);

    m.def("open_ipc_handle(Tensor handle) -> (int, int)");
    m.impl("open_ipc_handle", open_ipc_handle);

    m.def("close_ipc_handle(int ptr) -> int");
    m.impl("close_ipc_handle", close_ipc_handle);

    m.def("tensor_from_cuda_ptr(int ptr, int[] shape, str dtype) -> Tensor");
    m.impl("tensor_from_cuda_ptr", tensor_from_cuda_ptr);

    m.def("allocate_ipc_safe_tensor(int[] shape, str dtype) -> Tensor");
    m.impl("allocate_ipc_safe_tensor", allocate_ipc_safe_tensor);

}
