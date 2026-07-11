#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

// =======================
// Custom NCCLUniqueId Wrapper
// =======================

struct NCCLUniqueIdWrapper : torch::CustomClassHolder {
    ncclUniqueId id;

    NCCLUniqueIdWrapper() {
        ncclGetUniqueId(&id);
    }

    std::vector<int64_t> get_raw() {
        std::vector<int64_t> raw(sizeof(ncclUniqueId) / sizeof(int));
        std::memcpy(raw.data(), &id, sizeof(ncclUniqueId));
        return raw;
    }

    void set_raw(std::vector<int64_t> raw) {
        TORCH_CHECK(raw.size() == sizeof(ncclUniqueId) / sizeof(int), "Invalid raw ID size");
        std::memcpy(&id, raw.data(), sizeof(ncclUniqueId));
    }
};

// =======================
// Helper to get NCCL dtype
// =======================

ncclDataType_t get_nccl_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat: return ncclFloat;
        case at::kHalf: return ncclHalf;
        case at::kByte: return ncclUint8;
        case at::kInt: return ncclInt;
        case at::kLong: return ncclInt64;
        case at::kDouble: return ncclDouble;
        default: throw std::runtime_error("Unsupported dtype for NCCL");
    }
}

// =======================
// NCCL Copy Function
// =======================

at::Tensor async_tensor_copy_nccl(at::Tensor src_tensor, int64_t src_rank, int64_t dst_rank, c10::intrusive_ptr<NCCLUniqueIdWrapper> id_wrapper) {
    TORCH_CHECK(src_tensor.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(src_tensor.is_contiguous(), "Tensor must be contiguous");

    int64_t numel = src_tensor.numel();
    auto dtype = src_tensor.scalar_type();
    auto shape = src_tensor.sizes().vec();
    auto nccl_dtype = get_nccl_dtype(dtype);

    at::Tensor dst_tensor = torch::empty(shape, src_tensor.options());

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ncclComm_t comm;
    ncclCommInitRank(&comm, 2, id_wrapper->id, dst_rank);

    ncclGroupStart();
    if (dst_rank == 0) {
        ncclRecv(dst_tensor.data_ptr(), numel, nccl_dtype, 1, comm, stream);
    } else if (dst_rank == 1) {
        ncclSend(src_tensor.data_ptr(), numel, nccl_dtype, 0, comm, stream);
    }
    ncclGroupEnd();

    cudaStreamSynchronize(stream);
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);

    return dst_tensor;
}

// =======================
// Torch Bindings
// =======================

TORCH_LIBRARY(nccl_copy_ops, m) {
    m.class_<NCCLUniqueIdWrapper>("UniqueId")
        .def(torch::init<>())
        .def("get_raw", &NCCLUniqueIdWrapper::get_raw)
        .def("set_raw", &NCCLUniqueIdWrapper::set_raw);

    m.def("async_tensor_copy_nccl(Tensor src_tensor, int src_rank, int dst_rank, __torch__.nccl_copy_ops.UniqueId id) -> Tensor");
    m.impl("async_tensor_copy_nccl", async_tensor_copy_nccl);
}
