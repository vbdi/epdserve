#include "block_migration.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <iostream>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_utils.h"
#include "debug_utils.h"
#include <torch/extension.h>
#include <torch/library.h>

namespace st::util {

/*
The following two functions convert cudaIpcMemHandle_t to/from bytes
We need this because we need to pass cudaIpcMemHandle_t to Python
*/

static std::vector<int64_t> cudaIpcMemHandle2Bytes(const cudaIpcMemHandle_t &handle) {
	std::vector<int64_t> result;
	for (size_t i = 0; i < sizeof(handle); ++i) {
		result.push_back(((uint8_t*) &handle)[i]);
	}
	return result;
}

static cudaIpcMemHandle_t bytes2CudaIpcMemHandle(const std::vector<int64_t> &bytes) {
	assert_whenever(bytes.size() == sizeof(cudaIpcMemHandle_t));
	cudaIpcMemHandle_t result;
	for (size_t i = 0; i < sizeof(result); ++i) {
		((uint8_t*) &result)[i] = bytes[i];
	}
	return result;
}


/*
get_ipc_mem_handle: Get the IPC memory handle of a tensor
The returned handle can be used to open the tensor in another process.
*/
std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor) {
	cudaIpcMemHandle_t handle;
	CUDA_CHECK(cudaIpcGetMemHandle(&handle, tensor.data_ptr()));
	return cudaIpcMemHandle2Bytes(handle);
}


/*
register_ipc_mem_handle: Register an IPC memory handle

This function receives a IPC memory handle and the context worker's info
(context_pp_rank and context_tp_rank) that the handle belongs to, then
it checks whether it needs to register the handle (i.e. whether the k/v range
it needs overlaps with the k/v range that the context worker calculates). If
the answer is YES, register it and note down its local address.

Return true if the handle is registered, false otherwise.
*/
static constexpr int64_t MAX_PARALLEL_HASH = 64*4096;	// Assume 64 dp, 64 pp and 64 tp stages
static void* context_worker_kv_cache_addr[MAX_PARALLEL_HASH];

bool register_ipc_mem_handle_context(
	std::vector<int64_t> kv_cache_handle_vec,
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &context_parallel_config,
	const std::vector<int64_t> &decoding_parallel_config
) {
	// Convert the handles to cudaIpcMemHandle_t
	const cudaIpcMemHandle_t kv_cache_handle = bytes2CudaIpcMemHandle(kv_cache_handle_vec);

	// First we check whether the two k/v cache area overlaps
	const int64_t context_dp_size = context_parallel_config[0];
	const int64_t context_dp_rank = context_parallel_config[1];
	const int64_t context_tp_size = context_parallel_config[2];
	const int64_t context_tp_rank = context_parallel_config[3];
	const int64_t context_pp_size = context_parallel_config[4];
	const int64_t context_pp_rank = context_parallel_config[5];

	const int64_t decoding_dp_size = decoding_parallel_config[0];
	const int64_t decoding_dp_rank = decoding_parallel_config[1];
	const int64_t decoding_tp_size = decoding_parallel_config[2];
	const int64_t decoding_tp_rank = decoding_parallel_config[3];
	const int64_t decoding_pp_size = decoding_parallel_config[4];
	const int64_t decoding_pp_rank = decoding_parallel_config[5];

	const int64_t layers_per_context_worker = num_layers / context_pp_size;
	const int64_t heads_per_context_worker = num_heads / context_tp_size;
	const int64_t layers_per_decoding_worker = num_layers / decoding_pp_size;
	const int64_t heads_per_decoding_worker = num_heads / decoding_tp_size;

	const int64_t context_start_layer = context_pp_rank * layers_per_context_worker;
	const int64_t context_end_layer = context_start_layer + layers_per_context_worker;
	const int64_t context_start_head = context_tp_rank * heads_per_context_worker;
	const int64_t context_end_head = context_start_head + heads_per_context_worker;

	const int64_t decoding_start_layer = decoding_pp_rank * layers_per_decoding_worker;
	const int64_t decoding_end_layer = decoding_start_layer + layers_per_decoding_worker;
	const int64_t decoding_start_head = decoding_tp_rank * heads_per_decoding_worker;
	const int64_t decoding_end_head = decoding_start_head + heads_per_decoding_worker;

	if (context_end_layer <= decoding_start_layer || context_start_layer >= decoding_end_layer ||
		context_end_head <= decoding_start_head || context_start_head >= decoding_end_head) {
		return false;
	} else {
		const int64_t context_worker_hash =  (context_dp_rank<<12) + (context_pp_rank<<6) + context_tp_rank;
		cudaError_t err = cudaIpcOpenMemHandle(&context_worker_kv_cache_addr[context_worker_hash], kv_cache_handle, cudaIpcMemLazyEnablePeerAccess);
		if (err == cudaErrorPeerAccessUnsupported) {
			printf("Error: Peer-to-peer access is unsupported on this platform.\n");
			printf("In the current version of distserve, it is necessary to use a platform that supports GPU P2P access.\n");
			printf("Exiting...");
			exit(1);
		}
		return true;
	}
}


bool unregister_ipc_mem_handle_context(
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &context_parallel_config,
	const std::vector<int64_t> &decoding_parallel_config
) {
	const int64_t context_dp_size = context_parallel_config[0];
	const int64_t context_dp_rank = context_parallel_config[1];
	const int64_t context_tp_size = context_parallel_config[2];
	const int64_t context_tp_rank = context_parallel_config[3];
	const int64_t context_pp_size = context_parallel_config[4];
	const int64_t context_pp_rank = context_parallel_config[5];

	const int64_t decoding_dp_size = decoding_parallel_config[0];
	const int64_t decoding_dp_rank = decoding_parallel_config[1];
	const int64_t decoding_tp_size = decoding_parallel_config[2];
	const int64_t decoding_tp_rank = decoding_parallel_config[3];
	const int64_t decoding_pp_size = decoding_parallel_config[4];
	const int64_t decoding_pp_rank = decoding_parallel_config[5];

	const int64_t layers_per_context_worker = num_layers / context_pp_size;
	const int64_t heads_per_context_worker = num_heads / context_tp_size;
	const int64_t layers_per_decoding_worker = num_layers / decoding_pp_size;
	const int64_t heads_per_decoding_worker = num_heads / decoding_tp_size;

	const int64_t context_start_layer = context_pp_rank * layers_per_context_worker;
	const int64_t context_end_layer = context_start_layer + layers_per_context_worker;
	const int64_t context_start_head = context_tp_rank * heads_per_context_worker;
	const int64_t context_end_head = context_start_head + heads_per_context_worker;

	const int64_t decoding_start_layer = decoding_pp_rank * layers_per_decoding_worker;
	const int64_t decoding_end_layer = decoding_start_layer + layers_per_decoding_worker;
	const int64_t decoding_start_head = decoding_tp_rank * heads_per_decoding_worker;
	const int64_t decoding_end_head = decoding_start_head + heads_per_decoding_worker;

	if (context_end_layer <= decoding_start_layer || context_start_layer >= decoding_end_layer ||
		context_end_head <= decoding_start_head || context_start_head >= decoding_end_head) {
		// No overlap
		return false;
	} else {
		const int64_t context_worker_hash = (context_dp_rank<<12) + (context_pp_rank<<6) + context_tp_rank;
		cudaError_t err = cudaIpcCloseMemHandle(context_worker_kv_cache_addr[context_worker_hash]);
		assert(err==cudaSuccess);
		return true; 
	}
}


/*
migrate_blocks: Migrate blocks from the context stage engine to the decoding stage engine

This function is called by every decoding stage worker when the decoding
stage engine decides to migrate some blocks from the context stage engine
to the decoding stage engine.

In the following code, "pp" stands for "pipeline parallel", and "tp" stands for "tensor parallel".

Here we do not pass a cudaStream to the function. Instead we use the current
stream indicated by at::cuda::getCurrentCUDAStream(). So it is python's
responsibility to set the current stream before calling this function.
*/

void migrate_blocks_context(
	// Parallelism parameters for the context stage engine
	const int64_t context_pp_size,
	const int64_t context_tp_size,
	const int64_t context_dp_rank,

	// Block indexes of the context stage engine
	const std::vector<int64_t> &context_block_indexes,

	// Parallelism parameters for the decoding stage engine
	const int64_t decoding_pp_size,
	const int64_t decoding_tp_size,

	// Rank of the decoding stage worker that calls this function
	const int64_t decoding_pp_rank,
	const int64_t decoding_tp_rank,

	// Block indexes of the decoding stage engine
	const std::vector<int64_t> &decoding_block_indexes,

	// The decoding stage worker's KV cache
	torch::Tensor decoding_worker_kv_cache // VLLM Shape  [layers_per_decoding_worker, k/v, num_blocks, block_size, heads_per_decoding_worker,  head_dim]
) {
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	assert_whenever(decoding_worker_kv_cache.is_contiguous());

	// Calculate some misc stuff
	const int64_t layers_per_decoding_worker = decoding_worker_kv_cache.size(0);
	const int64_t heads_per_decoding_worker = decoding_worker_kv_cache.size(4);
	const int64_t block_size = decoding_worker_kv_cache.size(3);
	const int64_t head_dim = decoding_worker_kv_cache.size(5);
	const int64_t total_num_blocks = decoding_worker_kv_cache.size(2);

	const int64_t num_layers = layers_per_decoding_worker * decoding_pp_size;
	const int64_t num_heads = heads_per_decoding_worker * decoding_tp_size;
	const int64_t heads_per_context_worker = num_heads / context_tp_size;
	const int64_t num_blocks_to_copy = decoding_block_indexes.size();
	const int64_t dtype_size = decoding_worker_kv_cache.dtype().itemsize();

	// The current decoding worker's region of the k/v cache
	const int64_t decoding_start_layer = decoding_pp_rank * layers_per_decoding_worker;
	const int64_t decoding_start_head = decoding_tp_rank * heads_per_decoding_worker;

	int64_t context_pp_rank = 0;
	int64_t context_tp_rank = 0;
	const int64_t context_worker_hash = (context_dp_rank<<12) + (context_pp_rank<<6) + context_tp_rank;
	char* context_worker_base_ptr = (char*) context_worker_kv_cache_addr[context_worker_hash];

	for (int64_t layer_id = 0; layer_id < layers_per_decoding_worker; ++layer_id) {
		for (int64_t block_id = 0; block_id < num_blocks_to_copy; ++block_id) {
			for (int is_value = 0; is_value < 2; ++is_value) {
				const int64_t context_block_index = context_block_indexes[block_id];
				const int64_t decoding_block_index = decoding_block_indexes[block_id];
				CUDA_CHECK(cudaMemcpy2DAsync(
					(char*) decoding_worker_kv_cache.data_ptr()
						+ INDEX_6D(layers_per_decoding_worker, 2, total_num_blocks, block_size, heads_per_decoding_worker, head_dim,
								layer_id, is_value, decoding_block_index, 0, 0, 0) * dtype_size,
					(uint64_t) ((head_dim * heads_per_decoding_worker * block_size) * dtype_size),
					context_worker_base_ptr
						+ INDEX_6D(layers_per_context_worker, 2, total_num_blocks, block_size, heads_per_context_worker, head_dim,
								layer_id, is_value, context_block_index, 0, 0, 0) * dtype_size,
					(uint64_t) ((head_dim * heads_per_context_worker * block_size) * dtype_size),
					(size_t) ((head_dim * heads_per_context_worker * block_size) * dtype_size),
					(size_t) (1),
					cudaMemcpyDeviceToDevice,
					stream
				));
			}
		}
	}
}


/////////////////////////////////////// ENCODING RELATED FUNCTIONS /////////////////////////////////////////

static constexpr int64_t MAX_PARALLEL_HASH_DP = 64*4096; 	// Assume 64 dp, 64 pp and 64 tp stages

static void* vision_cache_addr[MAX_PARALLEL_HASH_DP];
bool register_ipc_mem_handle_encoding(std::vector<int64_t> vision_handle_vec,
   									  const std::vector<int64_t> &encoding_parallel_config,
									  const std::vector<int64_t> &context_parallel_config
) {
	// Convert the handles to cudaIpcMemHandle_t
	const cudaIpcMemHandle_t vision_cache_handle = bytes2CudaIpcMemHandle(vision_handle_vec);

	const int64_t encoding_dp_size = encoding_parallel_config[0];
	const int64_t encoding_dp_rank = encoding_parallel_config[1];
	const int64_t encoding_tp_size = encoding_parallel_config[2];
	const int64_t encoding_tp_rank = encoding_parallel_config[3];
	const int64_t encoding_pp_size = encoding_parallel_config[4];
	const int64_t encoding_pp_rank = encoding_parallel_config[5];

	const int64_t context_dp_size = context_parallel_config[0];
	const int64_t context_dp_rank = context_parallel_config[1];
	const int64_t context_tp_size = context_parallel_config[2];
	const int64_t context_tp_rank = context_parallel_config[3];
	const int64_t context_pp_size = context_parallel_config[4];
	const int64_t context_pp_rank = context_parallel_config[5];

	const int64_t encoding_worker_hash = (encoding_dp_rank<<12) + (encoding_pp_rank<<6) + encoding_tp_rank;
	// std::cout << "DEBUGGING INFO START" << std::endl;
	// std::cout << "encoding_worker_hash " << encoding_worker_hash << std::endl;

	cudaError_t err = cudaIpcOpenMemHandle(&vision_cache_addr[encoding_worker_hash], vision_cache_handle, cudaIpcMemLazyEnablePeerAccess);
	
	if (err == cudaErrorPeerAccessUnsupported) {
		printf("Error: Peer-to-peer access is unsupported on this platform.\n");
		printf("In the current version of distserve, it is necessary to use a platform that supports GPU P2P access.\n");
		printf("Exiting...");
		exit(1);
	}
	return true;
}

void migrate_blocks_encoding(
	const int64_t encoding_pp_size,
	const int64_t encoding_tp_size,

	// DP rank of the encoding stage engine from where this kv cache is located
	const int64_t encoding_dp_rank,

	// Block indexes of the encoding stage engine
	const std::vector<int64_t> &encoding_block_indexes,

	const int64_t context_pp_size,
	const int64_t context_tp_size,

	// Rank of the context stage worker that calls this function
	const int64_t context_pp_rank,
	const int64_t context_tp_rank,

	// Block indexes of the context stage engine
	const std::vector<int64_t> &context_block_indexes,

	// The decoding stage worker's KV cache
	torch::Tensor context_worker_v_cache
) {
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	assert_whenever(context_worker_v_cache.is_contiguous());

	// Calculate some misc stuff
	const int64_t num_blocks_to_copy = context_block_indexes.size();
	const int64_t dtype_size = context_worker_v_cache.dtype().itemsize();

	// context_worker_v_cache   1000, 576, 4096
	const int64_t total_num_blocks = context_worker_v_cache.size(0);
	const int64_t num_token = context_worker_v_cache.size(1);
	const int64_t token_dim = context_worker_v_cache.size(2);


	int64_t encoding_pp_rank = 0;
	int64_t encoding_tp_rank = 0;
	const int64_t encoding_worker_hash = (encoding_dp_rank<<12) + (encoding_pp_rank<<6) + encoding_tp_rank;
	// std::cout << "DEBUGGING INFO START" << std::endl;
	// std::cout << "encoding_worker_hash " << encoding_worker_hash << std::endl;

	char* encoding_worker_base_ptr = (char*) vision_cache_addr[encoding_worker_hash];

		for (int64_t block_id = 0; block_id < num_blocks_to_copy; ++block_id) {
			const int64_t encoding_block_index = encoding_block_indexes[block_id];
			const int64_t context_block_index = context_block_indexes[block_id];
			CUDA_CHECK(cudaMemcpy2DAsync(
				(char*) context_worker_v_cache.data_ptr()       // DST_PTR
					+ INDEX_3D(total_num_blocks, num_token, token_dim,
					           context_block_index, 0, 0) * dtype_size,
				(uint64_t) ((num_token * token_dim) * dtype_size),        // DST_STRIDE

				encoding_worker_base_ptr                        // DST_PTR
					+ INDEX_3D(total_num_blocks,  num_token, token_dim,
					           encoding_block_index, 0, 0) * dtype_size,
				(uint64_t) ((num_token * token_dim) * dtype_size),       // SRC_STRIDE

				(size_t) ((num_token * token_dim) * dtype_size),        // WIDTH
				(size_t) (1),                                // HEIGHT
				cudaMemcpyDeviceToDevice,
				stream
			));
		}
}
}


TORCH_LIBRARY(block_migration_ops, m) {
	m.def("get_ipc_mem_handle", &st::util::get_ipc_mem_handle);
    m.def("register_ipc_mem_handle_context", &st::util::register_ipc_mem_handle_context);
    m.def("unregister_ipc_mem_handle_context", &st::util::unregister_ipc_mem_handle_context);
    m.def("migrate_blocks_context", &st::util::migrate_blocks_context);
    m.def("register_ipc_mem_handle_encoding", &st::util::register_ipc_mem_handle_encoding);
    m.def("migrate_blocks_encoding", &st::util::migrate_blocks_encoding);
  }
