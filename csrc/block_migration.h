#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <torch/extension.h>

namespace st::util {

std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor);

bool register_ipc_mem_handle_context(
	std::vector<int64_t> kv_cache_handle_vec,
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &context_parallel_config,
	const std::vector<int64_t> &decoding_parallel_config
);


bool unregister_ipc_mem_handle_context(
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &context_parallel_config,
	const std::vector<int64_t> &decoding_parallel_config
);

void migrate_blocks_context(
	const int64_t context_pp_size,
	const int64_t context_tp_size,
	const int64_t context_dp_rank,

	const std::vector<int64_t> &context_block_indexes,

	const int64_t decoding_pp_size,
	const int64_t decoding_tp_size,

	const int64_t decoding_pp_rank,
	const int64_t decoding_tp_rank,

	const std::vector<int64_t> &decoding_block_indexes,

	torch::Tensor decoding_worker_kv_cache
);

bool register_ipc_mem_handle_encoding(
	std::vector<int64_t> v_cache_handle_vec,
	const std::vector<int64_t> &encoding_parallel_config,
	const std::vector<int64_t> &context_parallel_config
);

void migrate_blocks_encoding(
	const int64_t encoding_pp_size,
	const int64_t encoding_tp_size,
	const int64_t encoding_dp_rank,

	const std::vector<int64_t> &encoding_block_indexes,

	const int64_t context_pp_size,
	const int64_t context_tp_size,

	const int64_t context_pp_rank,
	const int64_t context_tp_rank,

	const std::vector<int64_t> &context_block_indexes,

	torch::Tensor context_worker_v_cache
);

} // namespace st::util
