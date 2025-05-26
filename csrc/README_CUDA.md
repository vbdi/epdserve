# 🚀 CUDA Utilities for EPDServe

This directory contains custom CUDA/C++ utilities to support high-performance, inter-process (GPU→GPU ) communication for cache and model weight management in the EPD disaggregated inference system. These utilities are implemented in C/C++ and leverage low-level CUDA APIs (e.g., CUDA IPC, NCCL, and asynchronous memory operations) to enable:

- Efficient KV/MM cache block migration across stages
- Fast GPU-to-GPU weight transfers (e.g., during role switching)
- Zero-copy tensor sharing between processes

---

## 📂 Core Components

### 🔧 `block_migration.cpp`

Implements migration logic for **KV and Multimodal (MM) cache blocks** between processes and GPUs.

- This implementation builds upon the `distserve` baseline, which originally adapted from **SwiftTransformers** (ST). However, SwiftTransformers supports only a narrow set of LLMs.
- We significantly restructured the codebase to support the **vLLM** backend, which enables compatibility with a broader class of LLM architectures.
- This required non-trivial modifications due to substantial differences in KV cache memory layout.

#### 📐 Cache Layout Comparison

| Framework         | KV Cache Shape                                             |
|-------------------|------------------------------------------------------------|
| **ST (SwiftTransformers)** | `k/v x [num_blocks, layers_per_worker, heads_per_worker, block_size, head_dim]` |
| Example:          | `2 x [7491, 32, 32, 16, 128]`                               |
| **vLLM**          | `layers_per_worker x [k/v, num_blocks, block_size, heads_per_worker, head_dim]` |
| Example:          | `32 x [2, 7491, 16, 32, 128]`                               |
| **Ours (EPD)**    | `[layers_per_worker, k/v, num_blocks, block_size, heads_per_worker, head_dim]` |
| Example:          | `[32, 2, 7491, 16, 32, 128]`                                |

---

### 🚚 `nccl_copy.py`

Implements **fast GPU-to-GPU weight transfer** using **NCCL and NVLink**.

- Critical for **role switching** (e.g., from Encoding → Prefill/Decoding), where the model type needs to change:
  - Encoding stage only loads **encoder**
  - Prefill and Decoding stages load the **LLM**
- Instead of reading model weights from disk (Disk → CPU → GPU), we **initialize the model on the GPU** and directly **copy the weights asynchronously to another GPU**.
- Assumes that at least one worker of each stage remains live (ensuring a valid source for GPU-to-GPU transfer).

---

### 🪞 `zero_copy.py`

Implements **zero-copy tensor sharing** between processes using CUDA IPC.

- Designed to allow **model weights to be shared across processes** without duplicating memory.
- This is especially useful when multiple GPU workers need access to the same model (e.g., encoder or LLM) and dramatically reduces memory usage and initialization time.

---


### 🧩 `py_nccl.cc`

- Provides a Python extension for **generating unique NCCL communicators** for each process.
- Derived and simplified from `distserve`'s implementation.
- Enables multi-process GPU-to-GPU communication with synchronized collectives for weight transfer and coordination.

---

## 🛠️ Build Instructions

To compile the CUDA IPC library:

```bash
cd csrc
mkdir build && cd build
cmake ..
make
```

This will generate `libcuda_ipc_utils.so`, which must be exported via:

```bash
export CUDA_IPC_LIB_PATH=./csrc/build/libcuda_ipc_utils.so
```

before running the main `epdserve` components.

---

## Tests
### Zero-copy model
``` bash
cd tests
# Start Server
python server.py

# Start client (another terminal)
python client.py
```


---
