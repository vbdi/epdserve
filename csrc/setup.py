import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
TORCH_LIB_DIR = "/docker/anaconda3/envs/distserve/lib/python3.10/site-packages/torch/lib"

setup(
    name='cuda_ipc_utils',
    ext_modules=[
        CppExtension(
            name='cuda_ipc_utils',
            sources=['block_migration.cpp', 'py_nccl.cc', 'zero_copy.cpp'],
            include_dirs=[
                os.path.join(CUDA_HOME, 'include'),
                os.path.join(TORCH_LIB_DIR, 'include'),
            ],
            library_dirs=[
                os.path.join(CUDA_HOME, 'lib64'),
                TORCH_LIB_DIR,
            ],
            libraries=['c10', 'torch', 'torch_cpu', 'torch_cuda', 'c10_cuda', 'nccl'],
            extra_compile_args=['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
