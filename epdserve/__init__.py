import os
import torch
import warnings
warnings.filterwarnings('ignore')

# os.environ["PYTHONBREAKPOINT"]="0"

if not os.environ.get("CUDA_IPC_LIB_PATH"):
    BASE_DIR = os.path.dirname(__file__)
    LIB_PATH = os.path.join(BASE_DIR, "../csrc/build/", "libcuda_ipc_utils.so")
    print(LIB_PATH)
else:
    LIB_PATH = os.environ["CUDA_IPC_LIB_PATH"]

if not os.path.exists(LIB_PATH):
    raise RuntimeError(
        f"Could not find the CUDA_IPC library libcuda_ipc_utils.so at {LIB_PATH}. "
        "Please build the csrc library first."
    )

torch.ops.load_library(LIB_PATH)

from epdserve.llm_offline import OfflineLLM
from epdserve.request import SamplingParams
