import numpy as np
import psutil
import random
import subprocess as sp
import torch
import uuid
from typing import TypeAlias, List
from enum import Enum
from random import randint
import os
from PIL import Image
import concurrent.futures
import base64
from io import BytesIO


GB = 1 << 30
MB = 1 << 20


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def load_images_parallel(image_paths):
    def load_and_convert(image_path):
        image = Image.open(image_path)
        return image.convert("RGB")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        images_rgb = list(executor.map(load_and_convert, image_paths))
    return images_rgb

def image_to_base64(image):
    image_data = image.tobytes()
    return base64.b64encode(image_data).decode('utf-8')

# def base64_to_image(base64_string):
#     image_bytes = base64.b64decode(base64_string)
#     image_stream = BytesIO(image_bytes)
#     image = Image.open(image_stream)    
#     return image

def base64_to_image(base64_string):
    # Remove the data URI prefix if present
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Create a BytesIO object to handle the image data
    image_stream = BytesIO(image_bytes)
    
    # Open the image using Pillow (PIL)
    try:
        image = Image.open(image_stream)
        image.verify()  # Verify that it is, in fact, an image
        image_stream.seek(0)  # Reset stream position to the beginning
        image = Image.open(image_stream)  # Reopen the image
        return image
    except (IOError, SyntaxError) as e:
        print(f"Error: {e}")
        return None


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_gpu_memory_usage(gpu: int = 0):
    """
    Python equivalent of nvidia-smi, copied from https://stackoverflow.com/a/67722676
    and verified as being equivalent ✅
    """
    output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"

    try:
        memory_use_info = output_to_list(
            sp.check_output(COMMAND.split(), stderr=sp.STDOUT)
        )[1:]

    except sp.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )

    return int(memory_use_info[gpu].split()[0])


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def check_create_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


cudaMemoryIpcHandle: TypeAlias = List[int]


class Stage(Enum):
    """The stage of a SingleStageLLMEngine"""

    ENCODING = "encoding"
    CONTEXT = "context"
    DECODING = "decoding"

    def __str__(self) -> str:
        return self.value
    