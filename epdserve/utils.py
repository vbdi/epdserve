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
import dataclasses
from typing import List, Any
import marshal
import dataclasses
import numpy as np
from typing import List
import json
from epdserve.lifetime import LifetimeEvent, LifetimeEventType, json_decode_lifetime_events
from sortedcontainers import SortedList

GB = 1 << 30
MB = 1 << 20

class MyGeneralQueue(SortedList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def append(self, item):
        super().add(item)
    
    def put_nowait(self, item):
        self.add(item) 

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


class EngineStage(Enum):
    """The stage of a StageEngine"""

    ENCODING = "encoding"
    CONTEXT = "context"
    DECODING = "decoding"

    def __str__(self) -> str:
        return self.value



class EngineStatus(Enum):
    """The status of a StageEngine"""

    ACTIVE = 2 # accepting new requests
    INACTIVE = 1 # not accepting new requests
    SLEEP = 0 # event loop not running

    def __str__(self) -> str:
        return self.value
    


@dataclasses.dataclass
class TestRequest():
    """TestRequest: A request for testing the server's performance"""
    prompt: str
    prompt_len: int
    output_len: int
    images_paths: List[str] = None
    images: List[Any] = None

@dataclasses.dataclass
class Dataset:
    """ Dataset: A dataset for testing the server's performance """
 
    dataset_name: str	# "sharegpt" / "alpaca" / ...
    reqs: List[TestRequest]
    
    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "reqs": [(req.prompt, req.prompt_len, req.output_len) for req in self.reqs]
        }, open(output_path, "wb"))
    
    @staticmethod
    def load(input_path: str):
        loaded_data = marshal.load(open(input_path, "rb"))
        return Dataset(
            loaded_data["dataset_name"],
            [TestRequest(req[0], req[1], req[2]) for req in loaded_data["reqs"]]
        )
        

class RequestResult:
    """ A class for storing the results of a single request """
    
    def __init__(
        self,
        prompt_len: int,
        output_len: int,
        start_time: float,
        end_time: float,
        token_timestamps: List[float],
        lifetime_events: List[LifetimeEvent] = None
    ):
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.start_time = start_time
        self.end_time = end_time
        self.token_timestamps = token_timestamps
        self.lifecycle_events = lifetime_events
        
        self.latency = end_time - start_time # client side latency

        # server side timing
        issued_event = lifetime_events[0]
        assert(issued_event['event_type']=='issued')
        finished_event = lifetime_events[-1]
        assert(finished_event['event_type']=='finished')
        context_end_event = [event for event in lifetime_events if event['event_type']=='context_end'][0]


        self.start_time = issued_event['timestamp']
        self.end_time = finished_event['timestamp']
        self.server_latency = self.end_time - self.start_time
        self.ftl = token_timestamps[0] - self.start_time
        self.ttft = token_timestamps[0] - self.start_time
        # self.ttft = context_end_event['timestamp'] - self.start_time
        self.tpot = 0 if output_len == 1 else (token_timestamps[-1] - token_timestamps[0]) / (output_len-1)

def read_request_results(path: str) -> List[RequestResult]:
    with open(path, "r") as f:
        request_results: List[RequestResult] = [
            RequestResult(
                item["prompt_len"],
                item["output_len"],
                item["start_time"],
                item["end_time"],
                item["token_timestamps"],
                json_decode_lifetime_events(item["lifecycle_events"]) if item.get("lifecycle_events", None) is not None else None
            )
            for item in json.load(f)
        ]
    return request_results

def count_valid_results(request_results: list[RequestResult], ftl: float, tpot: float) -> int:
    """ count_valid_results: Count the number of requests that satisfy the given FTL and TPOT. """
    count = 0
    for req in request_results:
        if req.ftl <= ftl and req.tpot <= tpot:
            count += 1
    return count

def get_slo_attainment(request_results: list[RequestResult], ftl: float, tpot: float) -> float:
    """ get_slo_attainment: Get the SLO attainment of the given request results under the given FTL and TPOT. """
    return count_valid_results(request_results, ftl, tpot) / len(request_results)
