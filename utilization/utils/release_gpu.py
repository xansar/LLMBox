from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import torch
import gc
from logging import getLogger
logger = getLogger(__name__)
# import pynvml

# def get_gpu_memory_info():
#     pynvml.nvmlInit()
#     device_count = pynvml.nvmlDeviceGetCount()
#     gpu_memory_info = []

#     for i in range(device_count):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#         memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         gpu_memory_info.append({
#             'total_memory_MB': memory_info.total // 1024 ** 2,
#             'used_memory_MB': memory_info.used // 1024 ** 2,
#             'free_memory_MB': memory_info.free // 1024 ** 2
#         })

#     pynvml.nvmlShutdown()
#     for idx, info in enumerate(gpu_memory_info):
#         logger.debug(f"GPU {idx}: Total Memory: {info['total_memory_MB']} MB, Used Memory: {info['used_memory_MB']} MB, Free Memory: {info['free_memory_MB']} MB")
    
    
import contextlib
import gc

import torch
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.utils import is_cpu

# def cleanup():
#     destroy_model_parallel()
#     destroy_distributed_environment()
#     with contextlib.suppress(AssertionError):
#         torch.distributed.destroy_process_group()
#     gc.collect()
#     if not is_cpu():
#         torch.cuda.empty_cache()

# llm = LLM(...)
# del llm
# cleanup()

# def release_vllm():
#     destroy_model_parallel()
#     destroy_distributed_environment()
#     with contextlib.suppress(AssertionError):
#         torch.distributed.destroy_process_group()
#     gc.collect()
#     if not is_cpu():
#         torch.cuda.empty_cache()

#     logger.info("VLLM released.")
    


def release_vllm(llm):
    # destroy_model_parallel()
    # destroy_distributed_environment()
    # del llm.llm_engine.model_executor
    # del llm
    # gc.collect()
    # torch.cuda.empty_cache()

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm 
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if not is_cpu():
        torch.cuda.empty_cache()

    logger.info("VLLM released.")
    # get_gpu_memory_info()

def release_gpu_memory():
    # FIXME(xansar): 没释放掉,需要手动删除model
    from torch.cuda import empty_cache
    empty_cache()

    from gc import collect
    collect()
    # get_gpu_memory_info()

