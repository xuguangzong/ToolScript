import pynvml

# 显卡相关工具
pynvml.nvmlInit()


def get_gpu_memory_info(gpu=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    if gpu < 0 or gpu >= pynvml.nvmlDeviceGetCount():
        raise f"gpu_id {gpu} 对应的显卡不存在!"
    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    return meminfo


def get_total_gpu_memory(gpu: int = 0) -> float:
    """
    获取显存总量, MB
    :param gpu:
    :return:
    """
    memory_info = get_gpu_memory_info(gpu)
    return round(memory_info.total / 1024 / 1024, 2)


def get_used_gpu_memory(gpu: int = 0) -> float:
    """
    获取已用显存，MB
    :param gpu:
    :return:
    """
    memory_info = get_gpu_memory_info(gpu)
    return round(memory_info.used / 1024 / 1024, 2)


def get_free_gpu_memory(gpu: int = 0) -> float:
    """
    获取剩余显存，MB
    :param gpu:
    :return:
    """
    memory_info = get_gpu_memory_info(gpu)
    return round(memory_info.free / 1024 / 1024, 2)


if __name__ == '__main__':
    print(get_free_gpu_memory())
