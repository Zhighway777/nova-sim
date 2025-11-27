import logging
logger = logging.getLogger(__name__)


def get_gpu_count():
    try:
        import pycuda.driver as cuda
        # Initialize CUDA
        cuda.init()
        device_count = cuda.Device.count()
        logger.info("found %s GPUs", device_count)
        return device_count
    except:
        return 0
