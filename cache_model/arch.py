class AbstractGCU:
    """Base class emulating the external cache model API."""

    def __init__(self, hardware):
        self.hardware = hardware

    def get_last_memory_manager(self):
        raise NotImplementedError

    def process(self, request, timestamp):
        raise NotImplementedError

    def post_process(self, timestamp):
        pass

    def stat_dict(self):
        return {}

    def histogram_dict(self):
        return {}


__all__ = ["AbstractGCU"]
