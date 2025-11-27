from collections import Counter


class AbstractMemoryManager:
    def __init__(self, config, context, next_level=None):
        self.config = config
        self.context = context
        self.next_level = next_level

    def process(self, request):
        raise NotImplementedError

    def post_process(self, timestamp):
        if self.next_level:
            self.next_level.post_process(timestamp)

    def stat(self) -> Counter:
        return Counter()

    def histogram(self) -> Counter:
        return Counter()


__all__ = ["AbstractMemoryManager"]
