from nova_platform.perfetto_protobuf._tgen import TraceGenerator


class AbstractPostProcessor:
    def get_trace_generator(self) -> TraceGenerator:
        raise NotImplementedError()
