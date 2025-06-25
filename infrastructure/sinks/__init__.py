# nireon_v4/infrastructure/sinks/__init__.py
from .trace_sink import TraceSink
from .metadata import TRACE_SINK_METADATA

__all__ = ['TraceSink', 'TRACE_SINK_METADATA']