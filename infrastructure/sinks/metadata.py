# nireon_v4/infrastructure/sinks/metadata.py
from core.lifecycle import ComponentMetadata

TRACE_SINK_METADATA = ComponentMetadata(
    id="trace_sink_sqlite",
    name="BlockTrace SQLite Sink",
    version="1.0.0",
    description="Subscribes to trace events and persists them to a SQLite database for offline learning.",
    category="infrastructure_service",
    epistemic_tags=["persistence", "logger", "sink", "learning_data"],
    requires_initialize=True,
    dependencies={'EventBusPort': '*'}
)