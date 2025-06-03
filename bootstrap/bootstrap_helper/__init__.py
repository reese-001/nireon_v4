# V4: Nireon Bootstrap Helper Package
from .exceptions import BootstrapError #, StepCommandError # StepCommandError might be for later phases
from .health_reporter import BootstrapHealthReporter # HealthReporter for V4
from .utils import import_by_path, load_yaml_robust, detect_manifest_type
# from .context_builder import build_execution_context # For later phases
from .metadata import DEFAULT_COMPONENT_METADATA_MAP, get_default_metadata, create_service_metadata
from .placeholders import PlaceholderLLMPortImpl, PlaceholderEmbeddingPortImpl, PlaceholderEventBusImpl, PlaceholderIdeaRepositoryImpl
from .service_resolver import find_event_bus_service # find_event_bus_service might be useful earlier

__version__ = '1.0.0' # V4 version for this helper package
__author__ = 'Nireon Bootstrap Team V4'

__all__ = [
    'BootstrapError',
    # 'StepCommandError',
    'BootstrapHealthReporter',
    'import_by_path',
    'load_yaml_robust',
    'detect_manifest_type',
    # 'build_execution_context',
    'DEFAULT_COMPONENT_METADATA_MAP',
    'get_default_metadata',
    'create_service_metadata',
    'PlaceholderLLMPortImpl',
    'PlaceholderEmbeddingPortImpl',
    'PlaceholderEventBusImpl',
    'PlaceholderIdeaRepositoryImpl',
    'find_event_bus_service',
]