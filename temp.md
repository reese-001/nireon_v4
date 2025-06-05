Okay, here are the V4 files for **Phase 1** of the Bootstrap component implementation. These files are adapted from your V3 codebase, with necessary changes to align with the V4 structure and Phase 1 objectives.

**Phase 1 Focus:**
*   Core `bootstrap_nireon_system` function, `BootstrapResult`, `BootstrapValidationData`.
*   Global configuration loading.
*   Initial logging.
*   Key `bootstrap_helper` modules: `exceptions`, `utils`, `metadata`, `placeholders`, `service_resolver`.
*   `FeatureFlagsManager` setup.
*   Core services setup (LLM, Embedding, EventBus, IdeaRepo, IdeaService) using placeholders.

**Important Notes:**
*   **Imports:** All import paths are updated to reflect the `nireon_v4` structure.
*   **ComponentRegistry:** The `ComponentRegistry` class itself is now located in `nireon_v4.core.registry.component_registry`. The `bootstrap` module will use this.
*   **NireonBaseComponent & NireonExecutionContext:** These are adapted from V3 and placed in their respective V4 locations.
*   **Manifest Processing:** For Phase 1, the manifest processing loop (`_process_all_configurations`), component initialization loop, and interface validation loop from V3's `bootstrap.py` are **omitted** or **stubbed out**. They belong to later phases.
*   **Factory Setup:** The call to `_setup_factories_and_validators` is commented out in `bootstrap_nireon_system` for Phase 1, as full factory implementation is for a later phase.
*   **`__init__.py` files:** Empty `__init__.py` files are crucial for Python to recognize directories as packages. These will be created as needed.

---
**File Structure for Phase 1 (Illustrative):**
```
nireon_v4/
├── __init__.py
├── application/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── lifecycle.py
│   ├── context.py
│   ├── ports/
│   │   ├── __init__.py
│   │   ├── embedding_port.py
│   │   ├── event_bus_port.py
│   │   ├── idea_repository_port.py
│   │   └── llm_port.py
│   └── services/
│       ├── __init__.py
│       └── idea_service.py
├── bootstrap/
│   ├── __init__.py
│   ├── bootstrap.py
│   └── bootstrap_helper/
│       ├── __init__.py
│       ├── exceptions.py
│       ├── metadata.py
│       ├── placeholders.py
│       ├── service_resolver.py
│       └── utils.py
├── configs/
│   ├── __init__.py
│   ├── config_utils.py
│   └── loader.py
├── core/
│   ├── __init__.py
│   └── registry/
│       ├── __init__.py
│       └── component_registry.py
├── domain/
│   ├── __init__.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── vector.py
│   └── ideas/
│       ├── __init__.py
│       └── idea.py
└── infrastructure/
    ├── __init__.py
    └── feature_flags.py
```

---
**File Contents:**

**`nireon_v4/__init__.py`**
```python
# Main package marker for nireon_v4
```

**`nireon_v4/application/__init__.py`**
```python
# Nireon V4 Application Layer
```

**`nireon_v4/application/components/__init__.py`**
```python
# Nireon V4 Components
```

**`nireon_v4/application/components/base.py`**
```python
# Adapted from nireon_staging/nireon/application/components/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, TypeVar, Generic
from datetime import datetime, timezone
import hashlib
import logging
# V4: Relative import for NireonExecutionContext
from ..context import NireonExecutionContext
# V4: Relative import for ComponentMetadata, ComponentRegistry, ComponentRegistryMissingError
from .lifecycle import ComponentMetadata, ComponentRegistry, ComponentRegistryMissingError
# V4: Relative import for results
from .results import ProcessResult, AnalysisResult, SystemSignal, AdaptationAction, ComponentHealth

logger = logging.getLogger(__name__)

TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')


class ComponentLifecycle(ABC):
    @abstractmethod
    async def initialize(self, context: NireonExecutionContext) -> None:
        ...

    @abstractmethod
    async def process(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        ...

    @abstractmethod
    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        ...

    @abstractmethod
    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        ...

    @abstractmethod
    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        ...

    @abstractmethod
    async def recover_from_error(self, error: Exception, context: NireonExecutionContext) -> bool:
        ...

    @abstractmethod
    async def shutdown(self, context: NireonExecutionContext) -> None:
        ...

    @abstractmethod
    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        ...


class NireonBaseComponent(ComponentLifecycle, ABC):
    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata):
        if not isinstance(metadata_definition, ComponentMetadata):
            raise TypeError('metadata_definition must be a valid ComponentMetadata object')
        self._config = config or {}
        self._component_id = metadata_definition.id
        self._metadata_definition = metadata_definition
        self._initialized_properly: bool = False
        self._initialization_timestamp: Optional[datetime] = None
        self._last_process_timestamp: Optional[datetime] = None
        self._process_count: int = 0
        self._error_count: int = 0
        logger.debug(f"NireonBaseComponent '{self._component_id}' instantiated")

    @property
    def metadata(self) -> ComponentMetadata:
        return self._metadata_definition

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def component_id(self) -> str:
        return self._component_id

    @property
    def is_initialized(self) -> bool:
        return self._initialized_properly
    
    @property
    def process_count(self) -> int:
        return self._process_count

    @property
    def error_count(self) -> int:
        return self._error_count

    async def initialize(self, context: NireonExecutionContext) -> None:
        if self._initialized_properly:
            context.logger.warning(f"Component '{self.component_id}' already initialized. Re-initializing.")
        
        context.logger.info(f'Initializing component: {self.component_id} (v{self.metadata.version}, category: {self.metadata.category})')
        
        if context.component_registry is None:
            # V4: ComponentRegistry might be optional for some minimal contexts, but vital for initialize
            raise ValueError(f"ComponentRegistry missing in ExecutionContext for '{self.component_id}'")

        try:
            existing = context.component_registry.get(self.component_id)
            if existing is not self:
                context.logger.warning(f"Overwriting existing registration for '{self.component_id}'")
        except ComponentRegistryMissingError: # V4: Using specific error type
            pass # Not registered yet, which is fine.
        
        context.component_registry.register(self, self.metadata)
        await self._self_certify(context) # Assuming _self_certify is part of V4 base or will be added
        
        await self._initialize_impl(context)
        self._initialized_properly = True
        self._initialization_timestamp = datetime.now(timezone.utc)
        context.logger.info(f"Component '{self.component_id}' initialized successfully")

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        # Default implementation, can be overridden by subclasses
        pass

    async def _self_certify(self, context: NireonExecutionContext) -> None:
        # Simplified from V3 for Phase 1, can be expanded later
        config_hash = hashlib.sha256(str(sorted(self.config.items())).encode()).hexdigest()
        certification_data = {
            'component_id': self.component_id,
            'component_name': self.metadata.name,
            'version': self.metadata.version,
            'category': self.metadata.category,
            'status': 'initializing',
            'config_hash': config_hash,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        if hasattr(context.component_registry, 'register_certification'): # V4: Check if method exists
            context.component_registry.register_certification(self.component_id, certification_data)
            context.logger.info(f"Component '{self.component_id}' self-certification completed")
        else:
            context.logger.warning(f"ComponentRegistry for '{self.component_id}' does not support register_certification. Skipping.")


    async def process(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not self._initialized_properly:
            self._error_count += 1
            return ProcessResult(success=False, component_id=self.component_id, message='Component not initialized', error_code='NOT_INITIALIZED')
        
        self._process_count += 1
        self._last_process_timestamp = datetime.now(timezone.utc)
        
        try:
            return await self._process_impl(data, context)
        except Exception as e:
            self._error_count += 1
            context.logger.error(f"Component '{self.component_id}' process error: {e}", exc_info=True)
            if await self.recover_from_error(e, context):
                context.logger.info(f"Component '{self.component_id}' recovered from error")
                return ProcessResult(success=True, component_id=self.component_id, message=f'Recovered from error: {e}', error_code='RECOVERED')
            return ProcessResult(success=False, component_id=self.component_id, message=f'Processing error: {e}', error_code='PROCESS_ERROR')

    @abstractmethod
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        ...

    async def shutdown(self, context: NireonExecutionContext) -> None:
        context.logger.info(f"Shutting down component: '{self.component_id}'")
        try:
            await self._shutdown_impl(context)
        except Exception as e:
            context.logger.error(f"Shutdown error for '{self.component_id}': {e}")
        self._initialized_properly = False
        context.logger.info(f"Component '{self.component_id}' shutdown complete")

    async def _shutdown_impl(self, context: NireonExecutionContext) -> None:
        # Default implementation, can be overridden by subclasses
        pass
        
    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        # Default implementation, can be overridden
        return AnalysisResult(success=True, component_id=self.component_id, metrics={}, confidence=0.5)

    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        # Default implementation, can be overridden
        return []

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        # Default implementation, can be overridden
        return []

    async def recover_from_error(self, error: Exception, context: NireonExecutionContext) -> bool:
        # Default implementation, can be overridden
        context.logger.warning(f"No recovery implemented for '{self.component_id}': {error}")
        return False

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        # Default implementation, can be overridden
        status = "HEALTHY" if self._initialized_properly and self._error_count == 0 else "DEGRADED"
        return ComponentHealth(component_id=self.component_id, status=status, message="Basic health check.")

class TypedNireonComponent(NireonBaseComponent, Generic[TInput, TOutput]):
    @abstractmethod
    async def _process_impl(self, data: TInput, context: NireonExecutionContext) -> ProcessResult:
        ...

    async def process(self, data: TInput, context: NireonExecutionContext) -> ProcessResult: # type: ignore[override]
        return await super().process(data, context)

```

**`nireon_v4/application/components/lifecycle.py`**
```python
# Adapted from nireon_staging/nireon/application/components/lifecycle.py
# V4: ComponentRegistry class itself is moved to core.registry.component_registry
# This file now primarily holds ComponentMetadata and related errors/types.
from __future__ import annotations

import logging
import threading # Keep for potential future use if other classes here need it
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union, Type, Protocol

__all__ = ['ComponentMetadata', 'ComponentRegistryMissingError'] # V4: ComponentRegistry removed from __all__

logger = logging.getLogger(__name__)


@dataclass
class ComponentMetadata:
    id: str
    name: str
    version: str
    category: str # e.g., 'mechanism', 'observer', 'service', 'core_service', 'persistence_service'
    subcategory: Optional[str] = None
    description: str = ""
    capabilities: Set[str] = field(default_factory=set)
    invariants: List[str] = field(default_factory=list)
    accepts: List[str] = field(default_factory=list) # Input DTO types or concepts
    produces: List[str] = field(default_factory=list) # Output DTO types or concepts
    author: str = "Nireon Team"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    epistemic_tags: List[str] = field(default_factory=list)
    # V4: expected_interfaces aligns with V4 Component Developer Guide
    expected_interfaces: Optional[List[Type[Any]]] = field(default_factory=list)
    requires_initialize: bool = True # V4: Added as per V3, useful for bootstrap

    def __post_init__(self) -> None:
        if not all([self.id, self.name, self.version, self.category]):
            raise ValueError("ComponentMetadata fields id, name, version, category must be non-empty")
        
        if self.created_at.tzinfo is None: # Ensure timezone-aware datetime
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

        if not isinstance(self.epistemic_tags, list):
            raise TypeError("epistemic_tags must be a list of strings")
        for tag in self.epistemic_tags:
            if not isinstance(tag, str):
                raise TypeError(f"All epistemic_tags must be strings, got {type(tag)}")
        
        if self.expected_interfaces is not None and not isinstance(self.expected_interfaces, list):
            raise TypeError("expected_interfaces must be a list of Protocol types or None")
        if self.expected_interfaces:
            for iface in self.expected_interfaces:
                # Basic check, Protocol type checking is complex
                if not isinstance(iface, type) or not hasattr(iface, '__mro__'):
                    logger.warning(f"Item '{iface}' in expected_interfaces for '{self.id}' might not be a valid Protocol type.")
        
        if not isinstance(self.requires_initialize, bool):
            raise TypeError(f"'requires_initialize' for '{self.id}' must be a boolean.")


class ComponentRegistryMissingError(KeyError):
    """Custom exception for when a component is not found in the registry."""
    def __init__(self, component_id: str, message: Optional[str] = None):
        default_message = f"Component '{component_id}' not found in registry."
        final_message = message if message is not None else default_message
        super().__init__(final_message)
        self.component_id = component_id

# V4: The ComponentRegistry class is moved to nireon_v4.core.registry.component_registry
# The content of the V3 ComponentRegistry class will be adapted there.
```

**`nireon_v4/application/context.py`**
```python
# Adapted from nireon_staging/nireon/application/context.py
from __future__ import annotations
from copy import deepcopy # V4: Ensure deepcopy for context cloning
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# V4: Import ComponentRegistry from its new location
from nireon_v4.core.registry.component_registry import ComponentRegistry
from nireon_v4.application.ports.event_bus_port import EventBusPort

# V4: Placeholder for LoggerAdapter and ConfigProvider as per V3 context.py
# These can be properly typed or imported later if needed.
LoggerAdapter = Any
ConfigProvider = Any
StateManager = Any # V4: Adding StateManager as per V3 context.py


class NireonExecutionContext:
    def __init__(
        self,
        *,
        run_id: str,
        step: int = 0,
        feature_flags: Optional[Dict[str, Any]] = None,
        component_registry: Optional[ComponentRegistry] = None, # V4: Type hint to new ComponentRegistry
        event_bus: Optional[EventBusPort] = None,
        config: Optional[Dict[str, Any]] = None, # General config dict
        session_id: Optional[str] = None,
        component_id: Optional[str] = None, # ID of the component this context is for
        timestamp: Optional[datetime] = None,
        replay_mode: bool = False,
        replay_seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        logger: Optional[LoggerAdapter] = None,
        config_provider: Optional[ConfigProvider] = None,
        state_manager: Optional[StateManager] = None,
    ) -> None:
        self.run_id: str = run_id
        self.step: int = step
        self.feature_flags: Dict[str, Any] = feature_flags or {}
        self.component_registry = component_registry
        self.event_bus = event_bus
        self.logger = logger 
        self.config_provider = config_provider
        self.state_manager = state_manager
        self.config = config or {} 
        self.session_id = session_id
        self.component_id = component_id
        self.timestamp: datetime = timestamp or datetime.now(timezone.utc)
        self.replay_mode = replay_mode
        self.replay_seed = replay_seed
        self.metadata: Dict[str, Any] = metadata or {}
        self._custom_data: Dict[str, Any] = {} # For arbitrary data passing

    def is_flag_enabled(self, flag_name: str, default: bool = False) -> bool:
        return bool(self.feature_flags.get(flag_name, default))

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        return self._custom_data.get(key, default)

    def set_custom_data(self, key: str, value: Any) -> 'NireonExecutionContext':
        # V4: Return a new instance to maintain immutability where desired
        new_custom = deepcopy(self._custom_data)
        new_custom[key] = value
        return self._clone(_internal_custom_data_override=new_custom)


    def with_component_scope(self, component_id: str) -> "NireonExecutionContext":
        return self._clone(component_id=component_id)

    def with_step(self, step: int) -> "NireonExecutionContext":
        return self._clone(step=step)

    def with_metadata(self, **updates) -> "NireonExecutionContext":
        new_meta = {**self.metadata, **updates}
        return self._clone(metadata=new_meta)

    def with_flags(self, **flag_updates) -> "NireonExecutionContext":
        new_flags = {**self.feature_flags, **flag_updates}
        return self._clone(feature_flags=new_flags)
    
    def advance_step(self, new_step: int) -> "NireonExecutionContext": # Added from V3
        return self._clone(step=new_step)

    def _clone(self, **overrides) -> "NireonExecutionContext":
        # V4: Use deepcopy for mutable fields to ensure true clones
        params = {
            "run_id": self.run_id,
            "step": self.step,
            "feature_flags": deepcopy(self.feature_flags),
            "component_registry": self.component_registry, # Typically shared, not deepcopied
            "event_bus": self.event_bus, # Typically shared
            "config": deepcopy(self.config),
            "session_id": self.session_id,
            "component_id": self.component_id,
            "timestamp": self.timestamp, # Timestamp of original context, or make new? For now, keep.
            "replay_mode": self.replay_mode,
            "replay_seed": self.replay_seed,
            "metadata": deepcopy(self.metadata),
            "logger": self.logger, # Typically shared or re-scoped
            "config_provider": self.config_provider, # Typically shared
            "state_manager": self.state_manager, # Typically shared
        }
        # Handle _custom_data separately for precise control
        internal_custom_data_override = overrides.pop("_internal_custom_data_override", None)
        
        params.update(overrides)
        
        new_instance = NireonExecutionContext(**params)
        
        if internal_custom_data_override is not None:
            new_instance._custom_data = internal_custom_data_override
        else:
            new_instance._custom_data = deepcopy(self._custom_data)
            
        return new_instance

    def __repr__(self) -> str:
        return (
            f"NireonExecutionContext(run_id={self.run_id!r}, step={self.step}, "
            f"component_id={self.component_id!r}, session_id={self.session_id!r}, "
            f"feature_flags={self.feature_flags})"
        )

# V4: Alias for potential external use if some V3 code expects ExecutionContext
ExecutionContext = NireonExecutionContext
```

**`nireon_v4/application/ports/__init__.py`** (empty)
**`nireon_v4/application/ports/embedding_port.py`** (Copy from V3)
**`nireon_v4/application/ports/event_bus_port.py`** (Copy from V3)
**`nireon_v4/application/ports/idea_repository_port.py`** (Copy from V3)
**`nireon_v4/application/ports/llm_port.py`** (Copy from V3)

**`nireon_v4/application/services/__init__.py`** (empty)

**`nireon_v4/application/services/idea_service.py`**
```python
# Adapted from nireon_staging/nireon/application/services/idea_service.py
from __future__ import annotations
import logging
from datetime import datetime, timezone # V4: Ensure timezone is used
from typing import List, Optional

# V4: Use V4 imports
from nireon_v4.application.context import NireonExecutionContext
from nireon_v4.application.ports.event_bus_port import EventBusPort
from nireon_v4.application.ports.idea_repository_port import IdeaRepositoryPort
from nireon_v4.domain.ideas.idea import Idea

logger = logging.getLogger(__name__)

class IdeaService:
    def __init__(self, repository: IdeaRepositoryPort, event_bus: EventBusPort | None = None):
        self.repository = repository
        self.event_bus = event_bus
        logger.info("IdeaService initialized")

    def create_idea(
        self,
        *,
        text: str,
        parent_id: str | None = None,
        context: NireonExecutionContext | None = None,
    ) -> Idea:
        idea = Idea.create(text, parent_id)
        
        # V4: Add hash if compute_hash exists, as in V3
        if hasattr(idea, 'compute_hash'):
            try:
                idea.metadata['hash'] = idea.compute_hash()
            except Exception:
                logger.debug("compute_hash() failed for new idea", exc_info=True)

        self.repository.save(idea)

        if parent_id and hasattr(self.repository, 'add_child_relationship'):
            try: # V4: Add try-except for robustness, as in V3
                self.repository.add_child_relationship(parent_id, idea.idea_id)
            except Exception:
                logger.debug("Repository lacks add_child_relationship() or it failed", exc_info=True)
        
        if self.event_bus:
            self.event_bus.publish(
                "idea_created",
                {
                    "idea_id": idea.idea_id,
                    "text": idea.text[:50] + "..." if len(idea.text) > 50 else idea.text,
                    "parent_ids": idea.parent_ids,
                    "run_id": context.run_id if context else None,
                },
            )
        logger.info(f"Created Idea {idea.idea_id}")
        return idea

    def save_idea(self, idea: Idea, context: NireonExecutionContext | None = None) -> None:
        if not isinstance(idea, Idea):
            logger.error(f"Attempted to save non-Idea object: {type(idea)}")
            raise ValueError("Can only save Idea objects.")
        self.repository.save(idea)
        logger.info(f"Saved existing Idea {idea.idea_id} via save_idea method.")
        if self.event_bus:
            self.event_bus.publish('idea_persisted', {
                'idea_id': idea.idea_id,
                'text_snippet': idea.text[:50] + "..." if len(idea.text) > 50 else idea.text,
                'method': idea.method,
                'run_id': context.run_id if context else None
            })


    def get_idea(self, idea_id: str) -> Idea | None:
        return self.repository.get_by_id(idea_id)

    def get_all_ideas(self) -> List[Idea]:
        return self.repository.get_all()

    def add_world_fact(
        self, *, idea_id: str, fact_id: str, context: NireonExecutionContext | None = None
    ) -> bool:
        if not hasattr(self.repository, 'add_world_fact'):
            logger.debug("Repository lacks add_world_fact(); skipping")
            return False
        
        success: bool = self.repository.add_world_fact(idea_id, fact_id)
        if success and self.event_bus:
            self.event_bus.publish(
                "world_fact_added",
                {"idea_id": idea_id, "fact_id": fact_id, "run_id": context.run_id if context else None},
            )
        return success

    def get_children(self, idea_id: str) -> List[Idea]:
        if hasattr(self.repository, 'get_by_parent_id'):
            return self.repository.get_by_parent_id(idea_id)
        # Fallback if repository doesn't have specific method
        return [i for i in self.repository.get_all() if idea_id in i.parent_ids]

```

**`nireon_v4/bootstrap/__init__.py`**
```python
# V4: Nireon Bootstrap System
from .bootstrap import bootstrap_nireon_system, bootstrap, BootstrapResult, CURRENT_SCHEMA_VERSION, BootstrapValidationData

__all__ = [
    "bootstrap_nireon_system",
    "bootstrap",
    "BootstrapResult",
    "BootstrapValidationData",
    "CURRENT_SCHEMA_VERSION",
]
```

**`nireon_v4/bootstrap/bootstrap.py`**
```python
from __future__ import annotations
import asyncio
import logging
import random # Keep for now, though factories are out of scope for P1
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Set

# V4 imports
from nireon_v4.application.components.base import NireonBaseComponent
from nireon_v4.application.components.lifecycle import ComponentMetadata, ComponentRegistryMissingError
# V4: ComponentRegistry is now in core.registry
from nireon_v4.core.registry.component_registry import ComponentRegistry
from nireon_v4.application.context import NireonExecutionContext
# from nireon_v4.application.factories.dependencies import CommonMechanismDependencies # P2
# from nireon_v4.application.factories.mechanism_factory import SimpleMechanismFactory # P2
from nireon_v4.application.services.idea_service import IdeaService
# from nireon_v4.application.services.llm_router import LLMRouter # P2
# from nireon_v4.application.validation.interface_validator import InterfaceValidator # P2
from nireon_v4.application.ports.embedding_port import EmbeddingPort
from nireon_v4.application.ports.event_bus_port import EventBusPort
from nireon_v4.application.ports.idea_repository_port import IdeaRepositoryPort
from nireon_v4.application.ports.llm_port import LLMPort

from .bootstrap_helper.component_processor import process_simple_component, register_orchestration_command, instantiate_shared_service # Parts for P3/P4
# from .bootstrap_helper.context_builder import build_execution_context # P2
# from .bootstrap_helper.enhanced_components import init_full_component # P4
from .bootstrap_helper.health_reporter import BootstrapHealthReporter, ComponentStatus
from .bootstrap_helper.placeholders import PlaceholderEmbeddingPortImpl, PlaceholderEventBusImpl, PlaceholderIdeaRepositoryImpl, PlaceholderLLMPortImpl
from .bootstrap_helper.service_resolver import find_event_bus_service, get_or_create_idea_service, get_or_create_service, _safe_register_service_instance
from .bootstrap_helper.utils import detect_manifest_type, load_yaml_robust
from .bootstrap_helper.metadata import create_service_metadata
from nireon_v4.configs.loader import load_config
from nireon_v4.infrastructure.feature_flags import FeatureFlagsManager
from .bootstrap_helper.exceptions import BootstrapError

# V4: Setup logging similar to V3
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nireon_bootstrap_v4.log') # Log to a different file for V4
    ]
)
logger = logging.getLogger(__name__)

# V4: Version for validation data schema, if used.
CURRENT_SCHEMA_VERSION = '1.0.4' # Matching V3 for now

@dataclass
class BootstrapResult:
    registry: ComponentRegistry
    health_reporter: BootstrapHealthReporter
    validation_data: 'BootstrapValidationData' # Forward reference

    def __iter__(self): # Keep for V3 compatibility if needed
        return iter((self.registry, self.health_reporter, self.validation_data))

    def __getitem__(self, index: int): # Keep for V3 compatibility
        warnings.warn(
            "Positional indexing of BootstrapResult is deprecated and will be removed in v3.0. "
            "Use named attributes (e.g., result.registry) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return (self.registry, self.health_reporter, self.validation_data)[index]
    
    def __len__(self) -> int: # Keep for V3 compatibility
        return 3

    @property
    def success(self) -> bool:
        # V4: Simplified success check for Phase 1 (can be expanded later)
        # For now, assume success if no critical failures and registry exists.
        # V3 used health_reporter.has_critical_failures() == 0
        # For P1, let's assume basic success if we get here.
        return self.health_reporter.has_critical_failures() == 0 if hasattr(self.health_reporter, 'has_critical_failures') else True


class BootstrapValidationData: # Adapted from V3
    def __init__(self, global_config: Optional[Dict[str, Any]] = None):
        self.global_config = global_config or {}
        self._lock = threading.RLock() # Keep RLock for thread safety
        self.original_metadata_by_id: Dict[str, ComponentMetadata] = {}
        self.resolved_configs_by_id: Dict[str, Dict[str, Any]] = {}
        self.manifest_specs_by_id: Dict[str, Dict[str, Any]] = {}
        self.pruning_metrics: Dict[str, Dict[str, int]] = {}
        # V4: Methods like store_component_data, get_validation_data will be used in later phases

    def store_component_data(self, component_id: str, original_metadata: ComponentMetadata,
                             resolved_config: Dict[str, Any], manifest_spec: Dict[str, Any]):
        # V4 Phase 1: This method won't be called yet as component processing is later.
        logger.debug(f"V4-P1 STUB: BootstrapValidationData.store_component_data for {component_id}")
        pass

    def get_validation_data(self, component_id: str) -> Tuple[Optional[ComponentMetadata], Optional[Dict[str, Any]]]:
        # V4 Phase 1: This method won't be called yet.
        logger.debug(f"V4-P1 STUB: BootstrapValidationData.get_validation_data for {component_id}")
        return (None, None)

    def get_component_data(self, component_id: str) -> Optional[Tuple[ComponentMetadata, Dict, Dict]]:
        # V4 Phase 1: This method won't be called yet.
        logger.debug(f"V4-P1 STUB: BootstrapValidationData.get_component_data for {component_id}")
        return None


async def bootstrap_nireon_system(
    config_paths: List[str | Path],
    *,
    existing_registry: Optional[ComponentRegistry] = None,
    existing_event_bus: Optional[EventBusPort] = None,
    manifest_style: str = 'auto', # V4: Keep for compatibility, though enhanced is preferred
    replay: bool = False,
    env: Optional[str] = None,
    global_app_config: Optional[Dict[str, Any]] = None
) -> BootstrapResult:
    
    run_id = f"bootstrap_run_v4_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    logger.info("=== NIREON V4 System Bootstrap Starting (Single Bootstrap Authority Pattern) ===")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Config Paths: {config_paths}")
    logger.info(f"Manifest Style: {manifest_style}") # V4: Style detection may be simplified
    logger.info(f"Replay Mode: {replay}")

    if global_app_config is None:
        logger.warning("global_app_config not provided to bootstrap_nireon_system. Loading it now.")
        env_for_config = env if env is not None else "default"
        try:
            # V4: Use V4 config loader
            global_app_config_loaded: Dict[str, Any] = load_config(env=env_for_config)
            logger.debug(f"✓ Loaded global_application_config for env '{env_for_config}'")
        except Exception as exc:
            logger.critical(
                f"CRITICAL: Failed to load global_application_config for env '{env_for_config}'. Error: {exc}",
                exc_info=True
            )
            # V4: Use V4 BootstrapError
            raise BootstrapError(f"Failed to load essential global_application_config for env '{env_for_config}'. Cannot continue.") from exc
        final_global_app_config = global_app_config_loaded
    else:
        final_global_app_config = global_app_config

    is_strict_mode = final_global_app_config.get('bootstrap_strict_mode', True) # V3: Defaulted to True
    logger.info(f"Bootstrap Strict Mode: {is_strict_mode}")

    validation_data_store = BootstrapValidationData(global_config=final_global_app_config)
    
    # V4: Use ComponentRegistry from core.registry
    registry = existing_registry or ComponentRegistry()
    health_reporter = BootstrapHealthReporter(registry) # V4: health_reporter uses V4 registry

    logger.info("Phase 0: Setting up FeatureFlagsManager...")
    try:
        ff_manager = None
        try:
            ff_manager = registry.get_service_instance(FeatureFlagsManager)
            logger.info("✓ FeatureFlagsManager already exists in registry.")
        except (ComponentRegistryMissingError, AttributeError): # V4: Use V4 error type
            ff_manager = FeatureFlagsManager(final_global_app_config.get("feature_flags", {}))
            _safe_register_service_instance(registry, FeatureFlagsManager, ff_manager, "FeatureFlagsManager", "core_service")
            logger.info("✓ FeatureFlagsManager created and registered in registry.")
    except Exception as e_ff:
        logger.critical(f"CRITICAL: Failed to setup FeatureFlagsManager: {e_ff}", exc_info=True)
        if is_strict_mode:
            raise BootstrapError("FeatureFlagsManager setup failed in strict mode.") from e_ff
        logger.warning("FeatureFlagsManager setup failed but continuing in non-strict mode.")

    logger.info("Phase 1: Setting up core services...")
    try:
        await _setup_core_services(registry, existing_event_bus, final_global_app_config)
        logger.info("✓ Core services configured")
    except Exception as e_core_svc:
        logger.critical(f"CRITICAL: Failed to setup core services: {e_core_svc}", exc_info=True)
        if is_strict_mode:
            raise BootstrapError("Core services setup failed in strict mode.") from e_core_svc
        logger.warning("Core services setup failed but continuing in non-strict mode.")

    # V4 Phase 1: Defer factory setup and manifest processing to later phases
    # common_mechanism_deps, mechanism_factory, interface_validator = _setup_factories_and_validators(registry, run_id)
    # logger.info("✓ Factories and InterfaceValidator configured (STUBBED FOR V4-P1)")
    logger.info("V4-P1: Factory setup and manifest processing deferred to later phases.")


    # V4 Phase 1: No manifest processing loop
    # all_configs = _load_configuration_files(config_paths)
    # ...
    # total_processed_for_instantiation = await _process_all_configurations(...)

    # V4 Phase 1: No component initialization loop
    # ...

    # V4 Phase 1: No interface validation loop
    # ...

    # V4 Phase 1: Minimal finalization
    await _finalize_bootstrap(registry, health_reporter, run_id, 0, 0, 0, 0, 0, manifest_style, replay, is_strict_mode)
    
    return BootstrapResult(registry=registry, health_reporter=health_reporter, validation_data=validation_data_store)


async def _setup_core_services(
    registry: ComponentRegistry, 
    existing_event_bus: Optional[EventBusPort], 
    global_app_config: Dict[str, Any]
):
    # V4: Using _safe_register_service_instance from V4 bootstrap_helper
    _safe_register_service_instance(registry, ComponentRegistry, registry, "component_registry", "core_service")
    logger.debug("✓ ComponentRegistry instance ensured in itself.")

    # V4: Using get_or_create_service from V4 bootstrap_helper
    llm_service = get_or_create_service(registry, LLMPort, PlaceholderLLMPortImpl, "LLMPort")
    embedding_service = get_or_create_service(registry, EmbeddingPort, PlaceholderEmbeddingPortImpl, "EmbeddingPort")

    if existing_event_bus is not None:
        event_bus_service = existing_event_bus
        try:
            registry.get_service_instance(EventBusPort) # Check if already registered by type
            logger.debug("✓ EventBus already registered in registry.")
        except (ComponentRegistryMissingError, AttributeError): # V4: Use V4 error
            _safe_register_service_instance(registry, EventBusPort, event_bus_service, "provided_event_bus", "event_bus_service")
            logger.debug("✓ Provided EventBus instance registered in registry.")
    else:
        event_bus_service = get_or_create_service(registry, EventBusPort, PlaceholderEventBusImpl, "EventBusPort")

    idea_repo_instance = get_or_create_service(registry, IdeaRepositoryPort, PlaceholderIdeaRepositoryImpl, "IdeaRepositoryPort", category="repository_service")
    
    get_or_create_idea_service(registry, idea_repo_instance, event_bus_service)


async def _finalize_bootstrap(
    registry: ComponentRegistry, 
    health_reporter: BootstrapHealthReporter, 
    run_id: str,
    total_instantiated_registered: int, # V4-P1: will be 0
    total_initialized: int, # V4-P1: will be 0
    initialization_failures: int, # V4-P1: will be 0
    validation_success_count: int, # V4-P1: will be 0
    validation_failure_count: int, # V4-P1: will be 0
    manifest_style: str, 
    replay: bool,
    is_strict_mode: bool
):
    logger.info("Phase 4: Finalising bootstrap (V4-P1 minimal)...")
    summary = health_reporter.generate_summary() # V4: HealthReporter adapted
    # V4-P1: Logging can be simplified as most counts will be zero
    if initialization_failures > 0 or validation_failure_count > 0:
        if is_strict_mode:
            logger.critical(">>> BOOTSTRAP WOULD HAVE HALTED DUE TO ERRORS (strict_mode was on) <<<")
        else:
            logger.warning(">>> BOOTSTRAP COMPLETED PARTIALLY (strict_mode was off) <<<")
    logger.info(summary)

    try:
        event_bus_service = registry.get_service_instance(EventBusPort)
    except Exception:
        event_bus_service = None

    if event_bus_service is not None:
        try:
            event_bus_service.publish(
                "nireon_system_bootstrapped_v4", # V4: Different event type
                {
                    "run_id": run_id,
                    "component_count": len(registry.list_components()),
                    "instantiated_registered_count": total_instantiated_registered,
                    "initialized_count": total_initialized,
                    "initialization_failures": initialization_failures,
                    "validation_success_count": validation_success_count,
                    "validation_failure_count": validation_failure_count,
                    "manifest_style": manifest_style,
                    "replay_mode": replay,
                    "strict_mode": is_strict_mode,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "bootstrap_phase": "Phase 1 Complete" # V4: Indicate phase
                },
            )
        except Exception as exc:
            logger.warning(f"Failed to publish bootstrap completion event: {exc}")
    else:
        logger.warning("No event bus available – cannot publish bootstrap completion event")
    
    logger.info("=== NIREON V4 System Bootstrap Phase 1 Completed (Single Bootstrap Authority) ===")
    logger.info(f"Total components in registry: {len(registry.list_components())}")
    # V4-P1: Other counts will be 0
    logger.info(f"Bootstrap strict mode was: {('ON' if is_strict_mode else 'OFF')}")


async def bootstrap( # Wrapper from V3, adapted for V4
    manifest_files: List[str | Path],
    *,
    replay: bool = False,
    registry_ext: Optional[ComponentRegistry] = None,
    bus_ext: Optional[EventBusPort] = None,
    env: Optional[str] = None,
) -> ComponentRegistry: # V4: Returns V4 ComponentRegistry
    logger.info("=== V4 Standalone Bootstrap (Multi-Manifest) - Consider using build_container instead ===")
    logger.info(f"Manifests: {manifest_files}")
    logger.info(f"Replay: {replay}")

    env_for_config = env if env is not None else "default"
    try:
        # V4: Use V4 config loader
        global_application_config: Dict[str, Any] = load_config(env=env_for_config)
        logger.debug(f"✓ Loaded global_application_config for env '{env_for_config}' in bootstrap()")
    except Exception as exc:
        logger.warning(f"Failed to load global_application_config in bootstrap() – proceeding with empty config. Error: {exc}")
        global_application_config = {}

    result = await bootstrap_nireon_system(
        config_paths=manifest_files,
        existing_registry=registry_ext,
        existing_event_bus=bus_ext,
        manifest_style='auto', # V4: Default behavior might change later
        replay=replay,
        env=env,
        global_app_config=global_application_config,
    )
    return result.registry


async def _test_bootstrap_results(registry: ComponentRegistry): # V4: Test with V4 registry
    logger.info("Testing service retrieval (V4-P1)...")
    try:
        llm_service = registry.get_service_instance(LLMPort)
        embedding_service = registry.get_service_instance(EmbeddingPort)
        event_bus = registry.get_service_instance(EventBusPort)
        idea_service = registry.get_service_instance(IdeaService)
        logger.info(
            f"✓ Services retrieved via get_service_instance – LLM {type(llm_service)} | Embed {type(embedding_service)} | EventBus {type(event_bus)} | IdeaService {type(idea_service)}"
        )
    except Exception as exc:
        logger.error(f"Service retrieval failed: {exc}")
    
    logger.info(f"Available components: {registry.list_components()}")


async def main(): # V4: Smoke test main
    logger.info("Starting NIREON V4 Bootstrap System Phase 1 Smoke‑Test")
    # V4 Phase 1: No manifest processing, so dummy_yaml is not used yet.
    # Test will focus on core service setup.
    
    try:
        # V4: Use V4 config loader
        global_app_config_for_smoke_test = load_config(env='default')
    except Exception as e:
        logger.error(f"Smoke test: Failed to load global_app_config: {e}")
        global_app_config_for_smoke_test = {}

    try:
        # V4 Phase 1: Call with empty config_paths as manifest processing is later
        result = await bootstrap_nireon_system(config_paths=[], global_app_config=global_app_config_for_smoke_test)
        await _test_bootstrap_results(result.registry)
    finally:
        logger.info("V4 Smoke‑Test Phase 1 completed.")


if __name__ == '__main__':
    asyncio.run(main())

__all__ = ['BootstrapResult', 'BootstrapValidationData', 'bootstrap_nireon_system', 'bootstrap', 'CURRENT_SCHEMA_VERSION']

```

**`nireon_v4/bootstrap/bootstrap_helper/__init__.py`**
```python
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
```

**`nireon_v4/bootstrap/bootstrap_helper/exceptions.py`** (Copy from V3 `nireon_staging/nireon/application/bootstrap_helper/exceptions.py`, remove `StepCommandError` for Phase 1 if not needed)
```python
# Adapted from nireon_staging/nireon/application/bootstrap_helper/exceptions.py
class BootstrapError(RuntimeError):
    """Base exception for bootstrap process errors."""
    pass

# V4 Phase 1: StepCommandError is likely not needed yet as command processing is later.
# class StepCommandError(BootstrapError):
#     """Exception raised for errors specific to StepCommand execution during bootstrap."""
#     def __init__(self, command_id: str, msg: str, *, original: Exception | None = None) -> None:
#         super().__init__(f"[{command_id}] {msg}")
#         self.command_id = command_id
#         self.message = msg
#         self.original = original
```

**`nireon_v4/bootstrap/bootstrap_helper/metadata.py`**
```python
# Adapted from nireon_staging/nireon/application/bootstrap_helper/metadata.py
from typing import Dict, Optional
# V4: Use V4 ComponentMetadata
from nireon_v4.application.components.lifecycle import ComponentMetadata
# V4: ADVERSARIAL_CRITIC_METADATA path will change if/when that mechanism is ported
# For Phase 1, this map might be simpler or only contain stubs if mechanisms aren't used.
# from nireon.application.mechanisms.adversarial_critic import ADVERSARIAL_CRITIC_METADATA

# V4: Define default metadata for core components or placeholders if needed in Phase 1.
# This map might be less relevant for Phase 1 if no simple manifests are processed.
# For now, let's keep a similar structure.
EXPLORER_METADATA_DEFAULT = ComponentMetadata(
    id='explorer_mechanism_default_v4', name='ExplorerMechanismV4', version='1.0.0',
    category='mechanism', epistemic_tags=['mutator', 'innovator']
)
CATALYST_METADATA_DEFAULT = ComponentMetadata(
    id='catalyst_mechanism_default_v4', name='CatalystMechanismV4', version='1.0.0',
    category='mechanism', epistemic_tags=['synthesizer', 'cross_pollinator']
)
SENTINEL_METADATA_DEFAULT = ComponentMetadata(
    id='sentinel_mechanism_default_v4', name='SentinelMechanismV4', version='1.0.0',
    category='mechanism', epistemic_tags=['evaluator', 'gatekeeper']
)
# ADVERSARIAL_CRITIC_METADATA_DEFAULT = ADVERSARIAL_CRITIC_METADATA # Placeholder

DEFAULT_COMPONENT_METADATA_MAP: Dict[str, ComponentMetadata] = {
    'explorer_mechanism_v4': EXPLORER_METADATA_DEFAULT,
    'catalyst_mechanism_v4': CATALYST_METADATA_DEFAULT,
    'sentinel_mechanism_v4': SENTINEL_METADATA_DEFAULT,
    # 'adversarial_critic_mechanism_v4': ADVERSARIAL_CRITIC_METADATA_DEFAULT,
}


def get_default_metadata(factory_key: str) -> ComponentMetadata | None:
    return DEFAULT_COMPONENT_METADATA_MAP.get(factory_key)


def create_service_metadata(
    service_id: str,
    service_name: str,
    category: str = "service",
    description: Optional[str] = None,
    requires_initialize: bool = False # V4: Default to False for simple services
) -> ComponentMetadata:
    return ComponentMetadata(
        id=service_id,
        name=service_name,
        version="1.0.0", # V4 default version
        category=category,
        description=description or f"Bootstrap-created V4 {service_name}",
        epistemic_tags=[], # Services usually don't have epistemic tags
        requires_initialize=requires_initialize,
    )

```

**`nireon_v4/bootstrap/bootstrap_helper/placeholders.py`**
```python
# Adapted from nireon_staging/nireon/application/bootstrap_helper/placeholders.py
import logging
import random
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

# V4 imports
from nireon_v4.application.ports.llm_port import LLMPort
from nireon_v4.application.ports.embedding_port import EmbeddingPort
from nireon_v4.application.ports.event_bus_port import EventBusPort
from nireon_v4.application.ports.idea_repository_port import IdeaRepositoryPort
from nireon_v4.domain.embeddings.vector import Vector, DEFAULT_DTYPE # Ensure V4 Vector is used
from nireon_v4.domain.ideas.idea import Idea # Ensure V4 Idea is used

logger = logging.getLogger(__name__)

class PlaceholderLLMPortImpl(LLMPort):
    async def call_llm_async(self, prompt: str, **kwargs) -> str:
        logger.debug(f"V4-PlaceholderLLMPort: Async call_llm with prompt: {prompt[:50]}...")
        return f"V4 Async LLM response to: {prompt[:30]}"

    def call_llm(self, prompt: str, **kwargs) -> str:
        logger.debug(f"V4-PlaceholderLLMPort: Sync call_llm with prompt: {prompt[:50]}...")
        return f"V4 Sync LLM response to: {prompt[:30]}"

class PlaceholderEmbeddingPortImpl(EmbeddingPort):
    def encode(self, text: str) -> Vector:
        logger.debug(f"V4-PlaceholderEmbeddingPort: encode '{text[:50]}...'")
        logger.debug(f"V4-PlaceholderEmbeddingPort: Using DEFAULT_DTYPE: {DEFAULT_DTYPE} (numpy name: {np.dtype(DEFAULT_DTYPE).name})")
        # V4: Ensure Vector is from V4 domain
        return Vector(data=np.array([random.random() for _ in range(10)], dtype=DEFAULT_DTYPE))

    def encode_batch(self, texts: Sequence[str]) -> List[Vector]:
        logger.debug(f"V4-PlaceholderEmbeddingPort: encode_batch for {len(texts)} texts.")
        return [self.encode(text) for text in texts]

class PlaceholderEventBusImpl(EventBusPort):
    def __init__(self):
        self._loggers: Dict[str, logging.Logger] = {}

    def publish(self, event_type: str, payload: Any) -> None:
        logger.debug(f"V4-PlaceholderEventBus: Event '{event_type}' published with payload: {payload}")

    def subscribe(self, event_type: str, handler: Any) -> None:
        logger.debug(f"V4-PlaceholderEventBus: Subscribed handler to '{event_type}'")

    def get_logger(self, component_id: str) -> logging.Logger: # Added from V3 EventBus
        if component_id not in self._loggers:
            self._loggers[component_id] = logging.getLogger(f"nireon_v4.{component_id}")
        return self._loggers[component_id]


class PlaceholderIdeaRepositoryImpl(IdeaRepositoryPort):
    def __init__(self):
        self._ideas: Dict[str, Idea] = {} # V4: Use V4 Idea
        self._child_relationships: Dict[str, List[str]] = {} # V4: Keep similar structure
        self._world_facts: Dict[str, List[str]] = {} # V4: Keep similar structure
        logger.info("V4-PlaceholderIdeaRepositoryImpl initialized with in-memory store.")

    def save(self, idea: Idea) -> None: # V4: Use V4 Idea
        logger.debug(f"V4-PlaceholderIdeaRepository: save idea '{idea.idea_id}'")
        self._ideas[idea.idea_id] = idea

    def get_by_id(self, idea_id: str) -> Optional[Idea]: # V4: Use V4 Idea
        idea = self._ideas.get(idea_id)
        if idea:
            logger.debug(f"V4-PlaceholderIdeaRepository: get_by_id '{idea_id}' -> FOUND")
        else:
            logger.debug(f"V4-PlaceholderIdeaRepository: get_by_id '{idea_id}' -> None")
        return idea

    def get_all(self) -> List[Idea]: # V4: Use V4 Idea
        all_ideas = list(self._ideas.values())
        logger.debug(f"V4-PlaceholderIdeaRepository: get_all -> {len(all_ideas)} ideas")
        return all_ideas
    
    # V4: These methods are part of the V3 Port, retain them.
    def get_by_parent_id(self, parent_id: str) -> List[Idea]:
        children_ids = self._child_relationships.get(parent_id, [])
        children = [self._ideas[cid] for cid in children_ids if cid in self._ideas]
        logger.debug(f"V4-PlaceholderIdeaRepository: get_by_parent_id '{parent_id}' -> {len(children)} ideas")
        return children

    def add_child_relationship(self, parent_id: str, child_id: str) -> bool:
        if parent_id not in self._ideas or child_id not in self._ideas:
            logger.warning(f"V4-PlaceholderIdeaRepository: add_child_relationship failed. Parent '{parent_id}' or child '{child_id}' not found.")
            return False
        self._child_relationships.setdefault(parent_id, []).append(child_id)
        # Optionally update Idea domain objects if they have direct children/parent lists
        if hasattr(self._ideas[parent_id], 'children') and isinstance(self._ideas[parent_id].children, list):
             if child_id not in self._ideas[parent_id].children:
                self._ideas[parent_id].children.append(child_id)
        if hasattr(self._ideas[child_id], 'parent_ids') and isinstance(self._ideas[child_id].parent_ids, list):
             if parent_id not in self._ideas[child_id].parent_ids:
                self._ideas[child_id].parent_ids.append(parent_id)
        logger.debug(f"V4-PlaceholderIdeaRepository: add_child_relationship parent='{parent_id}', child='{child_id}' -> True")
        return True

    def add_world_fact(self, idea_id: str, fact_id: str) -> bool:
        if idea_id not in self._ideas:
            logger.warning(f"V4-PlaceholderIdeaRepository: add_world_fact failed. Idea '{idea_id}' not found.")
            return False
        self._world_facts.setdefault(idea_id, []).append(fact_id)
        if hasattr(self._ideas[idea_id], 'world_facts') and isinstance(self._ideas[idea_id].world_facts, list):
            if fact_id not in self._ideas[idea_id].world_facts:
                self._ideas[idea_id].world_facts.append(fact_id)
        logger.debug(f"V4-PlaceholderIdeaRepository: add_world_fact idea='{idea_id}', fact='{fact_id}' -> True")
        return True
```

**`nireon_v4/bootstrap/bootstrap_helper/service_resolver.py`**
```python
# Adapted from nireon_staging/nireon/application/bootstrap_helper/service_resolver.py
import asyncio
import dataclasses # V4: Added for ComponentMetadata manipulation
import logging
import inspect # V4: Added for _call_by_signature
from typing import Any, Dict, Type, Optional, Callable

# V4 imports
from nireon_v4.application.components.base import NireonBaseComponent
from nireon_v4.application.components.lifecycle import ComponentMetadata, ComponentRegistryMissingError
# V4: ComponentRegistry is now in core.registry
from nireon_v4.core.registry.component_registry import ComponentRegistry
from nireon_v4.application.ports.event_bus_port import EventBusPort
from nireon_v4.application.ports.idea_repository_port import IdeaRepositoryPort
from nireon_v4.application.services.idea_service import IdeaService
# V4: Other ports and services from V3 resolver might be needed later. For P1, these are key.
# from nireon_v4.application.ports.llm_port import LLMPort
# from nireon_v4.application.ports.world_fact_repository_port import WorldFactRepositoryPort
# from nireon_v4.application.ports.external_knowledge_port import ExternalKnowledgePort

from .metadata import create_service_metadata # V4: Local import

logger = logging.getLogger(__name__)

# V4: FeatureFlagsManager related lock, keep for now.
_ff_manager_lock = asyncio.Lock()

# V4: Define BootstrapError if not defined elsewhere in this helper package yet
class BootstrapError(RuntimeError):
    pass


def _safe_register_service_instance(
    registry: ComponentRegistry,
    service_protocol_type: Type, # Protocol type
    instance: Any,
    service_id_for_meta: str, # Preferred ID for metadata if not NireonBaseComponent
    category_for_meta: str,
    description_for_meta: Optional[str] = None,
    requires_initialize_override: Optional[bool] = None
) -> None:
    already_by_type = False
    already_by_id = False
    normalized_type_key = None

    # V4: ComponentRegistry has normalize_key, not _normalize_key
    if hasattr(registry, 'normalize_key') and callable(registry.normalize_key):
        normalized_type_key = registry.normalize_key(service_protocol_type)
    
    if hasattr(registry, 'get_service_instance'):
        try:
            existing_by_type = registry.get_service_instance(service_protocol_type)
            if existing_by_type is instance:
                already_by_type = True
            elif existing_by_type is not None and existing_by_type is not instance:
                logger.warning(
                    f"Service {service_protocol_type.__name__} already registered "
                    f"with a different instance (by type). Overwriting with new instance for type key."
                )
        except (ComponentRegistryMissingError, AttributeError):
            pass # Not registered by type yet or registry doesn't support it fully

    try:
        existing_by_id = registry.get(service_id_for_meta)
        if existing_by_id is instance:
            already_by_id = True
        elif existing_by_id is not None and existing_by_id is not instance:
             logger.warning(
                f"Service '{service_id_for_meta}' already registered "
                f"with a different instance (by ID). Overwriting with new instance for ID key."
            )
    except (ComponentRegistryMissingError, AttributeError): # V4: Use V4 error
        pass # Not registered by ID yet

    final_requires_initialize = False # V4: Default to False for simple services
    if requires_initialize_override is not None:
        final_requires_initialize = requires_initialize_override
    elif isinstance(instance, NireonBaseComponent):
        if hasattr(instance, 'metadata') and isinstance(instance.metadata, ComponentMetadata):
            final_requires_initialize = instance.metadata.requires_initialize
        else: # If NireonBaseComponent but no proper metadata, assume it needs init
            final_requires_initialize = True 
            
    desc = description_for_meta or f"Service instance for {service_id_for_meta}"
    id_metadata = create_service_metadata(
        service_id=service_id_for_meta,
        service_name=service_id_for_meta, # Name can be same as ID for services
        category=category_for_meta,
        description=desc,
        requires_initialize=final_requires_initialize
    )

    if not already_by_type and hasattr(registry, 'register_service_instance'):
        try:
            registry.register_service_instance(service_protocol_type, instance)
            logger.debug(f"Service '{service_id_for_meta}' (type: {service_protocol_type.__name__}) registered by type key.")
            # Register metadata for the type key if different from service_id_for_meta
            if normalized_type_key and normalized_type_key != service_id_for_meta:
                type_metadata = create_service_metadata(
                    service_id=normalized_type_key,
                    service_name=f"{service_protocol_type.__name__} (type registration)",
                    category=category_for_meta,
                    description=f"Type-based registration for {service_protocol_type.__name__}",
                    requires_initialize=final_requires_initialize
                )
                # V4: Registry's internal _metadata might not be directly accessible.
                # Rely on register method to handle metadata if it exists for type keys.
                # If ComponentRegistry.register requires metadata for type keys, this needs adjustment.
                # For now, assume register_service_instance handles type-key metadata implicitly or it's not strictly needed.
                if hasattr(registry, '_metadata') and isinstance(registry._metadata, dict):
                    if normalized_type_key not in registry._metadata:
                         registry._metadata[normalized_type_key] = type_metadata
                         logger.debug(f"Metadata registered for type key '{normalized_type_key}'.")

        except Exception as e:
            logger.error(f"Failed to register '{service_id_for_meta}' by type key {service_protocol_type.__name__}: {e}", exc_info=True)

    if not already_by_id:
        try:
            if isinstance(instance, NireonBaseComponent) and \
               hasattr(instance, 'metadata') and isinstance(instance.metadata, ComponentMetadata):
                meta_to_register = instance.metadata
                if meta_to_register.id != service_id_for_meta:
                    logger.warning(f"Mismatch for NireonBaseComponent '{service_id_for_meta}': instance metadata ID is '{meta_to_register.id}'. Using '{service_id_for_meta}' for registration key.")
                    # Correct the ID in a copy of the metadata for registration
                    corrected_meta_dict = dataclasses.asdict(meta_to_register)
                    corrected_meta_dict['id'] = service_id_for_meta
                    meta_to_register = ComponentMetadata(**corrected_meta_dict)
                registry.register(instance, meta_to_register) # V4: NireonBaseComponent has its own metadata
                logger.debug(f"NireonBaseComponent service '{service_id_for_meta}' registered with its metadata (ID: {meta_to_register.id}, ReqInit: {meta_to_register.requires_initialize}).")
            else:
                registry.register(instance, id_metadata) # Register with generated metadata
                logger.debug(f"Service '{service_id_for_meta}' registered with metadata (ID: {id_metadata.id}, ReqInit: {id_metadata.requires_initialize}).")
        except Exception as e:
            logger.error(f"Failed to register '{service_id_for_meta}' with metadata: {e}", exc_info=True)
            
    if already_by_type and already_by_id:
        logger.debug(f"Service '{service_id_for_meta}' (type: {service_protocol_type.__name__}) already fully registered with the same instance.")


def get_or_create_service(
    registry: ComponentRegistry,
    service_protocol_type: Type,
    placeholder_impl_class: Type,
    service_friendly_name: str,
    instance_id_prefix: str = "placeholder_",
    category: str = "placeholder_service",
    requires_initialize_for_placeholder: bool = False,
    **kwargs # Passthrough for placeholder constructor
) -> Any:
    try:
        service_instance = registry.get_service_instance(service_protocol_type)
        logger.info(f"'{service_friendly_name}' found in registry via service type. Using existing instance: {type(service_instance).__name__}.")
        return service_instance
    except (ComponentRegistryMissingError, AttributeError): # V4: Use V4 error
        pass # Continue to other resolution methods or creation

    # V4: Check by normalized key if ComponentRegistry supports it
    if hasattr(registry, 'normalize_key') and callable(registry.normalize_key):
        try:
            normalized_key = registry.normalize_key(service_protocol_type)
            service_instance = registry.get(normalized_key) # get by ID/normalized key
            logger.info(f"'{service_friendly_name}' found in registry with normalized key '{normalized_key}'. Using existing instance: {type(service_instance).__name__}.")
            return service_instance
        except (ComponentRegistryMissingError, AttributeError):
            pass

    # V4: Iterate and check type (fallback)
    try:
        for comp_id in registry.list_components():
            comp = registry.get(comp_id)
            if isinstance(comp, service_protocol_type):
                logger.info(f"'{service_friendly_name}' found in registry by type matching (ID: '{comp_id}'). Using existing instance: {type(comp).__name__}.")
                return comp
    except Exception as e: # Catch broad exceptions during iteration if registry is unstable
        logger.warning(f"Error during type matching search for '{service_friendly_name}': {e}")

    logger.warning(f"'{service_friendly_name}' not found in registry. Creating placeholder: {placeholder_impl_class.__name__}.")
    placeholder_instance = placeholder_impl_class(**kwargs)
    
    service_id_for_meta = f"{instance_id_prefix}{service_friendly_name.replace('.', '_').replace(' ', '')}"
    
    _safe_register_service_instance(
        registry, 
        service_protocol_type, 
        placeholder_instance, 
        service_id_for_meta, 
        category,
        description_for_meta=f"Placeholder implementation for {service_friendly_name}",
        requires_initialize_override=requires_initialize_for_placeholder
    )
    logger.info(f"Placeholder '{service_friendly_name}' instance created and registered (ID: {service_id_for_meta}).")
    return placeholder_instance


def get_or_create_idea_service(
    registry: ComponentRegistry,
    idea_repo: IdeaRepositoryPort,
    event_bus: EventBusPort
) -> IdeaService:
    try:
        idea_service_instance = registry.get_service_instance(IdeaService)
        logger.info("IdeaService found in registry. Using existing instance.")
        return idea_service_instance
    except (ComponentRegistryMissingError, AttributeError): # V4: Use V4 error
        logger.info("IdeaService not found in registry. Creating new instance.")
        # V4: Ensure IdeaService is imported from V4 services
        idea_service_instance = IdeaService(repository=idea_repo, event_bus=event_bus)
        _safe_register_service_instance(
            registry, 
            IdeaService, 
            idea_service_instance, 
            "bootstrap_idea_service_v4", # V4 ID
            "domain_service",
            description_for_meta="IdeaService created during V4 bootstrap"
        )
        logger.info("IdeaService created and registered.")
        return idea_service_instance

# V4 Phase 1: create_service_instance and its helpers might be too complex for P1.
# They are used in enhanced manifest processing.
# Let's keep find_event_bus_service as it's simple and used by _finalize_bootstrap.

def find_event_bus_service(registry: ComponentRegistry) -> Optional[EventBusPort]:
    try:
        return registry.get_service_instance(EventBusPort)
    except (ComponentRegistryMissingError, AttributeError): # V4: Use V4 error
        logger.debug("EventBusPort not found via get_service_instance. Trying string key 'EventBusPort'.")
        try:
            bus_candidate = registry.get('EventBusPort') # Check common string key
            if isinstance(bus_candidate, EventBusPort):
                return bus_candidate
        except (ComponentRegistryMissingError, AttributeError):
             logger.debug("EventBusPort not found via string key. Trying duck typing.")
             # Fallback to duck typing (less ideal, but robust for various registration methods)
             for comp_id in registry.list_components():
                comp = registry.get(comp_id)
                if hasattr(comp, 'publish') and callable(getattr(comp, 'publish')) and \
                   hasattr(comp, 'subscribe') and callable(getattr(comp, 'subscribe')):
                    # V4: Check for get_logger for more robust EventBusPort identification
                    if hasattr(comp, 'get_logger') and callable(getattr(comp, 'get_logger')):
                        logger.debug(f"Found EventBus by duck typing (with get_logger): {comp_id} ({type(comp)})")
                        return comp # type: ignore
                    elif not hasattr(comp, 'get_logger'): # Basic publish/subscribe is enough for some uses
                        logger.debug(f"Found EventBus by basic duck typing (publish/subscribe only): {comp_id} ({type(comp)})")
                        return comp # type: ignore
    logger.warning("EventBusPort not found in registry through any method.")
    return None

# V4: _call_by_signature might be useful later, keep it.
def _call_by_signature(cls, /, *args, **kwargs):
    sig = inspect.signature(cls.__init__)
    accepted = {k:v for k,v in kwargs.items() if k in sig.parameters}
    return cls(*args, **accepted)

```

**`nireon_v4/bootstrap/bootstrap_helper/utils.py`** (Copy from V3, update imports)
```python
# Adapted from nireon_staging/nireon/application/bootstrap_helper/utils.py
import importlib
import logging
from pathlib import Path
from types import ModuleType # V4: Keep for type hint consistency
from typing import Any, Dict, Mapping # V4: Keep Mapping for type hint consistency
import yaml

logger = logging.getLogger(__name__)

def import_by_path(path: str, suppress_expected_errors: bool = True) -> Any:
    if not isinstance(path, str):
        raise TypeError(f'Import path must be a string, got {type(path)}')

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    elif "." in path:
        module_name, attr_name = path.rsplit(".", 1)
        if not module_name and attr_name: # e.g. ".MyClass"
             raise ValueError(f"Relative import path '{path}' not supported. Provide full path.")
    else:
        logger.error(f"Import path '{path}' is ambiguous. Use 'pkg.mod:Class' or 'pkg.mod.func'.")
        raise ValueError(f"Import path '{path}' is ambiguous. Use 'pkg.mod:Class' or 'pkg.mod.func'.")

    if not module_name or not attr_name: # Check after split
        logger.error(f"Could not parse module and attribute from path '{path}'. Module: '{module_name}', Attribute: '{attr_name}'.")
        raise ValueError(f'Invalid import path format: {path}. Could not determine module and attribute.')

    log_level_if_not_found = logging.DEBUG if suppress_expected_errors else logging.ERROR
    exc_log_level_if_not_found = logging.DEBUG if suppress_expected_errors else logging.ERROR # For exc_info

    try:
        module = importlib.import_module(module_name)
        logger.debug(f'Successfully imported module: {module_name}')
    except ImportError as e:
        logger.log(log_level_if_not_found, f"Failed to import module '{module_name}' from path '{path}': {e}", exc_info=(log_level_if_not_found >= logging.ERROR))
        raise ImportError(f"Could not import module '{module_name}': {e}") from e

    try:
        attribute = getattr(module, attr_name)
        logger.debug(f"Successfully retrieved attribute '{attr_name}' from module '{module_name}'.")
        return attribute
    except AttributeError as e:
        logger.log(log_level_if_not_found, f"Attribute '{attr_name}' not found in module '{module_name}' (from path '{path}'): {e}", exc_info=(log_level_if_not_found >= logging.ERROR))
        raise AttributeError(f"Attribute '{attr_name}' not found in module '{module_name}': {e}") from e


def load_yaml_robust(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    
    path_str_representation = str(path) # For checking specific placeholder strings
    
    # V4: Check for known placeholder path strings explicitly
    if path_str_representation == 'nireon/configs/default/components/{id}.yaml':
        logger.debug(f"Path '{path_str_representation}' is the known placeholder for default component config. Returning empty config.")
        return {}
    elif '{id}' in path_str_representation or '{ID}' in path_str_representation: # General placeholder check
        logger.debug(f"Path '{path_str_representation}' contains placeholder tokens. Assuming default component config. Returning empty config.")
        return {}

    actual_path = Path(path)
    if not actual_path.exists():
        logger.warning(f"YAML file not found: '{actual_path}'. Returning empty config.")
        return {}

    try:
        with open(actual_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {} # Ensure dict even if file is empty
            if not isinstance(data, Mapping): # V4: Use Mapping for broader dict-like check
                logger.error(f"Top-level YAML object in '{actual_path}' must be a mapping/dictionary. Found {type(data)}. Returning empty config.")
                return {}
            return data
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file '{actual_path}': {exc}", exc_info=True)
        return {}
    except Exception as exc: # Catch other potential file I/O errors
        logger.error(f"Unexpected error loading YAML file '{actual_path}': {exc}", exc_info=True)
        return {}


def detect_manifest_type(config_data: Dict[str, Any], style_hint: str = 'auto') -> str:
    if style_hint != 'auto':
        return style_hint
    
    # Simple manifest markers (V3 style)
    if 'components' in config_data or 'nireon_components' in config_data:
        return 'simple'
    
    # Enhanced manifest markers (V3 style)
    manifest_keys = {
        'shared_services', 'mechanisms', 'observers', 
        'managers', 'composites', 'orchestration_commands'
    }
    if any(key in config_data for key in manifest_keys):
        return 'enhanced'
        
    # V4: Default or if unknown structure, assume simple for now or raise error
    # For Phase 1, defaulting to 'simple' if unsure is safer.
    # Later phases might introduce a V4-specific manifest type.
    return 'simple'
```

**`nireon_v4/bootstrap/bootstrap_helper/health_reporter.py`**
```python
# Adapted from nireon_staging/nireon/application/bootstrap_helper/health_reporter.py
import logging
from enum import Enum
from typing import Any, Dict, List

# V4 imports
from nireon_v4.application.components.lifecycle import ComponentMetadata
from nireon_v4.core.registry.component_registry import ComponentRegistry # V4: Use V4 ComponentRegistry

logger = logging.getLogger(__name__)

class ComponentStatus(Enum): # Keep V3 statuses for now
    DEFINITION_ERROR = "DEFINITION_ERROR"
    METADATA_ERROR = "METADATA_ERROR"
    METADATA_CONSTRUCTION_ERROR = "METADATA_CONSTRUCTION_ERROR"
    INSTANTIATION_ERROR = "INSTANTIATION_ERROR"
    INSTANCE_REGISTERED = "INSTANCE_REGISTERED"
    REGISTRATION_ERROR = "REGISTRATION_ERROR"
    BOOTSTRAP_ERROR = "BOOTSTRAP_ERROR"
    INITIALIZED_OK = "INITIALIZED_OK"
    INITIALIZATION_ERROR = "INITIALIZATION_ERROR"
    INITIALIZED_SKIPPED_NO_METHOD = "INITIALIZED_SKIPPED_NO_METHOD"
    INSTANCE_REGISTERED_NO_INIT = "INSTANCE_REGISTERED_NO_INIT" # Component registered, init not required/attempted.
    INSTANCE_REGISTERED_INIT_DEFERRED = "INSTANCE_REGISTERED_INIT_DEFERRED" # Registered, but init will happen in a later pass
    VALIDATED_OK = "VALIDATED_OK"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    HEALTHY = "HEALTHY" # Generic healthy state post-bootstrap, if applicable.

class BootstrapHealthReporter:
    def __init__(self, component_registry: ComponentRegistry):
        self.component_registry = component_registry # V4 registry
        self.component_statuses: Dict[str, Dict[str, Any]] = {}
        self.validation_errors: Dict[str, List[str]] = {} # V4: Store errors per component

    def add_component_status(
        self,
        component_id: str,
        status: ComponentStatus,
        metadata: ComponentMetadata, # V4: Use V4 ComponentMetadata
        validation_errors: List[str] # V4: Allow passing specific errors
    ):
        new_status_entry = {
            "id": component_id,
            "name": metadata.name,
            "type": metadata.category, # Using category as 'type' for reporting
            "status": status.value,
            "epistemic_tags": metadata.epistemic_tags,
            "validation_issue_count": len(validation_errors),
        }
        self.component_statuses[component_id] = new_status_entry
        
        if validation_errors:
            self.validation_errors[component_id] = validation_errors
        elif component_id in self.validation_errors: # Clear if previously had errors
            del self.validation_errors[component_id]
            
    def has_critical_failures(self) -> int: # Added for BootstrapResult.success
        critical_statuses = {
            ComponentStatus.DEFINITION_ERROR.value,
            ComponentStatus.METADATA_ERROR.value,
            ComponentStatus.METADATA_CONSTRUCTION_ERROR.value,
            ComponentStatus.INSTANTIATION_ERROR.value,
            ComponentStatus.REGISTRATION_ERROR.value,
            ComponentStatus.BOOTSTRAP_ERROR.value,
            # INITIALIZATION_ERROR and VALIDATION_FAILED can be critical depending on strict_mode
        }
        return sum(1 for s_info in self.component_statuses.values() if s_info['status'] in critical_statuses)


    def generate_summary(self) -> str:
        summary_lines = ["\n--- NIREON V4 Bootstrap Health Report ---"] # V4
        all_statuses = list(self.component_statuses.values())
        total_components = len(all_statuses)

        successful_components = sum(
            1 for s in all_statuses if s["status"] in [
                ComponentStatus.INITIALIZED_OK.value,
                ComponentStatus.HEALTHY.value,
                ComponentStatus.VALIDATED_OK.value,
                ComponentStatus.INSTANCE_REGISTERED_NO_INIT.value,
                ComponentStatus.INSTANCE_REGISTERED_INIT_DEFERRED.value,
                ComponentStatus.INSTANCE_REGISTERED.value # V4: Added INSTANCE_REGISTERED as success for P1
            ]
        )
        
        error_statuses = [
            ComponentStatus.DEFINITION_ERROR.value,
            ComponentStatus.METADATA_ERROR.value,
            ComponentStatus.METADATA_CONSTRUCTION_ERROR.value,
            ComponentStatus.INSTANTIATION_ERROR.value,
            ComponentStatus.REGISTRATION_ERROR.value,
            ComponentStatus.BOOTSTRAP_ERROR.value,
            ComponentStatus.INITIALIZATION_ERROR.value,
            ComponentStatus.VALIDATION_FAILED.value,
        ]
        failed_components = sum(1 for s in all_statuses if s["status"] in error_statuses)

        summary_lines.append(f"Total Components Processed: {total_components}")
        summary_lines.append(f"Successfully Loaded/Initialized/Validated: {successful_components}")
        summary_lines.append(f"Failed or Errored during Bootstrap: {failed_components}")

        if self.validation_errors:
            summary_lines.append("\nComponents with Current Validation/Bootstrap Issues:")
            for comp_id, errors in self.validation_errors.items():
                status_detail = self.component_statuses.get(comp_id, {})
                name = status_detail.get("name", comp_id)
                current_status = status_detail.get("status", "UNKNOWN_STATUS")
                summary_lines.append(f"  - ID: {comp_id} (Name: {name}, Status: {current_status})")
                for error in errors:
                    summary_lines.append(f"    - {error}")
        
        # V4: Epistemic tag distribution, keep from V3
        epistemic_tag_counts: Dict[str, int] = {}
        for status_info in all_statuses:
            if status_info["status"] in [ComponentStatus.INITIALIZED_OK.value, ComponentStatus.HEALTHY.value, ComponentStatus.VALIDATED_OK.value]:
                for tag in status_info.get("epistemic_tags", []):
                    epistemic_tag_counts[tag] = epistemic_tag_counts.get(tag, 0) + 1
        
        if epistemic_tag_counts:
            summary_lines.append("\nEpistemic Tag Distribution (for successfully processed components):")
            for tag, count in sorted(epistemic_tag_counts.items()):
                summary_lines.append(f"  - {tag}: {count}")

        # V4: Certification summary, adapt from V3
        self._add_certification_summary(summary_lines)

        summary_lines.append("--- End of V4 Report ---\n") # V4
        return "\n".join(summary_lines)

    def _add_certification_summary(self, summary_lines: List[str]):
        try:
            # V4: ComponentRegistry is passed in constructor
            registered_components = self.component_registry.list_components()
            if registered_components:
                summary_lines.append(f"\nTotal Registered Components: {len(registered_components)}")
                healthy_count = 0
                unhealthy_count = 0
                uncertified_count = 0

                for comp_id in registered_components:
                    try:
                        # V4: Check for get_certification method
                        if hasattr(self.component_registry, 'get_certification') and \
                           callable(self.component_registry.get_certification):
                            cert_data = self.component_registry.get_certification(comp_id)
                            if cert_data:
                                cert_status = cert_data.get("status", "unknown")
                                if cert_status in [
                                    "healthy", "initialized_ok", 
                                    "registered_no_init_method", "registered_init_deferred",
                                    ComponentStatus.INITIALIZED_OK.value.lower(), # Add enum versions
                                    ComponentStatus.HEALTHY.value.lower(),
                                    ComponentStatus.VALIDATED_OK.value.lower(),
                                    ComponentStatus.INSTANCE_REGISTERED.value.lower(), # V4: Added
                                    ComponentStatus.INSTANCE_REGISTERED_NO_INIT.value.lower()
                                ]:
                                    healthy_count += 1
                                else:
                                    unhealthy_count += 1
                            else:
                                uncertified_count += 1
                        else: # If no get_certification, assume uncertified
                            uncertified_count +=1
                    except Exception: # Catch errors during cert retrieval for a specific component
                        uncertified_count += 1
                
                summary_lines.append(f"  - Healthy/Registered: {healthy_count}")
                summary_lines.append(f"  - Unhealthy/Errored (in cert): {unhealthy_count}")
                summary_lines.append(f"  - Uncertified/NoCert: {uncertified_count}")
        except Exception as e:
            logger.debug(f"Could not generate detailed certification summary: {e}")
```

**`nireon_v4/configs/__init__.py`** (empty)

**`nireon_v4/configs/config_utils.py`** (Copy from V3 `nireon_staging/nireon/configs/config_utils.py`)
```python
# Adapted from nireon_staging/nireon/configs/config_utils.py
import logging
import copy
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigMerger:
    @staticmethod
    def merge(
        base: Dict[str, Any], 
        override: Dict[str, Any], 
        context_description: str = "ConfigMerge",
        strict_keys: bool = False, # If True, override keys must exist in base (unless allow_new_keys_in_strict)
        allow_new_keys_in_strict: bool = False # If strict_keys is True, allow new keys from override
    ) -> Dict[str, Any]:
        
        if not isinstance(base, dict):
            logger.error(f"[{context_description}] Base for merge is not a dictionary (type: {type(base)}). Returning override or empty.")
            return copy.deepcopy(override) if isinstance(override, dict) else {}
        if not isinstance(override, dict):
            logger.warning(f"[{context_description}] Override for merge is not a dictionary (type: {type(override)}). Returning base.")
            return copy.deepcopy(base)

        merged = copy.deepcopy(base)
        logger.debug(f"[{context_description}] Starting merge. Base keys: {list(base.keys())}, Override keys: {list(override.keys())}")

        for key, override_value in override.items():
            base_value = merged.get(key)

            if key not in merged: # Key is new from override
                if strict_keys and not allow_new_keys_in_strict:
                    raise ValueError(f"[{context_description}] Strict mode: Key '{key}' in override not found in base, and new keys not allowed.")
                merged[key] = copy.deepcopy(override_value)
                logger.debug(f"[{context_description}] Added new key '{key}' with value: {str(override_value)[:80]}{('...' if len(str(override_value)) > 80 else '')}")
            elif isinstance(base_value, dict) and isinstance(override_value, dict):
                # Recursive merge for nested dictionaries
                logger.debug(f"[{context_description}] Recursively merging dict for key '{key}'.")
                merged[key] = ConfigMerger.merge(
                    base_value, 
                    override_value, 
                    context_description=f"{context_description} -> {key}",
                    strict_keys=strict_keys,
                    allow_new_keys_in_strict=allow_new_keys_in_strict
                )
            elif base_value == override_value:
                # V3: Handle case where key might exist with None in base, but override has a value
                if base_value is None and key not in base: # Check if it was truly missing or explicitly None
                     merged[key] = copy.deepcopy(override_value)
                logger.debug(f"[{context_description}] Key '{key}' has same value in base and override. No change.")
            else: # Override value is different and not a dict, or types mismatch for recursive merge
                merged[key] = copy.deepcopy(override_value)
                original_base_value_repr = str(base.get(key))[:80] # Use base.get for original
                if len(str(base.get(key))) > 80: original_base_value_repr += '...'
                override_value_repr = str(override_value)[:80]
                if len(str(override_value)) > 80: override_value_repr += '...'
                logger.debug(f"[{context_description}] Overridden key '{key}'. Old: {original_base_value_repr}, New: {override_value_repr}")
        
        logger.debug(f"[{context_description}] Merge complete for this level. Result keys: {list(merged.keys())}")
        return merged

def merge_configs(base: Dict[str, Any], *overrides: Dict[str, Any], context:str = "ConfigChain", strict:bool = False) -> Dict[str, Any]:
    """ Helper to chain multiple merges. """
    result = base
    for i, override in enumerate(overrides):
        result = ConfigMerger.merge(result, override, context_description=f"{context}_Step{i+1}", strict_keys=strict, allow_new_keys_in_strict=True) # Allow new keys in chained merge steps
    return result

# V4: Main guard from V3 is useful for direct testing.
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    merger = ConfigMerger()

    base_conf = {"a": 1, "b": {"c": 2, "d": 3}, "e": 5}
    override_conf = {"b": {"c": 20, "f": 6}, "e": 50, "g": 7}

    print("\n--- Test Case 1: Standard Merge ---")
    merged1 = merger.merge(base_conf, override_conf, "TestCase1")
    print("Merged Result 1:", merged1)
    expected1 = {'a': 1, 'b': {'c': 20, 'd': 3, 'f': 6}, 'e': 50, 'g': 7}
    assert merged1 == expected1, f"Expected {expected1}, got {merged1}"

    print("\n--- Test Case 2: Strict Keys, New Keys Not Allowed (should fail) ---")
    try:
        merger.merge(base_conf, override_conf, "TestCase2", strict_keys=True, allow_new_keys_in_strict=False)
        assert False, "Expected ValueError in strict mode for new keys"
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
        assert "Key 'f' in override not found in base" in str(e) or \
               "Key 'g' in override not found in base" in str(e)
    
    # ... (other test cases from V3 can be included here) ...

    print("\nAll ConfigMerger tests passed! ✅")
```

**`nireon_v4/configs/loader.py`**
```python
# Adapted from nireon_staging/nireon/configs/loader.py
import json
import logging
import os
import re
import copy # V4: Ensure deepcopy for default config
from pathlib import Path
from typing import Any, Dict

import yaml

# V4: Use V4 ConfigMerger
from .config_utils import ConfigMerger 

logger = logging.getLogger(__name__)

# V4: Define a baseline default config. This can be expanded.
DEFAULT_CONFIG: Dict[str, Any] = {
    "env": "default",
    "feature_flags": {
        "debug_mode": False,
    },
    "storage": { # V4: As per V3 structure
        "lineage_db_path": "runtime/lineage_v4.db", # V4 specific path
        "ideas_path": "runtime/ideas_v4.json",
        "facts_path": "runtime/world_facts_v4.json",
    },
    "logging": { # V4: As per V3 structure
         "prompt_response_logger": {"output_dir": "runtime/llm_logs_v4"},
         "pipeline_event_logger": {"output_dir": "runtime/pipeline_event_logs_v4"}
    },
    "bootstrap_strict_mode": False, # V4: Keep this crucial flag
    "reactor_rules_module": "nireon_v4.application.orchestration.reactor_rules.default_rules" # V4 path
}

def _expand_env_var_string(value_str: str) -> str:
    if not isinstance(value_str, str):
        return value_str # type: ignore

    original_value = value_str

    # Regex for ${VAR:-default} or ${VAR}
    def repl_default(match_default):
        var_name = match_default.group(1)
        default_val = match_default.group(2) # Can be empty if :- is not used
        env_val = os.getenv(var_name)
        result = env_val if env_val is not None else default_val
        logger.debug(f"Expanding ${{{var_name}:-{default_val}}} -> '{result}' (env_val={env_val})")
        return result

    # V4: Updated regex to handle optional default value more robustly
    expanded_str = re.sub(r'\$\{([\w_]+):-?([^}]*)\}', repl_default, value_str)
    # Fallback for simple ${VAR}
    expanded_str = os.path.expandvars(expanded_str)

    if original_value != expanded_str:
        logger.debug(f"Expanded env vars in string: '{original_value}' -> '{expanded_str}'")
    elif '${' in original_value and '}' in original_value: # Check if any unexpanded ${VAR} syntax remains
        logger.warning(f"Environment variable placeholder not expanded: '{original_value}' - check if env var is set or syntax is correct (e.g. ${{VAR:-default_val}} or ${{VAR}})")
    return expanded_str

def _expand_env_vars_in_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in config_dict.items():
        config_dict[key] = _expand_env_vars_in_item(value)
    return config_dict

def _expand_env_vars_in_item(item: Any) -> Any:
    if isinstance(item, dict):
        return _expand_env_vars_in_dict(item)
    elif isinstance(item, list):
        return [_expand_env_vars_in_item(i) for i in item]
    elif isinstance(item, str):
        return _expand_env_var_string(item)
    return item

def load_config(env: str = "default") -> Dict[str, Any]:
    # V4: Determine package root based on this file's location
    # Assumes loader.py is in nireon_v4/configs/
    package_root = Path(__file__).resolve().parents[1] 
    
    config = copy.deepcopy(DEFAULT_CONFIG) # Start with hardcoded defaults
    config["env"] = env # Set the current environment
    logger.debug(f"Initialized config with DEFAULT_CONFIG for env '{env}'. Package root: {package_root}")

    # Define paths for global_app_config.yaml
    config_files_to_load = []
    default_global_path = package_root / "configs" / "default" / "global_app_config.yaml"
    config_files_to_load.append(("DEFAULT_GLOBAL_APP_CONFIG", default_global_path))
    
    if env != "default":
        env_global_path = package_root / "configs" / env / "global_app_config.yaml"
        config_files_to_load.append((f"ENV_GLOBAL_APP_CONFIG ({env})", env_global_path))

    for label, path in config_files_to_load:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    content = yaml.safe_load(fh) or {}
                if isinstance(content, dict):
                    config = ConfigMerger.merge(config, content, label)
                    logger.info(f"Loaded {label}: {path}")
                else:
                    logger.warning(f"{label} is not a dict: {path}")
            except Exception as exc:
                logger.error(f"Error loading {label} from {path}: {exc}", exc_info=True)
        elif "ENV_GLOBAL" in label: # Only warn if env-specific is missing
             logger.warning(f"{label} not found: {path}")


    # Load and merge LLM configurations
    llm_config_from_yaml: Dict[str, Any] = {}
    llm_config_paths_to_check = []
    default_llm_path = package_root / "configs" / "default" / "llm_config.yaml"
    llm_config_paths_to_check.append(("DEFAULT_LLM_CONFIG", default_llm_path))

    if env != "default":
        env_llm_path = package_root / "configs" / env / "llm_config.yaml"
        llm_config_paths_to_check.append((f"ENV_LLM_CONFIG ({env})", env_llm_path))

    for label, path in llm_config_paths_to_check:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    content = yaml.safe_load(fh) or {}
                if isinstance(content, dict):
                    llm_config_from_yaml = ConfigMerger.merge(llm_config_from_yaml, content, f"llm_config_merge: {label}")
                    logger.info(f"Loaded and merged LLM config from {label}: {path}")
                else:
                    logger.warning(f"{label} at {path} is not a dict, skipping.")
            except Exception as exc:
                logger.error(f"Error loading LLM config from {label} at {path}: {exc}", exc_info=True)
        elif "ENV_LLM" in label:
             logger.warning(f"{label} not found at {path}. Using defaults or previously loaded LLM config.")
    
    # Expand environment variables in the LLM config part
    if llm_config_from_yaml:
        logger.debug(f"LLM config BEFORE env var expansion: {json.dumps(llm_config_from_yaml, indent=2, default=str)}")
        llm_config_from_yaml = _expand_env_vars_in_dict(llm_config_from_yaml) # Applied to the LLM part
        logger.info(f"LLM config after env var expansion completed")
        logger.debug(f"LLM config AFTER env var expansion: {json.dumps(llm_config_from_yaml, indent=2, default=str)}")
    else:
        logger.warning("llm_config_from_yaml is empty before environment variable expansion step.")


    # V4: Basic validation of LLM config structure
    if not llm_config_from_yaml:
        # If global_app_config *itself* defined an llm section, use that, else error.
        if 'llm' in config and config['llm']:
            logger.info("Using 'llm' section defined directly in global_app_config.yaml as llm_config.yaml was not found or empty.")
            config['llm'] = _expand_env_vars_in_dict(config['llm']) # Expand here if taken from global
        else:
            raise RuntimeError(f"LLM configuration section is completely empty after attempting to load files for env '{env}'. Cannot proceed.")
    elif 'llm' not in config or not config['llm']: # if llm_config.yaml was loaded, but global_app_config has no llm section
        config['llm'] = llm_config_from_yaml # Assign loaded llm_config to the main config
    else: # Both global_app_config.llm and llm_config.yaml have content, merge them
        config['llm'] = ConfigMerger.merge(config['llm'], llm_config_from_yaml, "global_llm_with_specific_llm_config")
        config['llm'] = _expand_env_vars_in_dict(config['llm']) # Ensure expansion on final merged LLM config

    # Final check on the structure of config['llm']
    if not isinstance(config.get('llm'), dict) or \
       not isinstance(config['llm'].get('models'), dict) or \
       not config['llm']['models']:
        raise RuntimeError(
            f"Missing or empty 'models' dictionary in the final LLM configuration for env '{env}'. "
            "Valid LLM configuration with defined models is required."
        )
    
    # Validate default model existence
    default_model_key = config['llm'].get('default')
    if not default_model_key:
        raise RuntimeError(f"No 'default' LLM model key specified in LLM configuration for env '{env}'.")
    
    # Check if default key is in models or is a valid route
    if default_model_key not in config['llm']['models']:
        is_valid_route_to_model = False
        if 'routes' in config['llm'] and isinstance(config['llm']['routes'], dict):
            model_key_from_route = config['llm']['routes'].get(default_model_key)
            if model_key_from_route and model_key_from_route in config['llm']['models']:
                is_valid_route_to_model = True
        if not is_valid_route_to_model:
            raise RuntimeError(
                f"Default LLM key '{default_model_key}' (from 'llm.default') not found in 'llm.models' keys "
                f"or as a valid 'llm.routes' entry for env '{env}'. "
                f"Available models: {list(config['llm']['models'].keys())}"
            )
            
    logger.info(f"Config loaded for env='{env}'. Top-level keys: {list(config.keys())}")
    logger.info(f"LLM configuration active. Default model key: '{config['llm'].get('default', 'N/A')}'")
    return config
```

**`nireon_v4/core/__init__.py`** (empty)
**`nireon_v4/core/registry/__init__.py`**
```python
# Nireon V4 Core Registry Package
from .component_registry import ComponentRegistry

__all__ = ["ComponentRegistry"]
```

**`nireon_v4/core/registry/component_registry.py`**
```python
# V4: ComponentRegistry class, adapted from V3's application.components.lifecycle
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Set, Union, Type

# V4: Import ComponentMetadata and ComponentRegistryMissingError from their new V4 location
from nireon_v4.application.components.lifecycle import ComponentMetadata, ComponentRegistryMissingError

__all__ = ['ComponentRegistry'] # V4: Only ComponentRegistry is defined here

logger = logging.getLogger(__name__)

class ComponentRegistry:
    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}
        self._metadata: Dict[str, ComponentMetadata] = {}
        self._certifications: Dict[str, Dict[str, Any]] = {} # V4: Retain from V3
        self._lock = threading.RLock() # V4: Use RLock for reentrancy
        logger.debug("ComponentRegistry initialized with thread safety")

    def normalize_key(self, key: Union[str, Type, object]) -> str: # V4: Public method
        original_key_repr = repr(key)
        if isinstance(key, str):
            normalized = key
        elif isinstance(key, type): # V4: Robust type naming
            normalized = f"{key.__module__}.{key.__qualname__}"
        elif hasattr(key, '__name__'): # Fallback for other callables
            normalized = getattr(key, '__name__')
        else:
            normalized = str(key)
        logger.debug(f"[V4.ComponentRegistry.normalize_key] Original key: {original_key_repr}, Type: {type(key)}, Normalized to: '{normalized}'")
        return normalized
    
    def _warn_re_register(self, key: str, kind: str) -> None: # V4: Internal helper
        logger.warning(f"[V4.ComponentRegistry] {kind} with key '{key}' is being re-registered.")

    def is_service_registered(self, key: Union[str, Type]) -> bool:
        normalized_key = self.normalize_key(key)
        return normalized_key in self._components

    def register(self, component_value: Any, metadata: ComponentMetadata) -> None:
        if not isinstance(metadata, ComponentMetadata):
            raise TypeError("metadata must be an instance of ComponentMetadata")
        
        registration_key = metadata.id # Use metadata.id as the primary key
        with self._lock:
            if registration_key in self._components:
                existing_instance = self._components[registration_key]
                if existing_instance is component_value:
                    logger.debug(f"Component '{registration_key}' already registered with the exact same instance. Skipping re-registration.")
                    return
                else: # V4: Detailed logging for re-registration
                    logger.debug(
                        f"[V4.ComponentRegistry] Component '{registration_key}' is being re-registered with a new instance "
                        f"(Old: {type(existing_instance)}, New: {type(component_value)}). "
                        "This might be expected during multi-phase bootstrap."
                    )
            self._components[registration_key] = component_value
            self._metadata[registration_key] = metadata # Store metadata by the same ID
            logger.info(
                f"Component '{registration_key}' registered (or re-registered) successfully "
                f"(category: {metadata.category}, epistemic_tags: {metadata.epistemic_tags})"
            )

    def register_service_instance(self, key: Union[str, Type], instance: Any) -> None:
        normalized_key = self.normalize_key(key)
        if not normalized_key.strip(): # V4: Guard against empty keys
            raise ValueError("Service key cannot resolve to empty or whitespace")
            
        with self._lock:
            if normalized_key in self._components:
                self._warn_re_register(normalized_key, "Service")
            self._components[normalized_key] = instance
            # V4: Metadata for service instances (not full components) might be handled by _safe_register_service_instance
            # or could be added here if a convention is established.
            logger.debug(f"Service instance registered: {key} -> '{normalized_key}'")

    def get_service_instance(self, key: Union[str, Type]) -> Any:
        normalized_key = self.normalize_key(key)
        logger.debug(f"[V4.ComponentRegistry.get_service_instance] Attempting to get service with normalized key: '{normalized_key}'.")
        # V4: Improved debug logging
        logger.debug(f"[V4.ComponentRegistry.get_service_instance] Available keys in _components: {sorted(list(self._components.keys()))}")
        if normalized_key not in self._components:
            available_keys = sorted(self._components.keys()) # Get keys at the time of error
            raise ComponentRegistryMissingError(
                str(key), 
                message=f"Service '{key}' (normalized: '{normalized_key}') not found in V4 registry. Available components: {available_keys}"
            )
        return self._components[normalized_key]

    def get(self, key: Union[str, Type]) -> Any: # General get, can be by ID or type
        normalized_key = self.normalize_key(key)
        if normalized_key not in self._components:
            raise ComponentRegistryMissingError(normalized_key)
        return self._components[normalized_key]

    def get_metadata(self, component_id: str) -> ComponentMetadata:
        if component_id in self._metadata:
            return self._metadata[component_id]
        
        # Fallback: if component exists but metadata wasn't directly put in _metadata
        # (e.g. registered via register_service_instance without explicit metadata store step)
        if component_id in self._components:
            comp = self._components[component_id]
            if hasattr(comp, 'metadata') and isinstance(comp.metadata, ComponentMetadata):
                with self._lock: # Cache it if found this way
                    self._metadata[component_id] = comp.metadata
                logger.debug(f"Cached metadata for component '{component_id}' from instance")
                return comp.metadata
        
        available_metadata = sorted(self._metadata.keys())
        raise ComponentRegistryMissingError(
            component_id, 
            message=f"Metadata for component '{component_id}' not found in V4 registry. Components with metadata: {available_metadata}"
        )

    def get_certification(self, component_id: str) -> Dict[str, Any]: # V4: Retain from V3
        return self._certifications.get(component_id, {})

    def register_certification(self, component_id: str, cert: Dict[str, Any]) -> None: # V4: Retain
        if not isinstance(cert, dict):
            raise TypeError("Certification data must be a dictionary")
        with self._lock:
            if component_id not in self._components: # V4: More nuanced logging
                logger.debug(f"Registering certification for '{component_id}' which is not (yet) a fully registered component. This may be acceptable during bootstrap.")
            self._certifications[component_id] = cert
            logger.debug(f"Certification registered for '{component_id}'")

    def list_components(self) -> List[str]:
        return list(self._components.keys())

    def list_service_instances(self) -> List[str]: # V4: Alias or distinct if needed
        return list(self._components.keys())
    
    # V4: Retain these useful find methods from V3
    def find_by_category(self, category: str) -> List[str]:
        return [cid for cid, md in self._metadata.items() if md.category == category]

    def find_by_capability(self, capability: str) -> List[str]:
        return [cid for cid, md in self._metadata.items() if capability in md.capabilities]

    def find_by_epistemic_tag(self, tag: str) -> List[str]:
        return [cid for cid, md in self._metadata.items() if tag in md.epistemic_tags]

    def get_stats(self) -> Dict[str, Any]: # V4: Retain from V3
        certified_count = len([cid for cid in self._certifications if self._certifications[cid]])
        categories: Dict[str, int] = {}
        epistemic_tags: Dict[str, int] = {}
        for metadata in self._metadata.values():
            categories[metadata.category] = categories.get(metadata.category, 0) + 1
            for tag in metadata.epistemic_tags:
                epistemic_tags[tag] = epistemic_tags.get(tag, 0) + 1
        
        return {
            "total_components": len(self._components),
            "components_with_metadata": len(self._metadata),
            "certified_components": certified_count,
            "categories": categories,
            "epistemic_tags": epistemic_tags,
        }

```

**`nireon_v4/domain/__init__.py`** (empty)
**`nireon_v4/domain/embeddings/__init__.py`** (empty)
**`nireon_v4/domain/embeddings/vector.py`** (Copy from V3)
**`nireon_v4/domain/ideas/__init__.py`** (empty)
**`nireon_v4/domain/ideas/idea.py`** (Copy from V3)

**`nireon_v4/infrastructure/__init__.py`** (empty)
**`nireon_v4/infrastructure/feature_flags.py`**
```python
# Adapted from nireon_staging/nireon/infrastructure/feature_flags.py
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

class FeatureFlagsManager:
    def __init__(self, config: Dict[str, Any] = None):
        self._flags: Dict[str, bool] = {}
        self._descriptions: Dict[str, str] = {}
        self._registered_flags: Set[str] = set() # Keep track of explicitly registered flags

        if config:
            for flag_name, value in config.items():
                if isinstance(value, bool):
                    self._flags[flag_name] = value
                elif isinstance(value, dict) and 'enabled' in value: # V3: Support dict config
                    self._flags[flag_name] = bool(value['enabled'])
                    if 'description' in value:
                        self._descriptions[flag_name] = str(value['description'])
        
        # V4: Register some common flags expected by Nireon components or bootstrap itself
        # These act as defaults if not in config.
        self.register_flag("sentinel_enable_progression_adjustment", default_value=False, description="Enables progression bonus in Sentinel mechanism")
        self.register_flag("sentinel_enable_edge_trust", default_value=False, description="Enables edge trust calculations in Sentinel (if Idea supports graph structure)")
        self.register_flag("enable_exploration", default_value=True, description="Enables the Explorer mechanism to generate idea variations")
        self.register_flag("enable_catalyst", default_value=True, description="Enables the Catalyst mechanism for cross-domain idea blending")
        self.register_flag("catalyst_anti_constraints", default_value=False, description="Enables anti-constraint functionality in Catalyst") # From V3 Catalyst
        self.register_flag("catalyst_duplication_check", default_value=False, description="Enables duplication detection and adaptation in Catalyst") # From V3 Catalyst
        
        logger.info(f"FeatureFlagsManager initialized with {len(self._flags)} flags from config and defaults.")

    def register_flag(self, flag_name: str, default_value: bool = False, description: Optional[str] = None) -> None:
        self._registered_flags.add(flag_name)
        if flag_name not in self._flags: # Only set default if not already loaded from config
            self._flags[flag_name] = default_value
        if description:
            self._descriptions[flag_name] = description
        logger.debug(f"Registered feature flag: {flag_name} (default: {default_value})")

    def is_enabled(self, flag_name: str, default: Optional[bool] = None) -> bool:
        if flag_name in self._flags:
            return self._flags[flag_name]
        
        if default is not None: # If a specific default is passed by the caller
            logger.warning(f"Unregistered feature flag: {flag_name}, using provided default: {default}")
            return default
        
        # Fallback if flag is completely unknown and no default provided at call site
        logger.warning(f"Unregistered feature flag: {flag_name}, defaulting to False")
        return False

    def set_flag(self, flag_name: str, value: bool) -> None:
        if flag_name not in self._registered_flags:
            logger.warning(f"Setting unregistered feature flag: {flag_name}")
            self._registered_flags.add(flag_name) # Add it to registered if set at runtime
        self._flags[flag_name] = bool(value) # Ensure it's a boolean
        logger.info(f"Feature flag {flag_name} set to {value}")

    def get_all_flags(self) -> Dict[str, bool]:
        return dict(self._flags) # Return a copy

    def get_flag_description(self, flag_name: str) -> Optional[str]:
        return self._descriptions.get(flag_name)

    def get_registered_flags(self) -> List[Dict[str, Any]]:
        # V4: Provide a structured list of registered flags and their states
        result = []
        for flag_name in sorted(self._registered_flags): # Iterate over explicitly registered ones
            flag_info = {
                "name": flag_name,
                "enabled": self._flags.get(flag_name, False) # Get current value
            }
            if flag_name in self._descriptions:
                flag_info["description"] = self._descriptions[flag_name]
            result.append(flag_info)
        return result

# V4: Global function for registration, similar to V3's style, but likely not primary API.
# The manager instance is preferred.
def register_flag(flag_name: str, default_value: bool = False, description: Optional[str]=None) -> None:
    # This global function would typically interact with a singleton FeatureFlagsManager instance
    # For now, it's more of a placeholder or for contexts where a global manager is assumed.
    # In V4 bootstrap, we create and use an instance directly.
    logger.info(f"Feature flag registration requested: {flag_name} (default: {default_value})")
    logger.info(f"Description: {description}")
```

**`nireon_v4/application/components/results.py`** (Copy from V3, ensure `datetime` is timezone-aware)

```python
# Adapted from nireon_staging/nireon/application/components/results.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone # V4: Ensure timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Health Status Constants
HEALTH_STATUS_HEALTHY = "HEALTHY"
HEALTH_STATUS_DEGRADED = "DEGRADED"
HEALTH_STATUS_UNHEALTHY = "UNHEALTHY"
HEALTH_STATUS_UNKNOWN = "UNKNOWN" # Added from V3
VALID_HEALTH_STATUSES = {HEALTH_STATUS_HEALTHY, HEALTH_STATUS_DEGRADED, HEALTH_STATUS_UNHEALTHY, HEALTH_STATUS_UNKNOWN}


# Priority Constants
PRIORITY_LOW = "low"
PRIORITY_NORMAL = "normal"
PRIORITY_HIGH = "high"
PRIORITY_CRITICAL = "critical"
VALID_PRIORITIES = {PRIORITY_LOW, PRIORITY_NORMAL, PRIORITY_HIGH, PRIORITY_CRITICAL}


@dataclass
class BaseResult:
    success: bool
    component_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc)) # V4: TZ aware
    message: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # V4: Ensure timestamp is datetime and timezone-aware
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            except ValueError as e:
                logger.warning(f"Failed to parse timestamp '{self.timestamp}': {e}. Using current time.")
                self.timestamp = datetime.now(timezone.utc)
        
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

        if not self.success and not self.message and not self.error_code:
            self.message = "An unspecified error occurred."
            
    def is_success(self) -> bool:
        return self.success

    def is_failure(self) -> bool:
        return not self.success


@dataclass
class ProcessResult(BaseResult):
    output_data: Any = None
    metrics: Optional[Dict[str, Any]] = None # V4: Retain from V3

    def has_output(self) -> bool:
        return self.output_data is not None


@dataclass
class AnalysisResult(BaseResult):
    metrics: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    insights: List[str] = field(default_factory=list) # V4: Retain from V3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be a numeric value")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0")
    
    def has_insights(self) -> bool:
        return len(self.insights) > 0


@dataclass
class SystemSignal(BaseResult): # V4: Retain from V3
    signal_type: str = "generic"
    priority: str = PRIORITY_NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.priority not in VALID_PRIORITIES:
            raise ValueError(f"Invalid priority '{self.priority}' for SystemSignal from component '{self.component_id}'. Valid priorities: {sorted(VALID_PRIORITIES)}")

    def is_high_priority(self) -> bool:
        return self.priority in {PRIORITY_HIGH, PRIORITY_CRITICAL}


@dataclass
class ObservationResult(ProcessResult): # V4: Retain from V3
    observation_id: Optional[str] = None
    operation: Optional[str] = None
    stage: Optional[str] = None

    def get_observation_key(self) -> str:
        parts = []
        if self.observation_id: parts.append(f"id:{self.observation_id}")
        if self.operation: parts.append(f"op:{self.operation}")
        if self.stage: parts.append(f"stage:{self.stage}")
        parts.append(f"comp:{self.component_id}")
        return "|".join(parts)


@dataclass
class AdaptationAction(BaseResult): # V4: Retain from V3
    action_type: str = "generic"
    target_component_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    urgency: str = PRIORITY_NORMAL # Renamed from V3 'priority' for clarity if SystemSignal uses 'priority'

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.urgency not in VALID_PRIORITIES: # Check against VALID_PRIORITIES
            raise ValueError(f"Invalid urgency '{self.urgency}' for AdaptationAction from component '{self.component_id}'. Valid urgencies: {sorted(VALID_PRIORITIES)}")

    def is_system_wide(self) -> bool:
        return self.target_component_id is None
    
    def is_urgent(self) -> bool:
        return self.urgency in {PRIORITY_HIGH, PRIORITY_CRITICAL}


@dataclass
class ComponentHealth: # V4: Retain from V3
    component_id: str
    status: str = HEALTH_STATUS_UNKNOWN # Default to UNKNOWN
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: Optional[str] = None
    last_certification: Optional[Dict[str, Any]] = None
    error_count: int = 0
    operational_metrics: Dict[str, Any] = field(default_factory=dict)
    containment_reflection_summary: Optional[Dict[str, Any]] = None # From V3 NireonBaseComponent

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, str): # V4: More robust timestamp parsing
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            except ValueError as e:
                logger.warning(f"Failed to parse health timestamp '{self.timestamp}': {e}. Using current time.")
                self.timestamp = datetime.now(timezone.utc)

        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

        if self.status not in VALID_HEALTH_STATUSES:
            logger.warning(f"Invalid health status '{self.status}' for component '{self.component_id}'. Valid statuses: {sorted(VALID_HEALTH_STATUSES)}. Defaulting to UNKNOWN.")
            self.status = HEALTH_STATUS_UNKNOWN
        
        if self.error_count < 0:
            logger.warning(f"Negative error_count {self.error_count} for component '{self.component_id}'. Setting to 0.")
            self.error_count = 0

    def is_healthy(self) -> bool:
        return self.status == HEALTH_STATUS_HEALTHY

    def is_degraded(self) -> bool:
        return self.status == HEALTH_STATUS_DEGRADED
    
    def is_unhealthy(self) -> bool:
        return self.status == HEALTH_STATUS_UNHEALTHY
    
    def has_errors(self) -> bool:
        return self.error_count > 0
    
    def has_certification(self) -> bool:
        return self.last_certification is not None

    def has_containment_reflection(self) -> bool: # From V3 NireonBaseComponent
        return self.containment_reflection_summary is not None
    
    def get_health_score(self) -> float: # From V3
        base_score = {
            HEALTH_STATUS_HEALTHY: 1.0,
            HEALTH_STATUS_DEGRADED: 0.6,
            HEALTH_STATUS_UNHEALTHY: 0.2,
            HEALTH_STATUS_UNKNOWN: 0.5,
        }.get(self.status, 0.0)
        if self.error_count > 0:
            error_penalty = min(0.3, self.error_count * 0.05) # Diminishing penalty
            base_score = max(0.0, base_score - error_penalty)
        return base_score

```

This set of files establishes the very basic structure for Phase 1. The `bootstrap_nireon_system` function is significantly simplified, focusing only on loading global configuration and setting up core services with placeholders. Manifest processing, component instantiation/initialization loops, and validation are deferred to later phases as per your plan. The V4 `ComponentRegistry` from `core.registry` is used.