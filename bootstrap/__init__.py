"""
bootstrap package – public re-exports + compatibility shims
"""

from __future__ import annotations
import importlib
import sys

# ---------------------------------------------------------------------------
# 1.  Compatibility shim for dynamic phase imports
#     The bootstrap loader will try things like
#        importlib.import_module("phases.manifest_processing_phase")
#     so we alias "phases" → "bootstrap.phases"
# ---------------------------------------------------------------------------

# Alias the package itself
sys.modules.setdefault("phases", importlib.import_module("bootstrap.phases"))

# Alias each concrete phase module that might be requested
for _mod in (
    "manifest_processing_phase",
    "component_initialization_phase",
    "component_validation_phase",
    "rbac_setup_phase",
    "late_rebinding_phase",          # if used
):
    full_name = f"bootstrap.phases.{_mod}"
    sys.modules.setdefault(f"phases.{_mod}", importlib.import_module(full_name))

# ---------------------------------------------------------------------------
# 2.  Re-export public objects from the new package layout
# ---------------------------------------------------------------------------

from .exceptions import *  # noqa: F401,F403

# Core orchestrator & helpers
from .core.main import (                # noqa: E402
    BootstrapOrchestrator,
    bootstrap_nireon_system,
    bootstrap,
    bootstrap_sync,
)
from .core.phase_executor import (      # noqa: E402
    BootstrapPhaseExecutor,
    PhaseExecutionResult,
    PhaseExecutionSummary,
    execute_bootstrap_phases,
)
from .context.bootstrap_context_builder import (  # noqa: E402
    BootstrapContextBuilder,
    create_bootstrap_context,
)
from .context.bootstrap_context import BootstrapContext  # noqa: E402
from .config.bootstrap_config import BootstrapConfig     # noqa: E402

# Result + health
from .result_builder import (           # noqa: E402
    BootstrapResult,
    BootstrapResultBuilder,
    build_result_from_context,
    create_minimal_result,
)
from .validation_data import (          # noqa: E402
    BootstrapValidationData,
    ComponentValidationData,
)
from .health.reporter import (          # noqa: E402
    HealthReporter,
    ComponentStatus,
    ComponentHealthRecord,
)

# Global config loader (unchanged path)
from configs.config_loader import ConfigLoader  # noqa: E402

# ---------------------------------------------------------------------------
# 3.  Metadata
# ---------------------------------------------------------------------------

__version__ = "4.0.0"
__author__ = "NIREON V4 Bootstrap Team"
__description__ = "L0 Abiogenesis – Bootstrap Infrastructure"
CURRENT_SCHEMA_VERSION = "V4-alpha.1.0"

__all__ = [
    # Entry points
    "bootstrap_nireon_system", "bootstrap", "bootstrap_sync",
    # Config / context
    "BootstrapConfig", "BootstrapContext",
    "BootstrapContextBuilder", "create_bootstrap_context",
    # Orchestration
    "BootstrapOrchestrator", "BootstrapPhaseExecutor",
    "PhaseExecutionResult", "PhaseExecutionSummary", "execute_bootstrap_phases",
    # Results / data
    "BootstrapResult", "BootstrapResultBuilder",
    "build_result_from_context", "create_minimal_result",
    "BootstrapValidationData", "ComponentValidationData",
    # Health
    "HealthReporter", "ComponentStatus", "ComponentHealthRecord",
    # Config loader
    "ConfigLoader",
    # Exceptions – wildcard-imported above
    # Versioning
    "CURRENT_SCHEMA_VERSION", "__version__", "__author__", "__description__",
]
