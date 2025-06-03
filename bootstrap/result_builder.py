"""
Bootstrap result builder for NIREON V4.

This module provides utilities for constructing the final BootstrapResult
from the bootstrap context after all phases have completed.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class BootstrapResult:
    """
    Result of the NIREON V4 bootstrap process.
    
    Contains the initialized component registry, health report, validation data,
    and provides backward compatibility with tuple unpacking.
    """
    
    def __init__(
        self,
        registry,
        health_reporter,
        validation_data,
        run_id: str,
        bootstrap_duration: Optional[float] = None,
        global_config: Optional[Dict[str, Any]] = None
    ):
        self.registry = registry
        self.health_reporter = health_reporter
        self.validation_data = validation_data
        self.run_id = run_id
        self.bootstrap_duration = bootstrap_duration
        self.global_config = global_config or {}
        self.creation_time = datetime.now(timezone.utc)
        
        logger.info(f"BootstrapResult created for run_id: {run_id}")
    
    @property
    def success(self) -> bool:
        """
        Check if bootstrap was successful.
        
        Returns True if no critical failures occurred during bootstrap.
        """
        if hasattr(self.health_reporter, 'has_critical_failures'):
            return self.health_reporter.has_critical_failures() == 0
        return True
    
    @property
    def component_count(self) -> int:
        """Get the total number of registered components."""
        try:
            return len(self.registry.list_components())
        except Exception:
            return 0
    
    @property
    def healthy_component_count(self) -> int:
        """Get the number of healthy components."""
        if hasattr(self.health_reporter, 'get_healthy_component_count'):
            return self.health_reporter.get_healthy_component_count()
        return self.component_count
    
    @property
    def critical_failure_count(self) -> int:
        """Get the number of components with critical failures."""
        if hasattr(self.health_reporter, 'has_critical_failures'):
            return self.health_reporter.has_critical_failures()
        return 0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the bootstrap result.
        
        Returns:
            Dictionary with key bootstrap metrics
        """
        return {
            'run_id': self.run_id,
            'success': self.success,
            'component_count': self.component_count,
            'healthy_components': self.healthy_component_count,
            'critical_failures': self.critical_failure_count,
            'bootstrap_duration': self.bootstrap_duration,
            'creation_time': self.creation_time.isoformat(),
            'strict_mode': self.global_config.get('bootstrap_strict_mode', True)
        }
    
    def get_health_report(self) -> str:
        """Get the full health report as a string."""
        if hasattr(self.health_reporter, 'generate_summary'):
            return self.health_reporter.generate_summary()
        return "Health report not available"
    
    def get_component_registry(self):
        """Get the component registry (for backward compatibility)."""
        return self.registry
    
    def get_validation_data(self):
        """Get the validation data store."""
        return self.validation_data
    
    # Backward compatibility: support tuple unpacking
    def __iter__(self):
        """Support tuple unpacking: registry, health_reporter, validation_data = result"""
        import warnings
        warnings.warn(
            "Tuple unpacking of BootstrapResult is deprecated. Use .registry, .health_reporter, .validation_data properties.",
            DeprecationWarning,
            stacklevel=2
        )
        return iter([self.registry, self.health_reporter, self.validation_data])
    
    def __getitem__(self, index):
        """Support indexing for backward compatibility."""
        import warnings
        warnings.warn(
            "Index access to BootstrapResult is deprecated. Use .registry, .health_reporter, .validation_data properties.",
            DeprecationWarning,
            stacklevel=2
        )
        items = [self.registry, self.health_reporter, self.validation_data]
        return items[index]
    
    def __len__(self):
        """Support len() for backward compatibility."""
        return 3
    
    def __repr__(self) -> str:
        return (
            f"BootstrapResult(run_id='{self.run_id}', success={self.success}, "
            f"components={self.component_count}, healthy={self.healthy_component_count})"
        )


class BootstrapResultBuilder:
    """
    Builder for constructing BootstrapResult from bootstrap context.
    
    Extracts all necessary information from the context and packages
    it into a structured result object.
    """
    
    def __init__(self, context):
        self.context = context
        logger.debug(f"BootstrapResultBuilder initialized for run_id: {context.run_id}")
    
    def build(self) -> BootstrapResult:
        """
        Build the final BootstrapResult from context.
        
        Returns:
            Complete BootstrapResult with all bootstrap information
        """
        logger.info(f"Building BootstrapResult for run_id: {self.context.run_id}")
        
        # Calculate bootstrap duration if health reporter tracks it
        bootstrap_duration = None
        if hasattr(self.context.health_reporter, 'get_bootstrap_duration'):
            bootstrap_duration = self.context.health_reporter.get_bootstrap_duration()
        
        # Mark bootstrap as complete in health reporter
        if hasattr(self.context.health_reporter, 'mark_bootstrap_complete'):
            self.context.health_reporter.mark_bootstrap_complete()
        
        result = BootstrapResult(
            registry=self.context.registry,
            health_reporter=self.context.health_reporter,
            validation_data=self.context.validation_data_store,
            run_id=self.context.run_id,
            bootstrap_duration=bootstrap_duration,
            global_config=self.context.global_app_config
        )
        
        # Log summary
        summary = result.get_summary()
        logger.info(f"Bootstrap completed: {summary}")
        
        if not result.success:
            logger.warning(
                f"Bootstrap completed with {result.critical_failure_count} critical failures. "
                f"See health report for details."
            )
        else:
            logger.info(
                f"Bootstrap successful: {result.healthy_component_count}/{result.component_count} "
                f"components healthy"
            )
        
        return result
    
    def build_with_validation_summary(self) -> BootstrapResult:
        """
        Build result and log detailed validation summary.
        
        Returns:
            BootstrapResult with validation summary logged
        """
        result = self.build()
        
        # Log detailed validation information
        if hasattr(self.context.validation_data_store, 'get_validation_summary'):
            validation_summary = self.context.validation_data_store.get_validation_summary()
            logger.info(f"Validation summary: {validation_summary}")
        
        # Log health report if there are issues
        if not result.success:
            health_report = result.get_health_report()
            logger.warning(f"Health report:\n{health_report}")
        
        return result


# Convenience functions for common result building patterns
def build_result_from_context(context) -> BootstrapResult:
    """
    Build a BootstrapResult from a bootstrap context.
    
    Args:
        context: Bootstrap context with registry, health reporter, etc.
        
    Returns:
        Complete BootstrapResult
    """
    builder = BootstrapResultBuilder(context)
    return builder.build()


def build_result_with_summary(context) -> BootstrapResult:
    """
    Build a BootstrapResult with detailed logging.
    
    Args:
        context: Bootstrap context
        
    Returns:
        BootstrapResult with summary information logged
    """
    builder = BootstrapResultBuilder(context)
    return builder.build_with_validation_summary()


def create_minimal_result(registry, run_id: str = "minimal_run") -> BootstrapResult:
    """
    Create a minimal BootstrapResult for testing.
    
    Args:
        registry: Component registry
        run_id: Run identifier
        
    Returns:
        Minimal BootstrapResult
    """
    # Create minimal health reporter
    from bootstrap.bootstrap_helper.health_reporter import BootstrapHealthReporter
    health_reporter = BootstrapHealthReporter(registry)
    
    # Create minimal validation data
    from bootstrap.validation_data import BootstrapValidationData
    validation_data = BootstrapValidationData({})
    
    return BootstrapResult(
        registry=registry,
        health_reporter=health_reporter,
        validation_data=validation_data,
        run_id=run_id,
        global_config={'bootstrap_strict_mode': False}
    )