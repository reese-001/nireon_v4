"""
Base Phase - Abstract interface for all bootstrap phases.

Defines the common contract and shared functionality for all bootstrap phases
following the L0 Abiogenesis pattern of system emergence with V2 context integration.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bootstrap.exceptions import BootstrapError

logger = logging.getLogger(__name__)

@dataclass
class PhaseResult:
    """Result of a bootstrap phase execution."""
    success: bool
    message: str
    errors: List[str]
    warnings: List[str] 
    metadata: Dict[str, Any]
    
    @classmethod
    def success_result(
        cls, 
        message: str, 
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'PhaseResult':
        """Create a successful phase result."""
        return cls(
            success=True,
            message=message,
            errors=[],
            warnings=warnings or [],
            metadata=metadata or {}
        )
    
    @classmethod
    def failure_result(
        cls, 
        message: str, 
        errors: List[str],
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'PhaseResult':
        """Create a failed phase result."""
        return cls(
            success=False,
            message=message,
            errors=errors,
            warnings=warnings or [],
            metadata=metadata or {}
        )
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to this result."""
        self.warnings.append(warning)
    
    def add_error(self, error: str) -> None:
        """Add an error to this result."""
        self.errors.append(error)
        self.success = False

class BootstrapPhase(ABC):
    """
    Abstract base class for all bootstrap phases.
    
    Each phase represents a distinct step in the L0 Abiogenesis process
    of bringing the NIREON system into existence from configuration.
    Enhanced with V2 context helper integration for better phase isolation and testing.
    """
    
    def __init__(self):
        self.phase_name = self.__class__.__name__
        self.logger = logging.getLogger(f"bootstrap.{self.phase_name.lower()}")
        # UPGRADED: V2 integration tracking
        self._v2_context_support = True
        self._phase_context = None
    
    @abstractmethod
    async def execute(self, context) -> PhaseResult:
        """
        Execute this bootstrap phase.
        
        Args:
            context: BootstrapContext containing shared state
            
        Returns:
            PhaseResult indicating success/failure and any warnings/errors
        """
        pass
    
    async def pre_execute(self, context) -> None:
        """
        Pre-execution hook called before execute().
        
        Enhanced with V2 context helper integration for phase isolation.
        Override this to perform phase-specific setup or validation.
        """
        self.logger.debug(f"Starting phase: {self.phase_name}")
        
        # UPGRADED: Create phase-specific context using V2 context helper
        await self._initialize_phase_context(context)
    
    async def post_execute(self, context, result: PhaseResult) -> None:
        """
        Post-execution hook called after execute().
        
        Enhanced with V2 context integration for cleanup and reporting.
        Override this to perform phase-specific cleanup or reporting.
        """
        if result.success:
            self.logger.info(f"✓ Phase completed: {self.phase_name} - {result.message}")
        else:
            self.logger.error(f"✗ Phase failed: {self.phase_name} - {result.message}")
            for error in result.errors:
                self.logger.error(f"  Error: {error}")
        
        for warning in result.warnings:
            self.logger.warning(f"  Warning: {warning}")
        
        # UPGRADED: Add V2 integration metadata
        if 'v2_integration' not in result.metadata:
            result.metadata['v2_integration'] = self._v2_context_support
            result.metadata['phase_context_created'] = self._phase_context is not None
    
    def validate_context(self, context) -> None:
        """
        Validate that context contains required elements for this phase.
        
        Enhanced with V2 context validation.
        Override this to add phase-specific context validation.
        Raises ValueError if context is invalid.
        """
        required_attrs = ['config', 'run_id', 'registry', 'health_reporter']
        for attr in required_attrs:
            if not hasattr(context, attr):
                raise ValueError(f"BootstrapContext missing required attribute: {attr}")
        
        # UPGRADED: Validate V2 context helper integration
        self._validate_v2_context_integration(context)
    
    def should_skip_phase(self, context) -> tuple[bool, str]:
        """
        Determine if this phase should be skipped.
        
        Enhanced with V2 context-aware skip conditions.
        
        Returns:
            (should_skip, reason) tuple
        """
        # UPGRADED: Check for V2-specific skip conditions
        if hasattr(context, 'feature_flags') and context.feature_flags:
            skip_flag = context.feature_flags.get(f'skip_{self.phase_name.lower()}', False)
            if skip_flag:
                return True, f"Phase skipped by feature flag: skip_{self.phase_name.lower()}"
        
        return False, ""
    
    async def execute_with_hooks(self, context) -> PhaseResult:
        """
        Execute phase with pre/post hooks and error handling.
        
        Enhanced with V2 context integration and improved error handling.
        This is the main entry point called by the orchestrator.
        """
        try:
            # Validate context
            self.validate_context(context)
            
            # Check if phase should be skipped
            should_skip, skip_reason = self.should_skip_phase(context)
            if should_skip:
                self.logger.info(f"Skipping phase {self.phase_name}: {skip_reason}")
                return PhaseResult.success_result(
                    message=f"Phase skipped: {skip_reason}",
                    metadata={
                        'skipped': True, 
                        'skip_reason': skip_reason,
                        'v2_integration': self._v2_context_support
                    }
                )
            
            # Pre-execution hook
            await self.pre_execute(context)
            
            # Main execution
            result = await self.execute(context)
            
            # Post-execution hook
            await self.post_execute(context, result)
            
            # Update health reporter
            self._update_health_reporter(context, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error in phase {self.phase_name}: {e}"
            self.logger.error(error_msg, exc_info=True)
            
            result = PhaseResult.failure_result(
                message=f"Phase {self.phase_name} failed with exception",
                errors=[error_msg],
                metadata={
                    'exception': str(e), 
                    'phase_name': self.phase_name,
                    'v2_integration': self._v2_context_support
                }
            )
            
            # Still call post-execute for cleanup
            try:
                await self.post_execute(context, result)
            except Exception:
                self.logger.error("Error in post_execute cleanup", exc_info=True)
            
            self._update_health_reporter(context, result)
            return result
    
    def _update_health_reporter(self, context, result: PhaseResult) -> None:
        """
        Update health reporter with phase results.
        
        Enhanced with V2 context integration metadata.
        """
        try:
            phase_status = "completed" if result.success else "failed"
            
            # UPGRADED: Add V2 integration info to health metadata
            health_metadata = result.metadata.copy()
            health_metadata.update({
                'v2_context_support': self._v2_context_support,
                'phase_context_available': self._phase_context is not None
            })
            
            # Add phase result to health reporter
            # Note: This assumes health reporter has a method to track phases
            if hasattr(context.health_reporter, 'add_phase_result'):
                context.health_reporter.add_phase_result(
                    phase_name=self.phase_name,
                    status=phase_status,
                    message=result.message,
                    errors=result.errors,
                    warnings=result.warnings,
                    metadata=health_metadata
                )
        except Exception as e:
            self.logger.warning(f"Failed to update health reporter: {e}")

    async def _initialize_phase_context(self, context) -> None:
        """
        UPGRADED: Initialize phase-specific context using V2 context helper.
        
        Creates an isolated context for this phase using the new context helper utilities.
        """
        try:
            from bootstrap.bootstrap_helper.context_helper import create_context_builder
            
            # Create phase-specific context for isolation
            builder = create_context_builder(
                component_id=f"phase_{self.phase_name.lower()}",
                run_id=f"{context.run_id}_phase"
            )
            
            # Configure builder with registry and event bus from main context
            if hasattr(context, 'registry'):
                builder.with_registry(context.registry)
            
            if hasattr(context, 'event_bus'):
                builder.with_event_bus(context.event_bus)
            
            # Add phase-specific metadata
            builder.with_metadata(
                phase_name=self.phase_name,
                phase_isolation=True,
                v2_context_helper=True
            )
            
            # Add feature flags if available
            if hasattr(context, 'feature_flags') and context.feature_flags:
                builder.with_feature_flags(context.feature_flags)
            
            # Build the phase context
            self._phase_context = builder.build()
            
            self.logger.debug(f"Phase context initialized for {self.phase_name} using V2 context helper")
            
        except ImportError:
            self.logger.debug(f"V2 context helper not available, skipping phase context creation")
            self._v2_context_support = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize phase context: {e}")
            self._v2_context_support = False

    def _validate_v2_context_integration(self, context) -> None:
        """
        UPGRADED: Validate V2 context helper integration.
        
        Checks if the context has the enhanced capabilities expected from V2.
        """
        try:
            # Check for V2 context helper features
            v2_features = [
                'with_component_scope',  # V2 context scoping
                'with_metadata',         # Enhanced metadata support
            ]
            
            missing_features = []
            for feature in v2_features:
                if not hasattr(context, feature):
                    missing_features.append(feature)
            
            if missing_features:
                self.logger.debug(f"Context missing V2 features: {missing_features}")
                self._v2_context_support = False
            else:
                self.logger.debug("V2 context integration validated successfully")
                
        except Exception as e:
            self.logger.debug(f"V2 context validation failed: {e}")
            self._v2_context_support = False

    def get_phase_context(self):
        """
        UPGRADED: Get the phase-specific context created by V2 context helper.
        
        Returns the isolated context for this phase, or None if not available.
        """
        return self._phase_context

    def supports_v2_context(self) -> bool:
        """
        UPGRADED: Check if this phase supports V2 context integration.
        
        Returns True if V2 context features are available and working.
        """
        return self._v2_context_support

class ErrorHandlingMixin:
    """
    Mixin providing common error handling utilities for phases.
    
    Enhanced with V2 context integration for better error reporting.
    """
    
    def handle_strict_mode_error(self, context, error_msg: str, exception: Optional[Exception] = None) -> None:
        """
        Handle errors according to strict mode setting.
        
        Enhanced with V2 context metadata for better error tracking.
        """
        # UPGRADED: Add V2 context info to error
        if hasattr(self, '_v2_context_support') and self._v2_context_support:
            error_msg = f"{error_msg} [V2 Context: Supported]"
        
        if context.strict_mode:
            if exception:
                raise BootstrapError(error_msg) from exception
            else:
                raise BootstrapError(error_msg)
        else:
            self.logger.warning(f"Non-strict mode: {error_msg}")
    
    def collect_errors(self, errors: List[str], new_error: str, context) -> None:
        """
        Collect error and handle according to strict mode.
        
        Enhanced with V2 context integration information.
        """
        # UPGRADED: Enhance error with V2 context info if available
        if hasattr(self, '_v2_context_support') and self._v2_context_support:
            enhanced_error = f"{new_error} [Phase Context: {'Available' if hasattr(self, '_phase_context') and self._phase_context else 'Not Available'}]"
            errors.append(enhanced_error)
        else:
            errors.append(new_error)
            
        if context.strict_mode:
            # In strict mode, we might want to fail fast
            # But for now, collect errors and let phase decide
            pass