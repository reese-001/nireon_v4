"""
Base Phase - Abstract interface for all bootstrap phases.

Defines the common contract and shared functionality for all bootstrap phases
following the L0 Abiogenesis pattern of system emergence.
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
    """
    
    def __init__(self):
        self.phase_name = self.__class__.__name__
        self.logger = logging.getLogger(f"bootstrap.{self.phase_name.lower()}")
    
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
        
        Override this to perform phase-specific setup or validation.
        """
        self.logger.debug(f"Starting phase: {self.phase_name}")
    
    async def post_execute(self, context, result: PhaseResult) -> None:
        """
        Post-execution hook called after execute().
        
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
    
    def validate_context(self, context) -> None:
        """
        Validate that context contains required elements for this phase.
        
        Override this to add phase-specific context validation.
        Raises ValueError if context is invalid.
        """
        required_attrs = ['config', 'run_id', 'registry', 'health_reporter']
        for attr in required_attrs:
            if not hasattr(context, attr):
                raise ValueError(f"BootstrapContext missing required attribute: {attr}")
    
    def should_skip_phase(self, context) -> tuple[bool, str]:
        """
        Determine if this phase should be skipped.
        
        Returns:
            (should_skip, reason) tuple
        """
        return False, ""
    
    async def execute_with_hooks(self, context) -> PhaseResult:
        """
        Execute phase with pre/post hooks and error handling.
        
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
                    metadata={'skipped': True, 'skip_reason': skip_reason}
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
                metadata={'exception': str(e), 'phase_name': self.phase_name}
            )
            
            # Still call post-execute for cleanup
            try:
                await self.post_execute(context, result)
            except Exception:
                self.logger.error("Error in post_execute cleanup", exc_info=True)
            
            self._update_health_reporter(context, result)
            return result
    
    def _update_health_reporter(self, context, result: PhaseResult) -> None:
        """Update health reporter with phase results."""
        try:
            phase_status = "completed" if result.success else "failed"
            
            # Add phase result to health reporter
            # Note: This assumes health reporter has a method to track phases
            if hasattr(context.health_reporter, 'add_phase_result'):
                context.health_reporter.add_phase_result(
                    phase_name=self.phase_name,
                    status=phase_status,
                    message=result.message,
                    errors=result.errors,
                    warnings=result.warnings,
                    metadata=result.metadata
                )
        except Exception as e:
            self.logger.warning(f"Failed to update health reporter: {e}")

class ErrorHandlingMixin:
    """Mixin providing common error handling utilities for phases."""
    
    def handle_strict_mode_error(self, context, error_msg: str, exception: Optional[Exception] = None) -> None:
        """Handle errors according to strict mode setting."""
        if context.strict_mode:
            if exception:
                raise BootstrapError(error_msg) from exception
            else:
                raise BootstrapError(error_msg)
        else:
            self.logger.warning(f"Non-strict mode: {error_msg}")
    
    def collect_errors(self, errors: List[str], new_error: str, context) -> None:
        """Collect error and handle according to strict mode."""
        errors.append(new_error)
        if context.strict_mode:
            # In strict mode, we might want to fail fast
            # But for now, collect errors and let phase decide
            pass