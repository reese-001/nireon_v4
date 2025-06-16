"""
Bootstrap Phase Executor - Consistent phase execution with proper error handling.

This module provides a standardized way to execute bootstrap phases with
consistent error handling, logging, and health reporting.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from bootstrap.phases.base_phase import BootstrapPhase, PhaseResult
    from context.bootstrap_context import BootstrapContext

logger = logging.getLogger(__name__)


@dataclass
class PhaseExecutionResult:
    """Result of executing a bootstrap phase."""
    phase_name: str
    success: bool
    duration_seconds: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Exception] = None
    
    @property
    def is_critical_failure(self) -> bool:
        """Check if this represents a critical failure."""
        return not self.success and self.exception is not None


@dataclass
class PhaseExecutionSummary:
    """Summary of all phase executions."""
    total_phases: int
    successful_phases: int
    failed_phases: int
    total_duration: float
    results: List[PhaseExecutionResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_phases == 0:
            return 100.0
        return (self.successful_phases / self.total_phases) * 100.0
    
    @property
    def has_critical_failures(self) -> bool:
        """Check if any phase had critical failures."""
        return any(result.is_critical_failure for result in self.results)


class BootstrapPhaseExecutor:
    """
    Executes bootstrap phases with consistent error handling and reporting.
    
    Provides standardized execution flow, timing, error collection,
    and health reporting for all bootstrap phases.
    """
    
    def __init__(self, context: BootstrapContext):
        self.context = context
        self.execution_results: List[PhaseExecutionResult] = []
        
    async def execute_phases(self, phases: List[BootstrapPhase]) -> PhaseExecutionSummary:
        """
        Execute all phases with consistent error handling.
        
        Args:
            phases: List of phases to execute in order
            
        Returns:
            Summary of all phase executions
        """
        logger.info(f'Executing {len(phases)} bootstrap phases')
        start_time = datetime.now(timezone.utc)
        
        successful_count = 0
        
        for i, phase in enumerate(phases, 1):
            phase_name = phase.__class__.__name__
            logger.info(f'Phase {i}/{len(phases)}: {phase_name}')
            
            result = await self._execute_single_phase(phase, i, len(phases))
            self.execution_results.append(result)
            
            if result.success:
                successful_count += 1
                logger.info(f'✓ Phase {phase_name} completed in {result.duration_seconds:.2f}s')
            else:
                logger.error(f'✗ Phase {phase_name} failed after {result.duration_seconds:.2f}s')
                
                # Handle failure based on strict mode
                if self.context.strict_mode and result.is_critical_failure:
                    logger.error(f'Critical failure in strict mode - stopping bootstrap')
                    break
                elif not self.context.strict_mode:
                    logger.warning(f'Phase failed but continuing in non-strict mode')
        
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - start_time).total_seconds()
        
        summary = PhaseExecutionSummary(
            total_phases=len(phases),
            successful_phases=successful_count,
            failed_phases=len(phases) - successful_count,
            total_duration=total_duration,
            results=self.execution_results.copy()
        )
        
        self._log_execution_summary(summary)
        await self._update_health_reporter(summary)
        
        return summary
    
    async def _execute_single_phase(
        self, 
        phase: BootstrapPhase, 
        phase_number: int, 
        total_phases: int
    ) -> PhaseExecutionResult:
        """Execute a single phase with timing and error handling."""
        phase_name = phase.__class__.__name__
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check if phase should be skipped
            should_skip, skip_reason = phase.should_skip_phase(self.context)
            if should_skip:
                logger.info(f'Skipping phase {phase_name}: {skip_reason}')
                return PhaseExecutionResult(
                    phase_name=phase_name,
                    success=True,
                    duration_seconds=0.0,
                    metadata={'skipped': True, 'skip_reason': skip_reason}
                )
            
            # Execute phase with hooks
            logger.debug(f'Starting phase {phase_name}')
            await phase.pre_execute(self.context)
            
            phase_result = await phase.execute(self.context)
            
            await phase.post_execute(self.context, phase_result)
            
            # Calculate duration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            # Update health reporter
            self._update_phase_health_status(phase_name, phase_result)
            
            return PhaseExecutionResult(
                phase_name=phase_name,
                success=phase_result.success,
                duration_seconds=duration,
                errors=phase_result.errors.copy(),
                warnings=phase_result.warnings.copy(),
                metadata=phase_result.metadata.copy()
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            error_msg = f'Unexpected exception in phase {phase_name}: {e}'
            logger.error(error_msg, exc_info=True)
            
            # Update health reporter for exception
            self._update_phase_health_status(phase_name, None, exception=e)
            
            return PhaseExecutionResult(
                phase_name=phase_name,
                success=False,
                duration_seconds=duration,
                errors=[error_msg],
                exception=e,
                metadata={'exception_type': type(e).__name__}
            )
    
    def _update_phase_health_status(
        self, 
        phase_name: str, 
        phase_result: Optional[PhaseResult], 
        exception: Optional[Exception] = None
    ) -> None:
        """Update health reporter with phase status."""
        try:
            if not hasattr(self.context, 'health_reporter'):
                return
            
            if exception:
                status = 'failed'
                message = f'Phase failed with exception: {exception}'
                errors = [str(exception)]
                warnings = []
            elif phase_result:
                status = 'completed' if phase_result.success else 'failed'
                message = phase_result.message
                errors = phase_result.errors
                warnings = phase_result.warnings
            else:
                status = 'unknown'
                message = 'Phase status unknown'
                errors = []
                warnings = []
            
            self.context.health_reporter.add_phase_result(
                phase_name=phase_name,
                status=status,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata={'execution_context': 'BootstrapPhaseExecutor'}
            )
            
        except Exception as e:
            logger.warning(f'Failed to update health reporter for phase {phase_name}: {e}')
    
    def _log_execution_summary(self, summary: PhaseExecutionSummary) -> None:
        """Log a summary of phase execution."""
        logger.info('=== Bootstrap Phase Execution Summary ===')
        logger.info(f'Total phases: {summary.total_phases}')
        logger.info(f'Successful: {summary.successful_phases}')
        logger.info(f'Failed: {summary.failed_phases}')
        logger.info(f'Success rate: {summary.success_rate:.1f}%')
        logger.info(f'Total duration: {summary.total_duration:.2f}s')
        
        if summary.failed_phases > 0:
            logger.warning('Failed phases:')
            for result in summary.results:
                if not result.success:
                    logger.warning(f'  - {result.phase_name}: {result.errors}')
        
        logger.info('=== End Bootstrap Phase Summary ===')
    
    async def _update_health_reporter(self, summary: PhaseExecutionSummary) -> None:
        """Update health reporter with execution summary."""
        try:
            if not hasattr(self.context, 'health_reporter'):
                return
            
            summary_data = {
                'total_phases': summary.total_phases,
                'successful_phases': summary.successful_phases,
                'failed_phases': summary.failed_phases,
                'success_rate': summary.success_rate,
                'total_duration': summary.total_duration,
                'has_critical_failures': summary.has_critical_failures
            }
            
            overall_status = 'completed' if summary.failed_phases == 0 else 'failed'
            message = f'Bootstrap phases: {summary.successful_phases}/{summary.total_phases} successful'
            
            self.context.health_reporter.add_phase_result(
                phase_name='OverallBootstrap',
                status=overall_status,
                message=message,
                errors=[],
                warnings=[],
                metadata=summary_data
            )
            
        except Exception as e:
            logger.warning(f'Failed to update health reporter with summary: {e}')
    
    def get_phase_metrics(self) -> Dict[str, Any]:
        """Get metrics about phase execution."""
        if not self.execution_results:
            return {}
        
        successful_durations = [r.duration_seconds for r in self.execution_results if r.success]
        failed_durations = [r.duration_seconds for r in self.execution_results if not r.success]
        
        return {
            'total_phases': len(self.execution_results),
            'successful_phases': len(successful_durations),
            'failed_phases': len(failed_durations),
            'total_duration': sum(r.duration_seconds for r in self.execution_results),
            'average_phase_duration': sum(r.duration_seconds for r in self.execution_results) / len(self.execution_results),
            'longest_phase': max(self.execution_results, key=lambda r: r.duration_seconds).phase_name if self.execution_results else None,
            'phase_durations': {r.phase_name: r.duration_seconds for r in self.execution_results}
        }


# Backwards compatibility function
async def execute_bootstrap_phases(
    phases: List[BootstrapPhase], 
    context: BootstrapContext
) -> PhaseExecutionSummary:
    """
    Execute bootstrap phases with standardized error handling.
    
    Args:
        phases: List of phases to execute
        context: Bootstrap context
        
    Returns:
        Summary of phase execution
    """
    executor = BootstrapPhaseExecutor(context)
    return await executor.execute_phases(phases)