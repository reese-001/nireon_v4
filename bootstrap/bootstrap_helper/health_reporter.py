"""
Health reporting system for NIREON V4 bootstrap process.

This module provides comprehensive health tracking and reporting for all
components and phases during the bootstrap process.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from application.components.lifecycle import ComponentMetadata
from core.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """
    Enumeration of possible component status values during bootstrap.
    
    These statuses track the progression of components through the
    bootstrap lifecycle from definition to full operational readiness.
    """
    
    # Definition and configuration errors
    DEFINITION_ERROR = 'DEFINITION_ERROR'
    METADATA_ERROR = 'METADATA_ERROR'
    METADATA_CONSTRUCTION_ERROR = 'METADATA_CONSTRUCTION_ERROR'
    
    # Instantiation errors
    INSTANTIATION_ERROR = 'INSTANTIATION_ERROR'
    
    # Registration states
    INSTANCE_REGISTERED = 'INSTANCE_REGISTERED'
    REGISTRATION_ERROR = 'REGISTRATION_ERROR'
    
    # Bootstrap process errors
    BOOTSTRAP_ERROR = 'BOOTSTRAP_ERROR'
    
    # Initialization states
    INITIALIZED_OK = 'INITIALIZED_OK'
    INITIALIZATION_ERROR = 'INITIALIZATION_ERROR'
    INITIALIZED_SKIPPED_NO_METHOD = 'INITIALIZED_SKIPPED_NO_METHOD'
    INSTANCE_REGISTERED_NO_INIT = 'INSTANCE_REGISTERED_NO_INIT'
    INSTANCE_REGISTERED_INIT_DEFERRED = 'INSTANCE_REGISTERED_INIT_DEFERRED'
    
    # Validation states
    VALIDATED_OK = 'VALIDATED_OK'
    VALIDATION_FAILED = 'VALIDATION_FAILED'
    
    # Final health states
    HEALTHY = 'HEALTHY'
    
    def is_error_status(self) -> bool:
        """Check if this status represents an error condition."""
        error_statuses = {
            self.DEFINITION_ERROR,
            self.METADATA_ERROR,
            self.METADATA_CONSTRUCTION_ERROR,
            self.INSTANTIATION_ERROR,
            self.REGISTRATION_ERROR,
            self.BOOTSTRAP_ERROR,
            self.INITIALIZATION_ERROR,
            self.VALIDATION_FAILED
        }
        return self in error_statuses
    
    def is_success_status(self) -> bool:
        """Check if this status represents a successful state."""
        success_statuses = {
            self.INSTANCE_REGISTERED,
            self.INITIALIZED_OK,
            self.INSTANCE_REGISTERED_NO_INIT,
            self.INSTANCE_REGISTERED_INIT_DEFERRED,
            self.VALIDATED_OK,
            self.HEALTHY
        }
        return self in success_statuses


class ComponentHealthRecord:
    """
    Individual health record for a component during bootstrap.
    
    Tracks all status changes, errors, and metadata for a single component
    throughout the bootstrap process.
    """
    
    def __init__(
        self,
        component_id: str,
        status: ComponentStatus,
        metadata: ComponentMetadata,
        validation_errors: List[str] = None
    ):
        self.component_id = component_id
        self.name = metadata.name
        self.category = metadata.category
        self.version = metadata.version
        self.epistemic_tags = list(metadata.epistemic_tags)
        self.requires_initialize = metadata.requires_initialize
        
        # Status tracking
        self.current_status = status
        self.status_history: List[Dict[str, Any]] = []
        self.validation_errors = validation_errors or []
        
        # Timestamps
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = self.created_at
        
        # Record initial status
        self._record_status_change(status, f"Initial status: {status.value}")
    
    def update_status(self, new_status: ComponentStatus, message: str = "") -> None:
        """Update the component status with a new value."""
        if new_status != self.current_status:
            self._record_status_change(new_status, message)
            self.current_status = new_status
            self.last_updated = datetime.now(timezone.utc)
    
    def add_validation_error(self, error: str) -> None:
        """Add a validation error to the record."""
        self.validation_errors.append(error)
        self.last_updated = datetime.now(timezone.utc)
        
        if self.current_status != ComponentStatus.VALIDATION_FAILED:
            self.update_status(ComponentStatus.VALIDATION_FAILED, f"Validation error: {error}")
    
    def _record_status_change(self, status: ComponentStatus, message: str) -> None:
        """Record a status change in the history."""
        self.status_history.append({
            'status': status.value,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def get_validation_issue_count(self) -> int:
        """Get the total number of validation issues."""
        return len(self.validation_errors)
    
    def is_healthy(self) -> bool:
        """Check if the component is in a healthy state."""
        return self.current_status.is_success_status()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the component health record."""
        return {
            'id': self.component_id,
            'name': self.name,
            'category': self.category,
            'version': self.version,
            'status': self.current_status.value,
            'epistemic_tags': self.epistemic_tags,
            'validation_issue_count': self.get_validation_issue_count(),
            'requires_initialize': self.requires_initialize,
            'is_healthy': self.is_healthy(),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }


class BootstrapHealthReporter:
    """
    Central health reporting system for the bootstrap process.
    
    Tracks the health and status of all components throughout bootstrap,
    provides summary statistics, and generates comprehensive reports.
    """
    
    def __init__(self, component_registry: ComponentRegistry):
        self.component_registry = component_registry
        self.component_records: Dict[str, ComponentHealthRecord] = {}
        self.phase_results: Dict[str, Dict[str, Any]] = {}
        self.bootstrap_start_time = datetime.now(timezone.utc)
        self.bootstrap_end_time: Optional[datetime] = None
        
        logger.info("BootstrapHealthReporter initialized")
    
    def add_component_status(
        self,
        component_id: str,
        status: ComponentStatus,
        metadata: ComponentMetadata,
        validation_errors: List[str] = None
    ) -> None:
        """
        Add or update component status in the health report.
        
        Args:
            component_id: Unique component identifier
            status: Current component status
            metadata: Component metadata
            validation_errors: List of validation errors (if any)
        """
        validation_errors = validation_errors or []
        
        if component_id in self.component_records:
            # Update existing record
            record = self.component_records[component_id]
            record.update_status(status, f"Status updated to {status.value}")
            
            for error in validation_errors:
                record.add_validation_error(error)
        else:
            # Create new record
            record = ComponentHealthRecord(
                component_id=component_id,
                status=status,
                metadata=metadata,
                validation_errors=validation_errors
            )
            self.component_records[component_id] = record
        
        logger.debug(f"Component '{component_id}' status updated to {status.value}")
    
    def add_phase_result(
        self,
        phase_name: str,
        status: str,
        message: str,
        errors: List[str] = None,
        warnings: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Record the result of a bootstrap phase.
        
        Args:
            phase_name: Name of the bootstrap phase
            status: Phase completion status
            message: Phase completion message
            errors: List of errors during phase
            warnings: List of warnings during phase
            metadata: Additional phase metadata
        """
        self.phase_results[phase_name] = {
            'status': status,
            'message': message,
            'errors': errors or [],
            'warnings': warnings or [],
            'metadata': metadata or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.debug(f"Phase '{phase_name}' result recorded: {status}")
    
    def has_critical_failures(self) -> int:
        """
        Count the number of components with critical failures.
        
        Returns:
            Number of components with critical failure status
        """
        critical_statuses = {
            ComponentStatus.DEFINITION_ERROR,
            ComponentStatus.METADATA_ERROR,
            ComponentStatus.METADATA_CONSTRUCTION_ERROR,
            ComponentStatus.INSTANTIATION_ERROR,
            ComponentStatus.REGISTRATION_ERROR,
            ComponentStatus.BOOTSTRAP_ERROR
        }
        
        return sum(
            1 for record in self.component_records.values()
            if record.current_status in critical_statuses
        )
    
    def get_component_count_by_status(self) -> Dict[str, int]:
        """Get count of components grouped by status."""
        status_counts: Dict[str, int] = {}
        
        for record in self.component_records.values():
            status = record.current_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return status_counts
    
    def get_healthy_component_count(self) -> int:
        """Get the number of healthy components."""
        return sum(1 for record in self.component_records.values() if record.is_healthy())
    
    def get_components_by_category(self) -> Dict[str, List[str]]:
        """Get components grouped by category."""
        categories: Dict[str, List[str]] = {}
        
        for record in self.component_records.values():
            category = record.category
            if category not in categories:
                categories[category] = []
            categories[category].append(record.component_id)
        
        return categories
    
    def get_epistemic_tag_distribution(self) -> Dict[str, int]:
        """Get distribution of epistemic tags across healthy components."""
        tag_counts: Dict[str, int] = {}
        
        for record in self.component_records.values():
            if record.is_healthy():
                for tag in record.epistemic_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return tag_counts
    
    def mark_bootstrap_complete(self) -> None:
        """Mark the bootstrap process as complete."""
        self.bootstrap_end_time = datetime.now(timezone.utc)
        logger.info("Bootstrap process marked as complete")
    
    def get_bootstrap_duration(self) -> Optional[float]:
        """Get the total bootstrap duration in seconds."""
        if self.bootstrap_end_time:
            return (self.bootstrap_end_time - self.bootstrap_start_time).total_seconds()
        return None
    
    def generate_summary(self) -> str:
        """
        Generate a comprehensive bootstrap health summary report.
        
        Returns:
            Multi-line string report
        """
        summary_lines = ['\n--- NIREON V4 Bootstrap Health Report ---']
        
        # Basic statistics
        total_components = len(self.component_records)
        healthy_components = self.get_healthy_component_count()
        failed_components = self.has_critical_failures()
        
        summary_lines.append(f'Total Components Processed: {total_components}')
        summary_lines.append(f'Successfully Loaded/Initialized/Validated: {healthy_components}')
        summary_lines.append(f'Failed or Errored during Bootstrap: {failed_components}')
        
        # Bootstrap duration
        duration = self.get_bootstrap_duration()
        if duration is not None:
            summary_lines.append(f'Bootstrap Duration: {duration:.2f} seconds')
        
        # Status breakdown
        status_counts = self.get_component_count_by_status()
        if status_counts:
            summary_lines.append('\nComponent Status Breakdown:')
            for status, count in sorted(status_counts.items()):
                summary_lines.append(f'  - {status}: {count}')
        
        # Components with validation issues
        components_with_errors = [
            record for record in self.component_records.values()
            if record.validation_errors
        ]
        
        if components_with_errors:
            summary_lines.append('\nComponents with Validation/Bootstrap Issues:')
            for record in components_with_errors:
                summary_lines.append(f'  - ID: {record.component_id} (Name: {record.name}, Status: {record.current_status.value})')
                for error in record.validation_errors:
                    summary_lines.append(f'    - {error}')
        
        # Epistemic tag distribution
        tag_counts = self.get_epistemic_tag_distribution()
        if tag_counts:
            summary_lines.append('\nEpistemic Tag Distribution (healthy components):')
            for tag, count in sorted(tag_counts.items()):
                summary_lines.append(f'  - {tag}: {count}')
        
        # Category breakdown
        categories = self.get_components_by_category()
        if categories:
            summary_lines.append('\nComponents by Category:')
            for category, component_ids in sorted(categories.items()):
                summary_lines.append(f'  - {category}: {len(component_ids)}')
        
        # Phase results
        if self.phase_results:
            summary_lines.append('\nBootstrap Phase Results:')
            for phase_name, result in self.phase_results.items():
                status_symbol = '✓' if result['status'] == 'completed' else '✗'
                summary_lines.append(f'  {status_symbol} {phase_name}: {result["status"]} - {result["message"]}')
                
                if result['errors']:
                    for error in result['errors']:
                        summary_lines.append(f'    Error: {error}')
                
                if result['warnings']:
                    for warning in result['warnings']:
                        summary_lines.append(f'    Warning: {warning}')
        
        # Certification summary from registry
        self._add_certification_summary(summary_lines)
        
        summary_lines.append('--- End of V4 Health Report ---\n')
        return '\n'.join(summary_lines)
    
    def _add_certification_summary(self, summary_lines: List[str]) -> None:
        """Add certification information from the component registry."""
        try:
            registered_components = self.component_registry.list_components()
            if registered_components:
                summary_lines.append(f'\nTotal Registered Components: {len(registered_components)}')
                
                certified_count = 0
                uncertified_count = 0
                
                for comp_id in registered_components:
                    try:
                        if hasattr(self.component_registry, 'get_certification'):
                            cert_data = self.component_registry.get_certification(comp_id)
                            if cert_data:
                                certified_count += 1
                            else:
                                uncertified_count += 1
                        else:
                            uncertified_count += 1
                    except Exception:
                        uncertified_count += 1
                
                summary_lines.append(f'  - Certified: {certified_count}')
                summary_lines.append(f'  - Uncertified: {uncertified_count}')
        
        except Exception as e:
            logger.debug(f'Could not generate certification summary: {e}')
    
    def get_health_data(self) -> Dict[str, Any]:
        """
        Get all health data as a structured dictionary.
        
        Returns:
            Dictionary containing all health information
        """
        return {
            'bootstrap_start_time': self.bootstrap_start_time.isoformat(),
            'bootstrap_end_time': self.bootstrap_end_time.isoformat() if self.bootstrap_end_time else None,
            'bootstrap_duration_seconds': self.get_bootstrap_duration(),
            'total_components': len(self.component_records),
            'healthy_components': self.get_healthy_component_count(),
            'critical_failures': self.has_critical_failures(),
            'status_counts': self.get_component_count_by_status(),
            'category_breakdown': self.get_components_by_category(),
            'epistemic_tag_distribution': self.get_epistemic_tag_distribution(),
            'component_records': {
                comp_id: record.get_summary()
                for comp_id, record in self.component_records.items()
            },
            'phase_results': dict(self.phase_results)
        }