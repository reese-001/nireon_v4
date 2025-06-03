from __future__ import annotations
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from application.components.lifecycle import ComponentMetadata
from core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

class ComponentStatus(Enum):
    UNKNOWN = 'UNKNOWN'
    DEFINITION_ERROR = 'DEFINITION_ERROR'
    METADATA_ERROR = 'METADATA_ERROR'
    METADATA_CONSTRUCTION_ERROR = 'METADATA_CONSTRUCTION_ERROR'
    INSTANTIATION_ERROR = 'INSTANTIATION_ERROR'
    INSTANCE_REGISTERED = 'INSTANCE_REGISTERED'
    REGISTRATION_ERROR = 'REGISTRATION_ERROR'
    BOOTSTRAP_ERROR = 'BOOTSTRAP_ERROR'
    INITIALIZATION_PENDING = 'INITIALIZATION_PENDING'
    INITIALIZED_OK = 'INITIALIZED_OK'
    INITIALIZATION_ERROR = 'INITIALIZATION_ERROR'
    INITIALIZATION_SKIPPED_NOT_REQUIRED = 'INITIALIZATION_SKIPPED_NOT_REQUIRED'
    INITIALIZATION_SKIPPED_NO_METHOD = 'INITIALIZATION_SKIPPED_NO_METHOD'
    VALIDATION_PENDING = 'VALIDATION_PENDING'
    VALIDATED_OK = 'VALIDATED_OK'
    VALIDATION_FAILED = 'VALIDATION_FAILED'
    HEALTHY = 'HEALTHY'
    DISABLED = 'DISABLED'

    def is_error_status(self) -> bool:
        error_statuses = {
            self.DEFINITION_ERROR, self.METADATA_ERROR, self.METADATA_CONSTRUCTION_ERROR,
            self.INSTANTIATION_ERROR, self.REGISTRATION_ERROR, self.BOOTSTRAP_ERROR,
            self.INITIALIZATION_ERROR, self.VALIDATION_FAILED
        }
        return self in error_statuses

    def is_terminal_success_status(self) -> bool:
        success_statuses = {
            self.HEALTHY, self.INITIALIZED_OK, self.VALIDATED_OK,
            self.INITIALIZATION_SKIPPED_NOT_REQUIRED, self.DISABLED
        }
        return self in success_statuses

    def is_intermediate_success_status(self) -> bool:
        intermediate_success = {
            self.INSTANCE_REGISTERED, self.INITIALIZATION_PENDING, self.VALIDATION_PENDING
        }
        return self in intermediate_success

class ComponentHealthRecord:
    def __init__(self, component_id: str, status: ComponentStatus, metadata: ComponentMetadata, 
                 errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        if not isinstance(metadata, ComponentMetadata):
            raise TypeError(f"metadata for '{component_id}' must be ComponentMetadata, got {type(metadata)}")
        
        self.component_id: str = component_id
        self.name: str = metadata.name
        self.category: str = metadata.category
        self.version: str = metadata.version
        self.epistemic_tags: List[str] = list(metadata.epistemic_tags or [])
        self.requires_initialize: bool = metadata.requires_initialize
        self.current_status: ComponentStatus = status
        self.status_history: List[Dict[str, Any]] = []
        self.errors: List[str] = errors or []
        self.warnings: List[str] = warnings or []
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_updated: datetime = self.created_at
        
        self._record_status_change(status, f'Initial status: {status.value}')

    def update_status(self, new_status: ComponentStatus, message: str = '') -> None:
        if not isinstance(new_status, ComponentStatus):
            logger.error(f'Invalid status type passed to update_status for {self.component_id}: {type(new_status)}')
            return
        
        if new_status != self.current_status:
            self._record_status_change(new_status, message)
            self.current_status = new_status
            self.last_updated = datetime.now(timezone.utc)

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.last_updated = datetime.now(timezone.utc)

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
        self.last_updated = datetime.now(timezone.utc)

    def _record_status_change(self, status: ComponentStatus, message: str) -> None:
        self.status_history.append({
            'status': status.value,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    def get_issue_count(self) -> int:
        return len(self.errors)

    def is_healthy(self) -> bool:
        if self.current_status == ComponentStatus.INITIALIZATION_SKIPPED_NO_METHOD:
            return not self.requires_initialize and not self.errors
        return self.current_status.is_terminal_success_status() and not self.errors

    def get_summary(self) -> Dict[str, Any]:
        return {
            'id': self.component_id,
            'name': self.name,
            'category': self.category,
            'version': self.version,
            'status': self.current_status.value,
            'epistemic_tags': self.epistemic_tags,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'requires_initialize': self.requires_initialize,
            'is_healthy_final_state': self.is_healthy(),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

class HealthReporter:
    """ Bootstrap Health Reporter with enhanced tracking and reporting"""
    
    def __init__(self, component_registry: ComponentRegistry):
        if not isinstance(component_registry, ComponentRegistry):
            raise TypeError('HealthReporter expects a ComponentRegistry instance.')
        
        self.component_registry = component_registry
        self.component_records: Dict[str, ComponentHealthRecord] = {}
        self.phase_results: Dict[str, Dict[str, Any]] = {}
        self.bootstrap_start_time: datetime = datetime.now(timezone.utc)
        self.bootstrap_end_time: Optional[datetime] = None
        
        logger.info('HealthReporter initialized for  bootstrap.')

    def add_component_status(self, component_id: str, status: ComponentStatus, metadata: ComponentMetadata,
                           errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None) -> None:
        errors = errors or []
        warnings = warnings or []
        
        if not isinstance(status, ComponentStatus):
            logger.error(f"Invalid status type '{type(status)}' for component '{component_id}'. Defaulting to UNKNOWN.")
            status = ComponentStatus.UNKNOWN
        
        if component_id in self.component_records:
            record = self.component_records[component_id]
            record.update_status(status, f'Status updated to {status.value}')
            for err in errors:
                record.add_error(err)
            for warn in warnings:
                record.add_warning(warn)
        else:
            if not isinstance(metadata, ComponentMetadata):
                logger.error(f"Invalid metadata type for new component '{component_id}': {type(metadata)}. Must be ComponentMetadata. Creating fallback.")
                metadata = ComponentMetadata(
                    id=component_id,
                    name=getattr(metadata, 'name', component_id),
                    version=getattr(metadata, 'version', '0.0.0'),
                    category=getattr(metadata, 'category', 'unknown'),
                    requires_initialize=getattr(metadata, 'requires_initialize', True)
                )
            
            record = ComponentHealthRecord(
                component_id=component_id,
                status=status,
                metadata=metadata,
                errors=errors,
                warnings=warnings
            )
            self.component_records[component_id] = record
        
        log_level = logging.ERROR if status.is_error_status() else logging.DEBUG
        logger.log(log_level, f"Component '{component_id}' status: {status.value}. Errors: {len(errors)}, Warnings: {len(warnings)}")

    def add_phase_result(self, phase_name: str, status: str, message: str,
                        errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        self.phase_results[phase_name] = {
            'status': status,
            'message': message,
            'errors': errors or [],
            'warnings': warnings or [],
            'metadata': metadata or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        log_level = logging.ERROR if status == 'failed' else logging.INFO
        logger.log(log_level, f"Phase '{phase_name}' result: {status} - {message}")

    def has_critical_failures(self) -> int:
        return sum(1 for record in self.component_records.values() if record.current_status.is_error_status())

    def get_component_count_by_status(self) -> Dict[str, int]:
        status_counts: Dict[str, int] = {}
        for record in self.component_records.values():
            status_val = record.current_status.value
            status_counts[status_val] = status_counts.get(status_val, 0) + 1
        return status_counts

    def get_healthy_component_count(self) -> int:
        return sum(1 for record in self.component_records.values() if record.is_healthy())

    def get_components_by_category(self) -> Dict[str, List[str]]:
        categories: Dict[str, List[str]] = {}
        for record in self.component_records.values():
            category = record.category
            if category not in categories:
                categories[category] = []
            categories[category].append(record.component_id)
        return categories

    def get_epistemic_tag_distribution(self) -> Dict[str, int]:
        tag_counts: Dict[str, int] = {}
        for record in self.component_records.values():
            if record.is_healthy():
                for tag in record.epistemic_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

    def mark_bootstrap_complete(self) -> None:
        self.bootstrap_end_time = datetime.now(timezone.utc)
        logger.info(' Bootstrap process marked as complete by HealthReporter.')

    def get_bootstrap_duration(self) -> Optional[float]:
        if self.bootstrap_end_time:
            return (self.bootstrap_end_time - self.bootstrap_start_time).total_seconds()
        return (datetime.now(timezone.utc) - self.bootstrap_start_time).total_seconds()

    def generate_summary(self) -> str:
        summary_lines = ['\n--- NIREON  Bootstrap Health Report ---']
        
        total_components = len(self.component_records)
        operational_components = self.get_healthy_component_count()
        critical_failures = self.has_critical_failures()
        
        summary_lines.append(f'Total Components Processed: {total_components}')
        summary_lines.append(f'Operational (Healthy & Ready): {operational_components}')
        summary_lines.append(f'Critical Failures during Bootstrap: {critical_failures}')
        
        duration = self.get_bootstrap_duration()
        if duration is not None:
            summary_lines.append(f'Bootstrap Duration: {duration:.2f} seconds')
        
        # Status breakdown
        status_counts = self.get_component_count_by_status()
        if status_counts:
            summary_lines.append('\nComponent Status Breakdown:')
            for status, count in sorted(status_counts.items()):
                summary_lines.append(f'  - {status}: {count}')
        
        # Components with issues
        components_with_issues = [record for record in self.component_records.values() 
                                if record.errors or record.warnings]
        if components_with_issues:
            summary_lines.append('\nComponents with Issues (Errors/Warnings):')
            for record in components_with_issues:
                summary_lines.append(f'  - ID: {record.component_id} (Name: {record.name}, Status: {record.current_status.value})')
                for err in record.errors:
                    summary_lines.append(f'    - ERROR: {err}')
                for warn in record.warnings:
                    summary_lines.append(f'    - WARNING: {warn}')
        
        # Epistemic tag distribution
        tag_counts = self.get_epistemic_tag_distribution()
        if tag_counts:
            summary_lines.append('\nEpistemic Tag Distribution (operational components):')
            for tag, count in sorted(tag_counts.items()):
                summary_lines.append(f'  - {tag}: {count}')
        
        # Categories
        categories = self.get_components_by_category()
        if categories:
            summary_lines.append('\nComponents by Category:')
            for category, component_ids in sorted(categories.items()):
                summary_lines.append(f'  - {category}: {len(component_ids)}')
        
        # Phase results
        if self.phase_results:
            summary_lines.append('\nBootstrap Phase Results:')
            for phase_name, result in self.phase_results.items():
                status_symbol = '✓' if result['status'] == 'completed' else '⚠' if result['status'] in ['warning', 'skipped'] else '✗'
                summary_lines.append(f"  {status_symbol} {phase_name}: {result['status']} - {result['message']}")
                for error in result.get('errors', []):
                    summary_lines.append(f'    Error: {error}')
                for warning in result.get('warnings', []):
                    summary_lines.append(f'    Warning: {warning}')
        
        # Certification summary
        self._add_certification_summary(summary_lines)
        
        summary_lines.append('--- End of NIREON  Health Report ---\n')
        return '\n'.join(summary_lines)

    def _add_certification_summary(self, summary_lines: List[str]) -> None:
        try:
            all_component_ids = self.component_registry.list_components()
            if not all_component_ids:
                summary_lines.append('\nNo components registered for certification summary.')
                return
            
            summary_lines.append(f'\nComponent Certification Summary (Total Registered: {len(all_component_ids)}):')
            
            certified_count = 0
            uncertified_details: List[str] = []
            
            for comp_id in all_component_ids:
                try:
                    cert_data = self.component_registry.get_certification(comp_id)
                    if cert_data and cert_data.get('certification_hash'):
                        certified_count += 1
                    else:
                        reason = 'No certification data' if not cert_data else 'Missing/Invalid certification_hash'
                        uncertified_details.append(f'{comp_id} ({reason})')
                except Exception:
                    uncertified_details.append(f'{comp_id} (Error retrieving cert)')
            
            summary_lines.append(f'  - Certified Components: {certified_count}')
            uncertified_count = len(all_component_ids) - certified_count
            summary_lines.append(f'  - Uncertified or Errored Certification: {uncertified_count}')
            
            if uncertified_details and len(uncertified_details) < 10:
                summary_lines.append('    Uncertified/Error Details:')
                for detail in uncertified_details:
                    summary_lines.append(f'      - {detail}')
            elif uncertified_details:
                summary_lines.append(f'    (Details for {len(uncertified_details)} uncertified/errored components omitted for brevity)')
                
        except Exception as e:
            logger.warning(f'Could not generate  certification summary: {e}', exc_info=True)
            summary_lines.append(f'\nCertification Summary Error: {type(e).__name__}')

    def get_health_data(self) -> Dict[str, Any]:
        cert_summary_data = {}
        try:
            all_ids = self.component_registry.list_components()
            certified_ids = []
            uncertified_ids = []
            
            for cid in all_ids:
                try:
                    cert_data = self.component_registry.get_certification(cid)
                    if cert_data and cert_data.get('certification_hash'):
                        certified_ids.append(cid)
                    else:
                        uncertified_ids.append(cid)
                except:
                    uncertified_ids.append(cid)
            
            cert_summary_data = {
                'total_registered': len(all_ids),
                'certified_count': len(certified_ids),
                'uncertified_count': len(uncertified_ids)
            }
        except Exception as e_cert_data:
            cert_summary_data = {'error': str(e_cert_data)}
        
        return {
            'bootstrap_start_time': self.bootstrap_start_time.isoformat(),
            'bootstrap_end_time': self.bootstrap_end_time.isoformat() if self.bootstrap_end_time else None,
            'bootstrap_duration_seconds': self.get_bootstrap_duration(),
            'total_components_processed': len(self.component_records),
            'operational_components': self.get_healthy_component_count(),
            'critical_failures': self.has_critical_failures(),
            'status_counts': self.get_component_count_by_status(),
            'category_breakdown': self.get_components_by_category(),
            'epistemic_tag_distribution': self.get_epistemic_tag_distribution(),
            'component_records': {comp_id: record.get_summary() for comp_id, record in self.component_records.items()},
            'phase_results': dict(self.phase_results),
            'certification_summary': cert_summary_data
        }