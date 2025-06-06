from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type
import hashlib

from core.registry import ComponentRegistry
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata

logger = logging.getLogger(__name__)

class RegistryManager:
    def __init__(self, registry: ComponentRegistry):
        if not isinstance(registry, ComponentRegistry):
            raise TypeError('registry must be a ComponentRegistry instance')
        self.registry = registry
        logger.debug('RegistryManager initialized with ComponentRegistry')

    def register_with_certification(self, component: Any, metadata: ComponentMetadata, 
                                   additional_cert_data: Optional[Dict[str, Any]] = None) -> None:
        """Register a component with proper metadata and certification."""
        component_id = metadata.id
        
        try:
            # First register the component
            self.registry.register(component, metadata)
            logger.debug(f"Component '{component_id}' registered in registry")
            
            # Then register certification
            cert_data = self._generate_certification_data(component, metadata, additional_cert_data)
            self.registry.register_certification(component_id, cert_data)
            logger.info(f"✓ Component '{component_id}' registered with self-certification")
            
        except Exception as e:
            logger.error(f"Failed to register component '{component_id}' with certification: {e}")
            # Clean up partial registration
            try:
                if hasattr(self.registry, 'unregister'):
                    self.registry.unregister(component_id)
            except Exception:
                pass
            raise

    def register_service_with_certification(self, service_type: Type, instance: Any, 
                                          service_id: str, category: str = 'service',
                                          description: Optional[str] = None,
                                          requires_initialize: bool = False) -> None:
        """Register a service instance with proper metadata and certification."""
        from bootstrap.bootstrap_helper.metadata import create_service_metadata
        
        # Create proper metadata for the service
        metadata = create_service_metadata(
            service_id=service_id,
            service_name=service_id,
            category=category,
            description=description or f'Service instance for {service_id}',
            requires_initialize=requires_initialize
        )
        
        try:
            # Register by type if the registry supports it
            if hasattr(self.registry, 'register_service_instance'):
                self.registry.register_service_instance(service_type, instance)
                logger.debug(f"Service '{service_id}' registered by type: {service_type.__name__}")
            
            # Always register with proper metadata and certification
            self.register_with_certification(instance, metadata)
            
        except Exception as e:
            logger.error(f"Failed to register service '{service_id}': {e}")
            raise

    def _generate_certification_data(self, component: Any, metadata: ComponentMetadata,
                                   additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate certification data for a component."""
        cert_data = {
            'component_id': metadata.id,
            'component_name': metadata.name,
            'version': metadata.version,
            'category': metadata.category,
            'status': 'registered',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'epistemic_tags': list(metadata.epistemic_tags),
            'capabilities': list(metadata.capabilities)
        }
        
        # Add config hash if available
        if hasattr(component, 'config'):
            config_str = str(sorted(component.config.items())) if hasattr(component.config, 'items') else str(component.config)
            cert_data['config_hash'] = hashlib.sha256(config_str.encode()).hexdigest()
        
        # Add NireonBaseComponent specific info
        if isinstance(component, NireonBaseComponent):
            cert_data['base_component'] = True
            cert_data['is_initialized'] = getattr(component, 'is_initialized', False)
            cert_data['process_count'] = getattr(component, 'process_count', 0)
            cert_data['error_count'] = getattr(component, 'error_count', 0)
        
        # Add any additional certification data
        if additional_data:
            cert_data.update(additional_data)
        
        # Generate certification hash
        cert_data['certification_hash'] = self._generate_certification_hash(cert_data)
        
        return cert_data

    def _generate_certification_hash(self, cert_data: Dict[str, Any]) -> str:
        """Generate a hash for certification integrity."""
        # Remove the hash itself from the data to hash
        data_for_hash = {k: v for k, v in cert_data.items() if k != 'certification_hash'}
        canonical_str = str(sorted(data_for_hash.items()))
        return hashlib.sha256(canonical_str.encode()).hexdigest()[:16]

    def verify_certification(self, component_id: str) -> bool:
        """Verify that a component's certification is valid."""
        try:
            cert_data = self.registry.get_certification(component_id)
            if not cert_data:
                logger.warning(f"No certification found for component '{component_id}'")
                return False
            
            stored_hash = cert_data.get('certification_hash')
            if not stored_hash:
                logger.warning(f"No certification hash found for component '{component_id}'")
                return False
            
            calculated_hash = self._generate_certification_hash(cert_data)
            is_valid = stored_hash == calculated_hash
            
            if not is_valid:
                logger.error(f"Certification hash mismatch for component '{component_id}'")
                logger.debug(f'Stored: {stored_hash}, Calculated: {calculated_hash}')
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying certification for component '{component_id}': {e}")
            return False

    def list_certified_components(self) -> List[str]:
        """Get a list of all properly certified components."""
        certified = []
        for component_id in self.registry.list_components():
            if self.verify_certification(component_id):
                certified.append(component_id)
        return certified

    def get_certification_summary(self) -> Dict[str, Any]:
        """Get a summary of component certifications."""
        all_components = self.registry.list_components()
        certified_components = self.list_certified_components()
        
        return {
            'total_components': len(all_components),
            'certified_components': len(certified_components),
            'certification_rate': len(certified_components) / len(all_components) if all_components else 0.0,
            'uncertified_components': [cid for cid in all_components if cid not in certified_components]
        }