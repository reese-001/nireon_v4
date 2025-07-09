import logging
from typing import Dict, Type, Optional, Any, List, Union
from domain.context import NireonExecutionContext

logger = logging.getLogger(__name__)


class ServiceResolutionMixin:
    """
    Mixin to provide standardized service resolution capabilities for components.
    
    This mixin helps components lazily resolve their service dependencies from the
    component registry, with options for error handling and fallback behavior.
    """
    
    def resolve_services(
        self,
        context: NireonExecutionContext,
        service_map: Dict[str, Union[Type, str]],
        raise_on_missing: bool = True,
        log_resolution: bool = True
    ) -> Dict[str, Optional[Any]]:
        """
        Resolve multiple services from the component registry.
        
        Args:
            context: The execution context containing the component registry
            service_map: Dict mapping attribute names to service types or string identifiers
            raise_on_missing: Whether to raise an exception if services can't be resolved
            log_resolution: Whether to log resolution attempts
            
        Returns:
            Dict of resolved services (attr_name -> service instance)
            
        Raises:
            RuntimeError: If raise_on_missing=True and any required services are missing
        """
        resolved_services = {}
        missing_services = []
        component_id = getattr(self, 'component_id', self.__class__.__name__)
        
        for attr_name, service_identifier in service_map.items():
            # Skip if already resolved
            if getattr(self, attr_name, None) is not None:
                if log_resolution:
                    logger.debug(f"[{component_id}] Service '{attr_name}' already resolved, skipping.")
                resolved_services[attr_name] = getattr(self, attr_name)
                continue
            
            try:
                # Resolve by type or string identifier
                if isinstance(service_identifier, type):
                    service = context.component_registry.get_service_instance(service_identifier)
                    service_name = service_identifier.__name__
                else:
                    service = context.component_registry.get(service_identifier)
                    service_name = service_identifier
                
                if service:
                    setattr(self, attr_name, service)
                    resolved_services[attr_name] = service
                    if log_resolution:
                        logger.info(f"[{component_id}] Resolved service '{attr_name}' -> {service_name}")
                else:
                    missing_services.append((attr_name, service_name))
                    if log_resolution:
                        logger.warning(f"[{component_id}] Failed to resolve service '{attr_name}' ({service_name})")
                        
            except Exception as e:
                logger.error(f"[{component_id}] Error resolving service '{attr_name}': {e}")
                missing_services.append((attr_name, str(service_identifier)))
        
        if missing_services and raise_on_missing:
            missing_list = ", ".join(f"{attr} ({svc})" for attr, svc in missing_services)
            raise RuntimeError(f"{component_id} failed to resolve required services: {missing_list}")
        
        return resolved_services
    
    def validate_required_services(
        self,
        required_attrs: List[str],
        context: Optional[NireonExecutionContext] = None
    ) -> bool:
        """
        Validate that all required service attributes are present.
        
        Args:
            required_attrs: List of attribute names that must be non-None
            context: Optional context for logging
            
        Returns:
            True if all required services are available, False otherwise
        """
        component_id = getattr(self, 'component_id', self.__class__.__name__)
        missing = []
        
        for attr in required_attrs:
            if not getattr(self, attr, None):
                missing.append(attr)
        
        if missing:
            msg = f"[{component_id}] Missing required services: {', '.join(missing)}"
            if context:
                context.logger.error(msg)
            else:
                logger.error(msg)
            return False
        
        return True
    
    def get_service_or_default(
        self,
        attr_name: str,
        service_identifier: Union[Type, str],
        context: NireonExecutionContext,
        default: Any = None
    ) -> Any:
        """
        Get a service or return a default value if resolution fails.
        
        Args:
            attr_name: Attribute name to set on self
            service_identifier: Service type or string identifier
            context: Execution context
            default: Default value if resolution fails
            
        Returns:
            The resolved service or the default value
        """
        # Return existing if already set
        existing = getattr(self, attr_name, None)
        if existing is not None:
            return existing
        
        # Try to resolve
        try:
            self.resolve_services(
                context,
                {attr_name: service_identifier},
                raise_on_missing=False,
                log_resolution=False
            )
            return getattr(self, attr_name, default)
        except Exception:
            return default