# -*- coding: utf-8 -*-
"""
RBAC Permission Decorators

Provides easy-to-use decorators for enforcing RBAC permissions on methods and functions.
Integrates seamlessly with the RBACPolicyEngine registered during bootstrap.
"""

import logging
from functools import wraps
from typing import Optional, Union, Any, Callable
import asyncio
import inspect

logger = logging.getLogger(__name__)


class RBACPermissionError(Exception):
    """Exception raised when RBAC permission is denied"""
    
    def __init__(self, subject: str, resource: str, action: str, message: str = None):
        self.subject = subject
        self.resource = resource
        self.action = action
        self.message = message or f"Access denied: {subject} cannot {action} on {resource}"
        super().__init__(self.message)


class RBACContextManager:
    """
    Context manager for RBAC operations.
    Provides multiple ways to get the current subject and access the RBAC engine.
    """
    
    @staticmethod
    def get_rbac_engine():
        """Get the RBAC engine from the component registry"""
        try:
            from registry.registry_manager import RegistryManager
            return RegistryManager.get_service("rbac_engine")
        except Exception as e:
            logger.error(f"Failed to get RBAC engine from registry: {e}")
            raise RuntimeError("RBAC engine not available - ensure RBAC bootstrap completed successfully")
    
    @staticmethod
    def get_current_subject(context_sources: list = None) -> str:
        """
        Get the current subject from various possible sources.
        
        Args:
            context_sources: List of additional context sources to check
            
        Returns:
            Current subject identifier, defaults to 'system' if none found
        """
        context_sources = context_sources or []
        
        # Try to get subject from various sources in order of preference
        subject_sources = [
            # 1. Try execution context
            lambda: _get_from_execution_context(),
            
            # 2. Try FastAPI request context  
            lambda: _get_from_fastapi_context(),
            
            # 3. Try custom context sources
            *[lambda src=src: _get_from_custom_source(src) for src in context_sources],
            
            # 4. Try environment/config
            lambda: _get_from_environment(),
            
            # 5. Default fallback
            lambda: "system"
        ]
        
        for source_func in subject_sources:
            try:
                subject = source_func()
                if subject:
                    return subject
            except Exception as e:
                logger.debug(f"RBAC context source failed: {e}")
                continue
        
        return "system"


def _get_from_execution_context() -> Optional[str]:
    """Try to get subject from NIREON execution context"""
    try:
        from security.execution_context import get_current_execution_context
        context = get_current_execution_context()
        return getattr(context, 'current_subject', None) or getattr(context, 'subject', None)
    except:
        return None


def _get_from_fastapi_context() -> Optional[str]:
    """Try to get subject from FastAPI request context"""
    try:
        from fastapi import Request
        import contextvars
        
        # Try to get from FastAPI request context
        request_context = contextvars.copy_context()
        for var, value in request_context.items():
            if hasattr(value, 'user') and hasattr(value.user, 'email'):
                return value.user.email
            elif hasattr(value, 'subject'):
                return value.subject
    except:
        return None


def _get_from_custom_source(source: Any) -> Optional[str]:
    """Try to get subject from custom source"""
    if callable(source):
        return source()
    elif isinstance(source, str):
        return source
    elif hasattr(source, 'subject'):
        return source.subject
    elif hasattr(source, 'email'):
        return source.email
    return None


def _get_from_environment() -> Optional[str]:
    """Try to get subject from environment variables"""
    import os
    return os.getenv('RBAC_CURRENT_SUBJECT') or os.getenv('CURRENT_USER')


def requires_permission(resource: str, action: str, subject_source: Union[str, Callable] = None, 
                       raise_on_deny: bool = True, audit: bool = False):
    """
    Decorator to require RBAC permission for a function or method.
    
    Args:
        resource: The resource being accessed (e.g., "idea", "component/sensitive")
        action: The action being performed (e.g., "read", "write", "delete")
        subject_source: Custom subject source (string, callable, or context object)
        raise_on_deny: Whether to raise exception on permission denial (default: True)
        audit: Whether to log detailed audit information (default: False)
        
    Usage:
        @requires_permission("idea", "delete")
        async def delete_idea(self, idea_id: str):
            # Only executes if current subject can delete ideas
            ...
            
        @requires_permission("component/sensitive", "read", subject_source="admin@system")
        def get_sensitive_data(self):
            # Only executes if admin@system can read sensitive components
            ...
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _check_permission_and_execute(
                func, args, kwargs, resource, action, subject_source, raise_on_deny, audit
            )
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _check_permission_and_execute_async(
                func, args, kwargs, resource, action, subject_source, raise_on_deny, audit
            )
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def _check_permission_and_execute(func, args, kwargs, resource, action, subject_source, raise_on_deny, audit):
    """Execute permission check for sync functions"""
    
    # Get subject
    if subject_source:
        if callable(subject_source):
            subject = subject_source()
        else:
            subject = str(subject_source)
    else:
        subject = RBACContextManager.get_current_subject()
    
    # Get RBAC engine
    engine = RBACContextManager.get_rbac_engine()
    
    # Perform permission check
    if audit:
        audit_result = engine.audit_permission(subject, resource, action)
        logger.info(f"RBAC Audit: {audit_result}")
        is_allowed = audit_result['final_decision']
    else:
        is_allowed = engine.is_allowed(subject, resource, action)
    
    # Handle permission denial
    if not is_allowed:
        error_msg = f"RBAC: {subject} denied {action} on {resource}"
        logger.warning(error_msg)
        
        if raise_on_deny:
            raise RBACPermissionError(subject, resource, action)
        else:
            return None
    
    # Permission granted - execute function
    logger.debug(f"RBAC: {subject} granted {action} on {resource}")
    return func(*args, **kwargs)


async def _check_permission_and_execute_async(func, args, kwargs, resource, action, subject_source, raise_on_deny, audit):
    """Execute permission check for async functions"""
    
    # Get subject
    if subject_source:
        if callable(subject_source):
            subject = subject_source()
        else:
            subject = str(subject_source)
    else:
        subject = RBACContextManager.get_current_subject()
    
    # Get RBAC engine
    engine = RBACContextManager.get_rbac_engine()
    
    # Perform permission check
    if audit:
        audit_result = engine.audit_permission(subject, resource, action)
        logger.info(f"RBAC Audit: {audit_result}")
        is_allowed = audit_result['final_decision']
    else:
        is_allowed = engine.is_allowed(subject, resource, action)
    
    # Handle permission denial
    if not is_allowed:
        error_msg = f"RBAC: {subject} denied {action} on {resource}"
        logger.warning(error_msg)
        
        if raise_on_deny:
            raise RBACPermissionError(subject, resource, action)
        else:
            return None
    
    # Permission granted - execute function
    logger.debug(f"RBAC: {subject} granted {action} on {resource}")
    return await func(*args, **kwargs)


# Convenience aliases for common permissions
def requires_read(resource: str, **kwargs):
    """Shorthand for read permission"""
    return requires_permission(resource, "read", **kwargs)


def requires_write(resource: str, **kwargs):
    """Shorthand for write permission"""
    return requires_permission(resource, "write", **kwargs)


def requires_delete(resource: str, **kwargs):
    """Shorthand for delete permission"""
    return requires_permission(resource, "delete", **kwargs)


def requires_execute(resource: str, **kwargs):
    """Shorthand for execute permission"""
    return requires_permission(resource, "execute", **kwargs)


def admin_required(resource: str = "*", **kwargs):
    """Require admin permissions (full access to resource)"""
    return requires_permission(resource, "*", **kwargs)


# Context manager for manual permission checking
class RBACContext:
    """
    Context manager for manual RBAC operations within a function.
    
    Usage:
        with RBACContext("user@example.com") as rbac:
            if rbac.can("idea", "delete"):
                # Perform delete operation
                pass
            
            rbac.require("sensitive_data", "read")  # Raises on deny
    """
    
    def __init__(self, subject: str = None):
        self.subject = subject or RBACContextManager.get_current_subject()
        self.engine = None
    
    def __enter__(self):
        self.engine = RBACContextManager.get_rbac_engine()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def can(self, resource: str, action: str) -> bool:
        """Check if subject can perform action on resource"""
        return self.engine.is_allowed(self.subject, resource, action)
    
    def require(self, resource: str, action: str) -> None:
        """Require permission, raise RBACPermissionError if denied"""
        if not self.can(resource, action):
            raise RBACPermissionError(self.subject, resource, action)
    
    def get_allowed_actions(self, resource: str) -> set:
        """Get all allowed actions for resource"""
        return self.engine.get_allowed_actions(self.subject, resource)
    
    def audit(self, resource: str, action: str) -> dict:
        """Get detailed audit information for permission check"""
        return self.engine.audit_permission(self.subject, resource, action)


# Utility functions for direct permission checking
def check_permission(subject: str, resource: str, action: str) -> bool:
    """
    Direct permission check without decorator.
    
    Args:
        subject: Subject requesting access
        resource: Resource being accessed
        action: Action being performed
        
    Returns:
        True if permission granted, False otherwise
    """
    engine = RBACContextManager.get_rbac_engine()
    return engine.is_allowed(subject, resource, action)


def require_permission(subject: str, resource: str, action: str) -> None:
    """
    Direct permission check that raises exception on denial.
    
    Args:
        subject: Subject requesting access
        resource: Resource being accessed
        action: Action being performed
        
    Raises:
        RBACPermissionError: If permission is denied
    """
    if not check_permission(subject, resource, action):
        raise RBACPermissionError(subject, resource, action)