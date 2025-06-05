# -*- coding: utf-8 -*-
"""
Execution Context Management for NIREON V4

Provides thread-safe execution context tracking for RBAC and other cross-cutting concerns.
"""

import asyncio
import contextvars
import threading
from typing import Optional, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Context variables for async contexts
_current_subject: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('current_subject', default=None)
_current_context: contextvars.ContextVar[Optional['ExecutionContext']] = contextvars.ContextVar('current_context', default=None)

# Thread-local storage for sync contexts
_thread_local = threading.local()


@dataclass
class ExecutionContext:
    """
    Execution context that tracks current subject and other execution metadata.
    
    This provides a way to track who is performing operations across
    the entire execution chain, useful for RBAC, auditing, and logging.
    """
    subject: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component_id: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Convenience properties for RBAC
    @property
    def current_subject(self) -> Optional[str]:
        """Primary subject identifier for RBAC"""
        return self.subject or self.user_id
    
    def update(self, **kwargs) -> 'ExecutionContext':
        """Create a new context with updated values"""
        new_data = {
            'subject': self.subject,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'component_id': self.component_id,
            'operation': self.operation,
            'metadata': self.metadata.copy(),
            'created_at': self.created_at
        }
        new_data.update(kwargs)
        return ExecutionContext(**new_data)


class ExecutionContextManager:
    """Manager for execution context operations"""
    
    @staticmethod
    def set_context(context: ExecutionContext) -> None:
        """Set the current execution context"""
        _current_context.set(context)
        # Also set subject separately for easier access
        if context.current_subject:
            _current_subject.set(context.current_subject)
        
        # Set in thread-local for sync contexts
        _thread_local.context = context
        if context.current_subject:
            _thread_local.subject = context.current_subject
    
    @staticmethod
    def get_context() -> Optional[ExecutionContext]:
        """Get the current execution context"""
        # Try context vars first (async)
        context = _current_context.get(None)
        if context:
            return context
        
        # Fall back to thread-local (sync)
        return getattr(_thread_local, 'context', None)
    
    @staticmethod
    def get_subject() -> Optional[str]:
        """Get the current subject (convenience method)"""
        # Try context vars first
        subject = _current_subject.get(None)
        if subject:
            return subject
        
        # Try from full context
        context = ExecutionContextManager.get_context()
        if context and context.current_subject:
            return context.current_subject
        
        # Fall back to thread-local
        return getattr(_thread_local, 'subject', None)
    
    @staticmethod
    def set_subject(subject: str) -> None:
        """Set just the current subject"""
        _current_subject.set(subject)
        _thread_local.subject = subject
        
        # Update full context if it exists
        context = ExecutionContextManager.get_context()
        if context:
            updated_context = context.update(subject=subject)
            ExecutionContextManager.set_context(updated_context)
        else:
            # Create minimal context
            new_context = ExecutionContext(subject=subject)
            ExecutionContextManager.set_context(new_context)
    
    @staticmethod
    def clear_context() -> None:
        """Clear the current execution context"""
        _current_context.set(None)
        _current_subject.set(None)
        
        if hasattr(_thread_local, 'context'):
            del _thread_local.context
        if hasattr(_thread_local, 'subject'):
            del _thread_local.subject


# Context manager for temporary context changes
class ExecutionContextScope:
    """
    Context manager for temporarily setting execution context.
    
    Usage:
        with ExecutionContextScope(subject="admin@company.com"):
            # All operations in this block run as admin
            await some_operation()
    """
    
    def __init__(self, subject: str = None, **context_data):
        self.new_context = ExecutionContext(subject=subject, **context_data)
        self.previous_context = None
    
    def __enter__(self):
        self.previous_context = ExecutionContextManager.get_context()
        ExecutionContextManager.set_context(self.new_context)
        return self.new_context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_context:
            ExecutionContextManager.set_context(self.previous_context)
        else:
            ExecutionContextManager.clear_context()


# Convenience functions
def get_current_execution_context() -> Optional[ExecutionContext]:
    """Get the current execution context (main function used by decorators)"""
    return ExecutionContextManager.get_context()


def get_current_subject() -> Optional[str]:
    """Get the current subject for RBAC"""
    return ExecutionContextManager.get_subject()


def set_execution_subject(subject: str) -> None:
    """Set the current subject"""
    ExecutionContextManager.set_subject(subject)


def with_subject(subject: str):
    """Decorator to run a function with a specific subject"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with ExecutionContextScope(subject=subject):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with ExecutionContextScope(subject=subject):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


# Integration with FastAPI (optional)
def create_fastapi_context_middleware():
    """
    Create FastAPI middleware to automatically set execution context from requests.
    
    Usage:
        from security.execution_context import create_fastapi_context_middleware
        app.add_middleware(create_fastapi_context_middleware())
    """
    try:
        from fastapi import Request, Response
        from starlette.middleware.base import BaseHTTPMiddleware
        import uuid
        
        class ExecutionContextMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Extract subject from request (customize based on your auth)
                subject = None
                
                # Try to get subject from various sources
                if hasattr(request.state, 'user') and hasattr(request.state.user, 'email'):
                    subject = request.state.user.email
                elif 'Authorization' in request.headers:
                    # You could decode JWT or API key here
                    pass
                elif 'X-Subject' in request.headers:
                    subject = request.headers['X-Subject']
                
                # Create execution context
                context = ExecutionContext(
                    subject=subject,
                    request_id=str(uuid.uuid4()),
                    metadata={
                        'method': request.method,
                        'url': str(request.url),
                        'client_ip': request.client.host if request.client else None
                    }
                )
                
                with ExecutionContextScope(subject=subject, **context.__dict__):
                    response = await call_next(request)
                    return response
        
        return ExecutionContextMiddleware
        
    except ImportError:
        # FastAPI not available
        return None


# For backwards compatibility and convenience
current_execution_context = get_current_execution_context
current_subject = get_current_subject