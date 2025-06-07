"""
Exception classes for the NIREON V4 bootstrap system.

This module defines custom exceptions used throughout the bootstrap process
to provide clear error handling and debugging information.
"""

from typing import List, Optional


class BootstrapError(RuntimeError):
    """
    Base exception for all bootstrap-related errors.
    
    This exception is raised when critical failures occur during system
    initialization that prevent successful bootstrap completion.
    """
    
    def __init__(self, message: str, component_id: Optional[str] = None, phase: Optional[str] = None):
        super().__init__(message)
        self.component_id = component_id
        self.phase = phase
    
    def __str__(self) -> str:
        base_msg = super().__str__()
        
        context_parts = []
        if self.phase:
            context_parts.append(f"phase={self.phase}")
        if self.component_id:
            context_parts.append(f"component={self.component_id}")
        
        if context_parts:
            return f"{base_msg} ({', '.join(context_parts)})"
        return base_msg


class ComponentInstantiationError(BootstrapError):
    """
    Raised when a component cannot be instantiated.
    
    This includes failures in class loading, constructor calls,
    dependency injection, or factory creation.
    """
    pass


class ComponentInitializationError(BootstrapError):
    """
    Raised when a component's initialize() method fails.
    
    This indicates the component was created successfully but
    failed during its initialization lifecycle method.
    """
    pass


class ComponentValidationError(BootstrapError):
    """
    Raised when a component fails interface or contract validation.
    
    This indicates the component was created and initialized but
    does not conform to expected interfaces or metadata.
    """
    
    def __init__(self, message: str, component_id: str, validation_errors: List[str]):
        super().__init__(message, component_id=component_id)
        self.validation_errors = validation_errors
    
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.validation_errors:
            error_list = "\n  - ".join(self.validation_errors)
            return f"{base_msg}\nValidation errors:\n  - {error_list}"
        return base_msg


class ManifestProcessingError(BootstrapError):
    """
    Raised when manifest files cannot be processed.
    
    This includes schema validation failures, missing required fields,
    or malformed YAML content.
    """
    
    def __init__(self, message: str, manifest_path: Optional[str] = None, schema_errors: Optional[List[str]] = None):
        super().__init__(message, phase="manifest_processing")
        self.manifest_path = manifest_path
        self.schema_errors = schema_errors or []
    
    def __str__(self) -> str:
        base_msg = super().__str__()
        
        context_parts = []
        if self.manifest_path:
            context_parts.append(f"manifest={self.manifest_path}")
        
        if context_parts:
            base_msg = f"{base_msg} ({', '.join(context_parts)})"
        
        if self.schema_errors:
            error_list = "\n  - ".join(self.schema_errors)
            return f"{base_msg}\nSchema errors:\n  - {error_list}"
        
        return base_msg


class ConfigurationError(BootstrapError):
    """
    Raised when configuration loading or merging fails.
    
    This includes missing required configuration files, invalid
    environment variable substitution, or configuration validation errors.
    """
    pass

class BootstrapTimeoutError(BootstrapError):
    """
    Raised when configuration loading or merging fails.
    
    This includes missing required configuration files, invalid
    environment variable substitution, or configuration validation errors.
    """
    pass

class BootstrapValidationError(BootstrapError):
    """
    Raised when configuration loading or merging fails.
    
    This includes missing required configuration files, invalid
    environment variable substitution, or configuration validation errors.
    """
    pass

class BootstrapContextBuildError(BootstrapError):
    """Raised when BootstrapContextBuilder cannot assemble a valid BootstrapContext."""
    pass


class DependencyResolutionError(BootstrapError):
    """
    Raised when component dependencies cannot be resolved.
    
    This indicates missing services, circular dependencies,
    or factory setup failures.
    """
    pass


class FactoryError(BootstrapError):
    """
    Raised when factory setup or component creation fails.
    
    This includes mechanism factory initialization errors
    or factory method execution failures.
    """
    pass


class StepCommandError(BootstrapError):
    """
    Raised when orchestration step commands fail during bootstrap.
    
    This is used for orchestration command registration and execution
    errors during the bootstrap process.
    """
    pass


class RegistryError(BootstrapError):
    """
    Raised when component registry operations fail.
    
    This includes registration failures, lookup errors,
    or registry state inconsistencies.
    """
    pass


class RBACError(BootstrapError):
    """
    Raised when RBAC policy loading or processing fails.
    
    This includes missing policy files, invalid policy syntax,
    or RBAC system initialization errors.
    """
    pass


class HealthReportingError(BootstrapError):
    """
    Raised when health reporting system fails.
    
    This is used for errors in health status tracking
    or report generation during bootstrap.
    """
    pass


class PhaseExecutionError(BootstrapError):
    """
    Raised when a bootstrap phase fails to execute.
    
    This wraps errors that occur during phase execution
    and provides context about which phase failed.
    """
    
    def __init__(self, message: str, phase: str, original_error: Optional[Exception] = None):
        super().__init__(message, phase=phase)
        self.original_error = original_error
    
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.original_error:
            return f"{base_msg}\nCaused by: {type(self.original_error).__name__}: {self.original_error}"
        return base_msg
    

