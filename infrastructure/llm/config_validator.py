# nireon/infrastructure/llm/config_validator.py
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationError:
    path: str
    message: str
    severity: str = 'error'  # 'error', 'warning'

class LLMConfigValidator:
    """Validates LLM configuration against expected schema."""
    
    REQUIRED_MODEL_FIELDS = {'provider', 'backend'}
    VALID_AUTH_STYLES = {'bearer', 'header_key', 'query_param'}
    VALID_HTTP_METHODS = {'GET', 'POST', 'PUT', 'PATCH'}
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate the entire LLM configuration."""
        errors = []
        
        # Validate top-level structure
        if 'models' not in config:
            errors.append(ValidationError('', 'Missing required "models" section'))
            return errors  # Can't continue without models
        
        if 'default' not in config:
            errors.append(ValidationError('', 'Missing required "default" field'))
        
        # Validate models
        models = config.get('models', {})
        for model_key, model_config in models.items():
            errors.extend(cls._validate_model(model_key, model_config))
        
        # Validate routes
        routes = config.get('routes', {})
        errors.extend(cls._validate_routes(routes, models))
        
        # Validate parameters
        parameters = config.get('parameters', {})
        errors.extend(cls._validate_parameters(parameters))
        
        # Validate default route exists
        default_route = config.get('default')
        if default_route and default_route not in models and default_route not in routes:
            errors.append(ValidationError(
                'default', 
                f'Default route "{default_route}" not found in models or routes'
            ))
        
        return errors
    
    @classmethod
    def _validate_model(cls, model_key: str, model_config: Dict[str, Any]) -> List[ValidationError]:
        """Validate a single model configuration."""
        errors = []
        path_prefix = f'models.{model_key}'
        
        if not isinstance(model_config, dict):
            errors.append(ValidationError(
                path_prefix, 
                f'Model configuration must be a dictionary, got {type(model_config)}'
            ))
            return errors
        
        # Check required fields
        for field in cls.REQUIRED_MODEL_FIELDS:
            if field not in model_config:
                errors.append(ValidationError(
                    f'{path_prefix}.{field}', 
                    f'Missing required field "{field}"'
                ))
        
        # Validate backend path format
        backend = model_config.get('backend', '')
        if backend and ':' not in backend:
            errors.append(ValidationError(
                f'{path_prefix}.backend', 
                f'Backend path "{backend}" must be in format "module.path:ClassName"'
            ))
        
        # Validate HTTP-specific configurations
        if 'base_url' in model_config:
            errors.extend(cls._validate_http_config(path_prefix, model_config))
        
        return errors
    
    @classmethod
    def _validate_http_config(cls, path_prefix: str, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate HTTP-specific configuration fields."""
        errors = []
        
        # Validate HTTP method
        method = config.get('method', 'POST')
        if method.upper() not in cls.VALID_HTTP_METHODS:
            errors.append(ValidationError(
                f'{path_prefix}.method', 
                f'Invalid HTTP method "{method}". Must be one of {cls.VALID_HTTP_METHODS}'
            ))
        
        # Validate auth style
        auth_style = config.get('auth_style')
        if auth_style and auth_style not in cls.VALID_AUTH_STYLES:
            errors.append(ValidationError(
                f'{path_prefix}.auth_style', 
                f'Invalid auth style "{auth_style}". Must be one of {cls.VALID_AUTH_STYLES}'
            ))
        
        # Validate required auth fields
        if auth_style == 'header_key' and 'auth_header_name' not in config:
            errors.append(ValidationError(
                f'{path_prefix}.auth_header_name', 
                'auth_header_name is required when auth_style is "header_key"'
            ))
        
        # Validate timeout
        timeout = config.get('timeout')
        if timeout is not None:
            try:
                timeout_val = float(timeout)
                if timeout_val <= 0:
                    errors.append(ValidationError(
                        f'{path_prefix}.timeout', 
                        'timeout must be a positive number'
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f'{path_prefix}.timeout', 
                    'timeout must be a number'
                ))
        
        # Validate JSONPath (basic check)
        response_path = config.get('response_text_path')
        if response_path and not response_path.startswith('$'):
            errors.append(ValidationError(
                f'{path_prefix}.response_text_path', 
                'response_text_path should be a valid JSONPath starting with "$"',
                severity='warning'
            ))
        
        return errors
    
    @classmethod
    def _validate_routes(cls, routes: Dict[str, str], models: Dict[str, Any]) -> List[ValidationError]:
        """Validate route mappings."""
        errors = []
        
        for route_name, model_key in routes.items():
            if model_key not in models:
                errors.append(ValidationError(
                    f'routes.{route_name}', 
                    f'Route "{route_name}" points to undefined model "{model_key}"'
                ))
        
        return errors
    
    @classmethod
    def _validate_parameters(cls, parameters: Dict[str, Any]) -> List[ValidationError]:
        """Validate parameter configuration."""
        errors = []
        
        # Check parameter structure
        valid_sections = {'defaults', 'by_stage', 'by_role', 'dynamic_rules'}
        for section in parameters:
            if section not in valid_sections:
                errors.append(ValidationError(
                    f'parameters.{section}', 
                    f'Unknown parameter section "{section}". Valid sections: {valid_sections}',
                    severity='warning'
                ))
        
        # Validate defaults
        defaults = parameters.get('defaults', {})
        errors.extend(cls._validate_parameter_values('parameters.defaults', defaults))
        
        # Validate by_stage
        by_stage = parameters.get('by_stage', {})
        for stage_name, stage_params in by_stage.items():
            errors.extend(cls._validate_parameter_values(
                f'parameters.by_stage.{stage_name}', 
                stage_params
            ))
        
        # Validate by_role
        by_role = parameters.get('by_role', {})
        for role_name, role_params in by_role.items():
            errors.extend(cls._validate_parameter_values(
                f'parameters.by_role.{role_name}', 
                role_params
            ))
        
        return errors
    
    @classmethod
    def _validate_parameter_values(cls, path_prefix: str, params: Dict[str, Any]) -> List[ValidationError]:
        """Validate individual parameter values."""
        errors = []
        
        # Validate temperature
        temp = params.get('temperature')
        if temp is not None:
            try:
                temp_val = float(temp)
                if not (0.0 <= temp_val <= 2.0):
                    errors.append(ValidationError(
                        f'{path_prefix}.temperature', 
                        'temperature should be between 0.0 and 2.0',
                        severity='warning'
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f'{path_prefix}.temperature', 
                    'temperature must be a number'
                ))
        
        # Validate max_tokens
        max_tokens = params.get('max_tokens')
        if max_tokens is not None:
            try:
                tokens_val = int(max_tokens)
                if tokens_val <= 0:
                    errors.append(ValidationError(
                        f'{path_prefix}.max_tokens', 
                        'max_tokens must be a positive integer'
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f'{path_prefix}.max_tokens', 
                    'max_tokens must be an integer'
                ))
        
        # Validate top_p
        top_p = params.get('top_p')
        if top_p is not None:
            try:
                top_p_val = float(top_p)
                if not (0.0 <= top_p_val <= 1.0):
                    errors.append(ValidationError(
                        f'{path_prefix}.top_p', 
                        'top_p should be between 0.0 and 1.0',
                        severity='warning'
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f'{path_prefix}.top_p', 
                    'top_p must be a number'
                ))
        
        return errors

def validate_and_log_config(config: Dict[str, Any], 
                           logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate configuration and log results.
    Returns True if validation passes (no errors), False otherwise.
    """
    if not logger_instance:
        logger_instance = logger
    
    errors = LLMConfigValidator.validate_config(config)
    
    if not errors:
        logger_instance.info("LLM configuration validation passed")
        return True
    
    # Separate errors and warnings
    validation_errors = [e for e in errors if e.severity == 'error']
    warnings = [e for e in errors if e.severity == 'warning']
    
    # Log warnings
    for warning in warnings:
        logger_instance.warning(f"Config validation warning at {warning.path}: {warning.message}")
    
    # Log errors
    for error in validation_errors:
        logger_instance.error(f"Config validation error at {error.path}: {error.message}")
    
    if validation_errors:
        logger_instance.error(f"LLM configuration validation failed with {len(validation_errors)} errors")
        return False
    else:
        logger_instance.info(f"LLM configuration validation passed with {len(warnings)} warnings")
        return True