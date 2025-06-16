# nireon/infrastructure/llm/exceptions.py
"""
Enhanced exception hierarchy for LLM subsystem.
"""
from typing import Optional, Dict, Any

class LLMError(Exception):
    """Base exception for all LLM-related errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.details = details or {}
        self.cause = cause

class LLMConfigurationError(LLMError):
    """Raised when there are configuration issues."""
    pass

class LLMAuthenticationError(LLMError):
    """Raised when authentication fails."""
    pass

class LLMRateLimitError(LLMError):
    """Raised when rate limits are exceeded."""
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after

class LLMTimeoutError(LLMError):
    """Raised when requests timeout."""
    pass

class LLMQuotaExceededError(LLMError):
    """Raised when quota/billing limits are exceeded."""
    pass

class LLMProviderError(LLMError):
    """Raised when the LLM provider returns an error."""
    def __init__(self, message: str, status_code: Optional[int] = None, provider_response: Optional[Dict] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.provider_response = provider_response

class LLMBackendNotAvailableError(LLMError):
    """Raised when a requested backend is not available."""
    pass

# nireon/infrastructure/llm/enhanced_openai_llm.py
"""
Enhanced OpenAI LLM adapter with better error handling.
"""
import logging
import os
import httpx
from typing import Any, Dict, Optional, Mapping
from domain.ports.llm_port import LLMPort, LLMResponse
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage
from .exceptions import (
    LLMAuthenticationError, LLMConfigurationError, LLMRateLimitError,
    LLMTimeoutError, LLMQuotaExceededError, LLMProviderError
)

logger = logging.getLogger(__name__)

class EnhancedOpenAILLMAdapter(LLMPort):
    """Enhanced OpenAI adapter with comprehensive error handling."""
    
    def __init__(self, config: Dict[str, Any] = None, model_name: Optional[str] = None):
        self.config = config or {}
        self.api_key_env = self.config.get('api_key_env', 'OPENAI_API_KEY')
        self.model = model_name or self.config.get('model', 'gpt-4')
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
        
        # Validate configuration
        self._validate_configuration()
        
        self.api_key = os.getenv(self.api_key_env) or self.config.get('api_key')
        
        if not self.api_key:
            raise LLMConfigurationError(
                f"OpenAI API key not found in environment variable '{self.api_key_env}' or config",
                details={'api_key_env': self.api_key_env, 'model': self.model}
            )
        
        self.client: Optional[httpx.Client] = None
        self.client = httpx.Client(
            timeout=self.timeout,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        )
        
        self.call_count = 0
        logger.info(f'Enhanced OpenAI LLM Adapter initialized (model: {self.model})')
    
    def _validate_configuration(self):
        """Validate adapter configuration."""
        required_fields = ['timeout', 'max_retries']
        
        for field in required_fields:
            if field not in self.config:
                continue
            
            value = self.config[field]
            if field in ['timeout', 'max_retries'] and (not isinstance(value, (int, float)) or value <= 0):
                raise LLMConfigurationError(
                    f"Configuration field '{field}' must be a positive number, got: {value}",
                    details={'field': field, 'value': value}
                )
    
    def _handle_http_error(self, response: httpx.Response, call_type: str) -> LLMResponse:
        """Convert HTTP errors to appropriate LLM exceptions and responses."""
        status_code = response.status_code
        
        try:
            error_data = response.json()
        except:
            error_data = {'error': {'message': response.text}}
        
        error_message = error_data.get('error', {}).get('message', 'Unknown error')
        
        # Handle specific error types
        if status_code == 401:
            error = LLMAuthenticationError(
                f"Authentication failed: {error_message}",
                details={'status_code': status_code, 'response': error_data}
            )
        elif status_code == 429:
            retry_after = response.headers.get('retry-after')
            error = LLMRateLimitError(
                f"Rate limit exceeded: {error_message}",
                retry_after=int(retry_after) if retry_after else None,
                details={'status_code': status_code, 'response': error_data}
            )
        elif status_code == 402:
            error = LLMQuotaExceededError(
                f"Quota exceeded: {error_message}",
                details={'status_code': status_code, 'response': error_data}
            )
        elif 400 <= status_code < 500:
            error = LLMProviderError(
                f"Client error ({status_code}): {error_message}",
                status_code=status_code,
                provider_response=error_data
            )
        elif 500 <= status_code < 600:
            error = LLMProviderError(
                f"Server error ({status_code}): {error_message}",
                status_code=status_code,
                provider_response=error_data
            )
        else:
            error = LLMProviderError(
                f"HTTP error ({status_code}): {error_message}",
                status_code=status_code,
                provider_response=error_data
            )
        
        logger.error(f'OpenAI {call_type} call failed: {error}')
        
        return LLMResponse({
            LLMResponse.TEXT_KEY: f'Error: {error_message}',
            'error': str(error),
            'error_type': type(error).__name__,
            'status_code': status_code,
            'provider_response': error_data
        })
    
    async def call_llm_async(self, prompt: str, *, stage: EpistemicStage, role: str, 
                            context: NireonExecutionContext, 
                            settings: Optional[Mapping[str, Any]] = None) -> LLMResponse:
        self.call_count += 1
        final_settings = {'temperature': 0.7, 'max_tokens': 1024, **(settings or {})}
        system_prompt = final_settings.pop('system_prompt', 'You are a helpful assistant.')
        
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            **final_settings
        }
        
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            ) as async_client:
                response = await async_client.post(f'{self.base_url}/chat/completions', json=payload)
                
                if not response.is_success:
                    return self._handle_http_error(response, 'async')
                
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                logger.debug(f'OpenAI async API call #{self.call_count} successful for model {self.model}')
                return LLMResponse({
                    LLMResponse.TEXT_KEY: content,
                    **result
                })
                
        except httpx.TimeoutException as e:
            error = LLMTimeoutError(f"Request timed out after {self.timeout}s", cause=e)
            logger.error(f'OpenAI async call #{self.call_count} timed out: {error}')
            return LLMResponse({
                LLMResponse.TEXT_KEY: f'Error: Request timed out',
                'error': str(error),
                'error_type': 'LLMTimeoutError'
            })
        
        except Exception as e:
            logger.error(f'OpenAI async API call #{self.call_count} failed: {e}', exc_info=True)
            return LLMResponse({
                LLMResponse.TEXT_KEY: f'Error calling OpenAI API async: {e}',
                'error': str(e),
                'error_type': type(e).__name__
            })
    
    def call_llm_sync(self, prompt: str, *, stage: EpistemicStage, role: str, 
                     context: NireonExecutionContext, 
                     settings: Optional[Mapping[str, Any]] = None) -> LLMResponse:
        self.call_count += 1
        final_settings = {'temperature': 0.7, 'max_tokens': 1024, **(settings or {})}
        system_prompt = final_settings.pop('system_prompt', 'You are a helpful assistant.')
        
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            **final_settings
        }
        
        try:
            response = self.client.post(f'{self.base_url}/chat/completions', json=payload)
            
            if not response.is_success:
                return self._handle_http_error(response, 'sync')
            
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            logger.debug(f'OpenAI sync API call #{self.call_count} successful for model {self.model}')
            return LLMResponse({
                LLMResponse.TEXT_KEY: content,
                **result
            })
            
        except httpx.TimeoutException as e:
            error = LLMTimeoutError(f"Request timed out after {self.timeout}s", cause=e)
            logger.error(f'OpenAI sync call #{self.call_count} timed out: {error}')
            return LLMResponse({
                LLMResponse.TEXT_KEY: f'Error: Request timed out',
                'error': str(error),
                'error_type': 'LLMTimeoutError'
            })
        
        except Exception as e:
            logger.error(f'OpenAI sync API call #{self.call_count} failed: {e}', exc_info=True)
            return LLMResponse({
                LLMResponse.TEXT_KEY: f'Error calling OpenAI API sync: {e}',
                'error': str(e),
                'error_type': type(e).__name__
            })
    
    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except:
                pass  # Ignore cleanup errors