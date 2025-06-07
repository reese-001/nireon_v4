# nireon_v4/infrastructure/llm/generic_http.py
import json
import logging
import os
from typing import Any, Dict, Optional, Mapping
from string import Template
import httpx

# Import jsonpath with fallback
try:
    from jsonpath_ng import parse as jsonpath_parse
    JSONPATH_AVAILABLE = True
except ImportError:
    JSONPATH_AVAILABLE = False
    jsonpath_parse = None

from domain.ports.llm_port import LLMPort, LLMResponse
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage

logger = logging.getLogger(__name__)

class GenericHttpLLM(LLMPort):
    """
    Generic HTTP-based LLM adapter that can work with various API providers
    by using configurable templates and response parsing.
    """
    
    def __init__(self, config: Dict[str, Any] = None, model_name: Optional[str] = None, **kwargs):
        # Merge config with any additional kwargs passed by the factory
        self.config = config or {}
        self.config.update(kwargs)  # Add any extra parameters from factory
        
        # IMPORTANT: For API calls, we need the model_name_for_api, not the internal model_name
        # The model_name parameter from factory is the internal key (e.g., "nano_default")
        # But we need the actual API model name (e.g., "gpt-4o-mini")
        self.internal_model_name = model_name or 'default'
        self.model_name = self.config.get('model_name_for_api', model_name or 'default')
        
        logger.info(f"GenericHttpLLM init: internal_name='{self.internal_model_name}', api_name='{self.model_name}'")
        
        # HTTP configuration
        self.method = self.config.get('method', 'POST').upper()
        self.base_url = self.config.get('base_url', '')
        self.endpoint = self.config.get('endpoint', '')
        self.timeout = self.config.get('timeout', 30)
        
        # Authentication configuration
        self.auth_style = self.config.get('auth_style', 'bearer')  # bearer, header_key, query_param, none
        self.auth_token_env = self.config.get('auth_token_env', '')
        self.auth_header_name = self.config.get('auth_header_name', 'Authorization')
        
        # Payload and response configuration
        self.payload_template_str = self.config.get('payload_template', '{}')
        self.response_text_path = self.config.get('response_text_path', '$.text')
        
        # Get authentication token (only if not using 'none' auth)
        self.auth_token = None
        if self.auth_style != 'none' and self.auth_token_env:
            self.auth_token = os.getenv(self.auth_token_env)
            if not self.auth_token:
                logger.warning(f"Auth token not found in environment variable '{self.auth_token_env}'")
        
        # Parse JSONPath for response extraction
        self.response_parser = None
        if JSONPATH_AVAILABLE:
            try:
                self.response_parser = jsonpath_parse(self.response_text_path)
            except Exception as e:
                logger.error(f"Invalid JSONPath '{self.response_text_path}': {e}")
                self.response_parser = None
        else:
            logger.warning("jsonpath_ng not available - using fallback response parsing")
            logger.info("Install with: pip install jsonpath-ng")
            
        self.call_count = 0
        logger.info(f'Generic HTTP LLM Adapter initialized (model: {self.model_name}, endpoint: {self.base_url}{self.endpoint})')
    
    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers including authentication."""
        headers = {'Content-Type': 'application/json'}
        
        if self.auth_style == 'none':
            # No authentication headers
            logger.info("No authentication (auth_style=none)")
        elif self.auth_token:
            if self.auth_style == 'bearer':
                headers['Authorization'] = f'Bearer {self.auth_token}'
                logger.info(f"Using Bearer auth with token: {self.auth_token[:8]}...")
            elif self.auth_style == 'header_key':
                headers[self.auth_header_name] = self.auth_token
                logger.info(f"Using header auth ({self.auth_header_name}) with token: {self.auth_token[:8]}...")
        else:
            logger.warning(f"Auth style '{self.auth_style}' specified but no token available")
        
        return headers
    
    def _build_payload(self, prompt: str, stage: EpistemicStage, role: str, 
                      context: NireonExecutionContext, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Build the request payload using the configured template."""
        try:
            # Prepare template variables
            template_vars = {
                'prompt': prompt,
                'role': role,
                'stage': stage.value if isinstance(stage, EpistemicStage) else str(stage),
                'model_name_for_api': self.model_name,  # This should be the API model name, not the internal key
                'system_prompt': settings.get('system_prompt', 'You are a helpful assistant.'),
                'temperature': settings.get('temperature', 0.7),
                'max_tokens': settings.get('max_tokens', 1024),
                'top_p': settings.get('top_p', 1.0),
                **settings  # Include all other settings
            }
            
            # Debug: Check what model_name we're actually using
            logger.info(f"Internal model name (self.model_name): {self.model_name}")
            logger.info(f"Template model_name_for_api: {template_vars['model_name_for_api']}")
            
            # Log the template and variables for debugging
            logger.debug(f"Template: {self.payload_template_str}")
            logger.debug(f"Variables: {template_vars}")
            
            # Convert {{ variable }} syntax to $variable syntax for Python Template
            template_str = self.payload_template_str
            
            # Replace {{ variable }} with $variable
            import re
            def replace_braces(match):
                var_name = match.group(1).strip()
                return f"${var_name}"
            
            template_str = re.sub(r'\{\{\s*(\w+)\s*\}\}', replace_braces, template_str)
            logger.debug(f"Converted template: {template_str}")
            
            # Use Template for substitution, then parse as JSON
            template = Template(template_str)
            payload_str = template.safe_substitute(**template_vars)
            
            # Log the final payload string before JSON parsing
            logger.debug(f"Generated payload string: {payload_str}")
            
            # Parse the resulting JSON
            payload = json.loads(payload_str)
            return payload
            
        except Exception as e:
            logger.error(f"Error building payload from template: {e}")
            logger.error(f"Template was: {self.payload_template_str}")
            logger.error(f"Variables were: {template_vars}")
            
            # Fallback to a basic payload that we know works
            fallback_payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": settings.get('system_prompt', 'You are a helpful assistant.')},
                    {"role": "user", "content": prompt}
                ],
                "temperature": settings.get('temperature', 0.7),
                "max_tokens": settings.get('max_tokens', 1024)
            }
            logger.info(f"Using fallback payload: {fallback_payload}")
            return fallback_payload
    
    def _extract_response_text(self, response_data: Dict[str, Any]) -> str:
        """Extract text from API response using configured JSONPath."""
        if not JSONPATH_AVAILABLE or not self.response_parser:
            # Fallback: try common response structures
            if 'choices' in response_data and response_data['choices']:
                # OpenAI-style response
                choice = response_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    return choice['message']['content']
                elif 'text' in choice:
                    return choice['text']
            elif 'candidates' in response_data and response_data['candidates']:
                # Gemini-style response
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if parts and 'text' in parts[0]:
                        return parts[0]['text']
            elif 'text' in response_data:
                return response_data['text']
            elif 'content' in response_data:
                return response_data['content']
            else:
                return str(response_data)
        
        try:
            matches = self.response_parser.find(response_data)
            if matches:
                return str(matches[0].value)
            else:
                logger.warning(f"JSONPath '{self.response_text_path}' found no matches in response")
                return "No response text found"
        except Exception as e:
            logger.error(f"Error extracting response text: {e}")
            return f"Error extracting response: {e}"
    
    async def call_llm_async(self, prompt: str, *, stage: EpistemicStage, role: str, 
                            context: NireonExecutionContext, 
                            settings: Optional[Mapping[str, Any]] = None) -> LLMResponse:
        self.call_count += 1
        final_settings = dict(settings or {})
        
        # Only check for auth token if auth is required
        if self.auth_style != 'none' and not self.auth_token:
            logger.warning(f'No auth token available, returning mock response for async call #{self.call_count}')
            return LLMResponse({
                LLMResponse.TEXT_KEY: f'Mock async response to: {prompt[:50]}...',
                'error': 'NoAuthToken'
            })
        
        url = f"{self.base_url.rstrip('/')}{self.endpoint}"
        headers = self._build_headers()
        payload = self._build_payload(prompt, stage, role, context, final_settings)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Log the full request details
                logger.info(f"Making {self.method} request to: {url}")
                logger.info(f"Headers: {headers}")
                logger.info(f"Payload: {json.dumps(payload, indent=2)}")
                
                if self.method == 'POST':
                    response = await client.post(url, json=payload, headers=headers)
                elif self.method == 'GET':
                    response = await client.get(url, params=payload, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {self.method}")
                
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {dict(response.headers)}")
                
                response.raise_for_status()
                result = response.json()
                
                # Extract text content
                content = self._extract_response_text(result)
                
                logger.debug(f'Generic HTTP LLM async call #{self.call_count} successful')
                return LLMResponse({
                    LLMResponse.TEXT_KEY: content,
                    'raw_response': result,
                    'model': self.model_name,
                    'provider': 'generic_http'
                })
                
        except Exception as e:
            logger.error(f'Generic HTTP LLM async call #{self.call_count} failed: {e}')
            return LLMResponse({
                LLMResponse.TEXT_KEY: f'Error calling Generic HTTP LLM API async: {e}',
                'error': str(e),
                'model': self.model_name
            })
    
    def call_llm_sync(self, prompt: str, *, stage: EpistemicStage, role: str, 
                     context: NireonExecutionContext, 
                     settings: Optional[Mapping[str, Any]] = None) -> LLMResponse:
        self.call_count += 1
        final_settings = dict(settings or {})
        
        # Only check for auth token if auth is required
        if self.auth_style != 'none' and not self.auth_token:
            logger.warning(f'No auth token available, returning mock response for sync call #{self.call_count}')
            return LLMResponse({
                LLMResponse.TEXT_KEY: f'Mock sync response to: {prompt[:50]}...',
                'error': 'NoAuthToken'
            })
        
        url = f"{self.base_url.rstrip('/')}{self.endpoint}"
        headers = self._build_headers()
        payload = self._build_payload(prompt, stage, role, context, final_settings)
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                if self.method == 'POST':
                    response = client.post(url, json=payload, headers=headers)
                elif self.method == 'GET':
                    response = client.get(url, params=payload, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {self.method}")
                
                response.raise_for_status()
                result = response.json()
                
                # Extract text content
                content = self._extract_response_text(result)
                
                logger.debug(f'Generic HTTP LLM sync call #{self.call_count} successful')
                return LLMResponse({
                    LLMResponse.TEXT_KEY: content,
                    'raw_response': result,
                    'model': self.model_name,
                    'provider': 'generic_http'
                })
                
        except Exception as e:
            logger.error(f'Generic HTTP LLM sync call #{self.call_count} failed: {e}')
            return LLMResponse({
                LLMResponse.TEXT_KEY: f'Error calling Generic HTTP LLM API sync: {e}',
                'error': str(e),
                'model': self.model_name
            })
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'call_count': self.call_count,
            'model': self.model_name,
            'base_url': self.base_url,
            'endpoint': self.endpoint,
            'has_auth_token': bool(self.auth_token),
            'auth_style': self.auth_style
        }