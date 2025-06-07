"""
OpenAI LLM Adapter for NIREON V4
"""
import logging
import os
import httpx
import json
from typing import Any, Dict, Optional
from domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class OpenAILLMAdapter(LLMPort):
    """
    OpenAI API adapter that implements the LLMPort interface.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration
        self.api_key_env = self.config.get('api_key_env', 'OPENAI_API_KEY')
        self.model = self.config.get('model', 'gpt-4')
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
        
        # Get API key from environment
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            logger.warning(f"OpenAI API key not found in environment variable '{self.api_key_env}'. "
                         f"LLM functionality will be limited.")
        
        # HTTP client
        self.client = httpx.Client(
            timeout=self.timeout,
            headers={
                'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
                'Content-Type': 'application/json'
            }
        )
        
        self.call_count = 0
        logger.info(f'OpenAI LLM Adapter initialized (model: {self.model})')
    
    def call_llm(self, prompt: str, **kwargs) -> str:
        """Synchronous LLM call."""
        self.call_count += 1
        
        if not self.api_key:
            logger.warning(f"No API key available, returning mock response for call #{self.call_count}")
            return f"Mock OpenAI response to: {prompt[:50]}..."
        
        try:
            # Prepare the request
            system_prompt = kwargs.get('system_prompt', 'You are a helpful assistant.')
            temperature = kwargs.get('temperature', 0.7)
            max_tokens = kwargs.get('max_tokens', 1024)
            
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            # Make the request
            response = self.client.post(
                f'{self.base_url}/chat/completions',
                json=payload
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            logger.debug(f"OpenAI API call #{self.call_count} successful")
            return content
            
        except Exception as e:
            logger.error(f"OpenAI API call #{self.call_count} failed: {e}")
            return f"Error calling OpenAI API: {e}"
    
    async def call_llm_async(self, prompt: str, **kwargs) -> str:
        """Asynchronous LLM call."""
        self.call_count += 1
        
        if not self.api_key:
            logger.warning(f"No API key available, returning mock response for async call #{self.call_count}")
            return f"Mock async OpenAI response to: {prompt[:50]}..."
        
        try:
            # Prepare the request
            system_prompt = kwargs.get('system_prompt', 'You are a helpful assistant.')
            temperature = kwargs.get('temperature', 0.7)
            max_tokens = kwargs.get('max_tokens', 1024)
            
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            # Make async request
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            ) as client:
                response = await client.post(
                    f'{self.base_url}/chat/completions',
                    json=payload
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                logger.debug(f"OpenAI async API call #{self.call_count} successful")
                return content
                
        except Exception as e:
            logger.error(f"OpenAI async API call #{self.call_count} failed: {e}")
            return f"Error calling OpenAI API async: {e}"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM."""
        return self.call_llm(prompt, **kwargs)
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM asynchronously."""
        return await self.call_llm_async(prompt, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            'call_count': self.call_count,
            'model': self.model,
            'has_api_key': bool(self.api_key),
            'base_url': self.base_url
        }