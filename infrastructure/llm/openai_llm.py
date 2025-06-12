# C:\Users\erees\Documents\development\nireon\infrastructure\llm\openai_llm.py
import logging
import os
import httpx
# import json # Not strictly needed if not manually creating JSON
from typing import Any, Dict, Optional, Mapping # Added Mapping
from domain.ports.llm_port import LLMPort, LLMResponse # MODIFIED
from domain.context import NireonExecutionContext # For type hinting if needed by call_llm_async
from domain.epistemic_stage import EpistemicStage # For type hinting

logger = logging.getLogger(__name__)

class OpenAILLMAdapter(LLMPort):
    def __init__(self, config: Dict[str, Any]=None, model_name: Optional[str] = None): # Added model_name to init
        self.config = config or {}
        self.api_key_env = self.config.get('api_key_env', 'OPENAI_API_KEY')
        self.model = model_name or self.config.get('model', 'gpt-4') # Use provided model_name or config
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
        self.api_key = os.getenv(self.api_key_env) or self.config.get('api_key') # Allow api_key from config

        if not self.api_key:
            logger.warning(f"OpenAI API key not found in environment variable '{self.api_key_env}' or config. LLM functionality will be limited.")
        
        # Client initialization deferred or handled carefully if api_key can be None
        self.client: Optional[httpx.Client] = None
        if self.api_key:
            self.client = httpx.Client(
                timeout=self.timeout,
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
            )
        self.call_count = 0
        logger.info(f'OpenAI LLM Adapter initialized (model: {self.model})')

    async def call_llm_async(
        self,
        prompt: str,
        *,
        stage: EpistemicStage, # Added from LLMPort
        role: str,            # Added from LLMPort
        context: NireonExecutionContext, # Added from LLMPort
        settings: Optional[Mapping[str, Any]] = None # Added from LLMPort
    ) -> LLMResponse:
        self.call_count += 1
        if not self.api_key or not self.client:
            logger.warning(f'No API key or client, returning mock response for async call #{self.call_count}')
            return LLMResponse({LLMResponse.TEXT_KEY: f'Mock async OpenAI response to: {prompt[:50]}...'})

        # Merge settings with defaults
        final_settings = {
            'temperature': 0.7,
            'max_tokens': 1024,
            **(settings or {})
        }
        system_prompt = final_settings.pop('system_prompt', 'You are a helpful assistant.')


        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            **final_settings # Add remaining settings like temperature, max_tokens
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
            ) as async_client:
                response = await async_client.post(f'{self.base_url}/chat/completions', json=payload)
                response.raise_for_status()
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                logger.debug(f'OpenAI async API call #{self.call_count} successful for model {self.model}')
                return LLMResponse({LLMResponse.TEXT_KEY: content, **result})
        except Exception as e:
            logger.error(f'OpenAI async API call #{self.call_count} for model {self.model} failed: {e}')
            return LLMResponse({LLMResponse.TEXT_KEY: f'Error calling OpenAI API async: {e}', 'error': str(e)})

    def call_llm_sync(
        self,
        prompt: str,
        *,
        stage: EpistemicStage,
        role: str,
        context: NireonExecutionContext,
        settings: Optional[Mapping[str, Any]] = None
    ) -> LLMResponse:
        self.call_count += 1
        if not self.api_key or not self.client:
            logger.warning(f'No API key or client, returning mock response for sync call #{self.call_count}')
            return LLMResponse({LLMResponse.TEXT_KEY: f'Mock sync OpenAI response to: {prompt[:50]}...'})

        final_settings = {
            'temperature': 0.7,
            'max_tokens': 1024,
            **(settings or {})
        }
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
            response.raise_for_status()
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            logger.debug(f'OpenAI sync API call #{self.call_count} successful for model {self.model}')
            return LLMResponse({LLMResponse.TEXT_KEY: content, **result})
        except Exception as e:
            logger.error(f'OpenAI sync API call #{self.call_count} for model {self.model} failed: {e}')
            return LLMResponse({LLMResponse.TEXT_KEY: f'Error calling OpenAI API sync: {e}', 'error': str(e)})
    
    # Keep these if they were part of your original design, but ensure they call the Protocol methods
    def generate(self, prompt: str, **kwargs) -> str:
        # This signature doesn't match LLMPort, consider deprecating or adapting
        # For now, let's assume it's a simplified call that needs context
        from domain.context import NireonExecutionContext # Local import if not always available
        from domain.epistemic_stage import EpistemicStage
        mock_context = NireonExecutionContext(run_id="sync_generate")
        response = self.call_llm_sync(prompt, stage=EpistemicStage.DEFAULT, role="default", context=mock_context, settings=kwargs)
        return response.text

    async def generate_async(self, prompt: str, **kwargs) -> str:
        # This signature doesn't match LLMPort, consider deprecating or adapting
        from domain.context import NireonExecutionContext # Local import
        from domain.epistemic_stage import EpistemicStage
        mock_context = NireonExecutionContext(run_id="async_generate")
        response = await self.call_llm_async(prompt, stage=EpistemicStage.DEFAULT, role="default", context=mock_context, settings=kwargs)
        return response.text

    def get_stats(self) -> Dict[str, Any]:
        return {'call_count': self.call_count, 'model': self.model, 'has_api_key': bool(self.api_key), 'base_url': self.base_url}