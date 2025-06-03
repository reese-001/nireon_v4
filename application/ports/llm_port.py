from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """
    Protocol defining the interface for LLM providers.
    Implementations should support both the original methods (call_llm)
    and the documented aliases (generate) for backward compatibility.
    """
    def call_llm(self, prompt: str, **kwargs) -> str:
        """
        Call the language model with a prompt
        
        Args:
            prompt: The text prompt to send to the model
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            str: The generated completion
        """
        ...
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Alias for call_llm to match documentation
        
        Args:
            prompt: The text prompt to send to the model
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            str: The generated completion
        """
        return self.call_llm(prompt, **kwargs)

    async def call_llm_async(self, prompt: str, **kwargs) -> str:
        """
        Asynchronously call the language model with a prompt
        
        Args:
            prompt: The text prompt to send to the model
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            str: The generated completion
        """
        ...
        
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """
        Alias for call_llm_async to match documentation
        
        Args:
            prompt: The text prompt to send to the model
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            str: The generated completion
        """
        return await self.call_llm_async(prompt, **kwargs)