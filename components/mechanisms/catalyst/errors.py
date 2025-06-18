# nireon_v4/components/mechanisms/catalyst/errors.py
"""Custom exceptions for the Catalyst mechanism."""

class CatalystError(Exception):
    """Base exception for all Catalyst-related errors."""
    pass

class VectorBlendError(CatalystError):
    """Error during vector blending operations."""
    pass

class CatalystLLMError(CatalystError):
    """Error during LLM text regeneration."""
    pass

class DuplicationError(CatalystError):
    """Error related to duplication detection."""
    pass

class AntiConstraintError(CatalystError):
    """Error related to anti-constraint application."""
    pass