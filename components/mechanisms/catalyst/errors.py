# nireon_v4/components/mechanisms/catalyst/errors.py

class CatalystError(Exception):
    """Base exception for the Catalyst mechanism."""
    pass

class VectorBlendError(CatalystError):
    """Error during vector blending operations."""
    pass

class CatalystLLMError(CatalystError):
    """Error related to LLM interactions, often wrapping a lower-level error."""
    pass

class DuplicationError(CatalystError):
    """Error related to duplication detection logic."""
    pass

class AntiConstraintError(CatalystError):
    """Error related to anti-constraint application."""
    pass