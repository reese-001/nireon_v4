# nireon_v4/components/mechanisms/catalyst/__init__.py
"""Catalyst Mechanism V4 - Cross-Domain Concept Injection for Creative Synthesis"""
from .service import CatalystMechanism
from .metadata import CATALYST_METADATA
from .config import CatalystMechanismConfig
from .errors import (
    CatalystError, 
    AntiConstraintError, 
    CatalystLLMError, 
    DuplicationError, 
    VectorBlendError
)

__all__ = [
    'CatalystMechanism',
    'CATALYST_METADATA',
    'CatalystMechanismConfig',
    'CatalystError',
    'VectorBlendError',
    'CatalystLLMError',
    'DuplicationError',
    'AntiConstraintError'
]