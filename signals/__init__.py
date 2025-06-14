# nireon_v4/signals/__init__.py

from .base import EpistemicSignal
from .core import (
    SeedSignal,
    LoopSignal,
    IdeaGeneratedSignal,      
    TrustAssessmentSignal,    
    StagnationDetectedSignal, 
    ErrorSignal               
)

# It's best practice to update __all__ to reflect the public API of the package.
__all__ = [
    'EpistemicSignal',
    'SeedSignal',
    'LoopSignal',
    'IdeaGeneratedSignal',
    'TrustAssessmentSignal',
    'StagnationDetectedSignal',
    'ErrorSignal'
]