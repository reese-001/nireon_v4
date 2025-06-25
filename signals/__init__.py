# nireon_v4/signals/__init__.py
from __future__ import annotations
import inspect
from typing import Dict, Type, List

from .base import EpistemicSignal
from .core import (
    SeedSignal, LoopSignal, IdeaGeneratedSignal, TrustAssessmentSignal,
    StagnationDetectedSignal, ErrorSignal, GenerativeLoopFinishedSignal,
    MathQuerySignal, MathResultSignal, ProtoTaskSignal, ProtoResultSignal,
    ProtoErrorSignal, MathProtoResultSignal, PlanNextStepSignal, TraceEmittedSignal
)

def get_all_subclasses(cls: Type) -> List[Type]:
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses

# FIX: Use a private variable for the map.
_signal_class_map: Dict[str, Type[EpistemicSignal]] = {
    cls.__name__: cls for cls in get_all_subclasses(EpistemicSignal) 
    if not inspect.isabstract(cls)
}
_signal_class_map['EpistemicSignal'] = EpistemicSignal

# FIX: Define __all__ explicitly for clean imports.
__all__ = [
    'EpistemicSignal', 'SeedSignal', 'LoopSignal', 'IdeaGeneratedSignal',
    'TrustAssessmentSignal', 'StagnationDetectedSignal', 'ErrorSignal',
    'GenerativeLoopFinishedSignal', 'MathQuerySignal', 'MathResultSignal',
    'ProtoTaskSignal', 'ProtoResultSignal', 'ProtoErrorSignal',
    'MathProtoResultSignal', 'PlanNextStepSignal', 'TraceEmittedSignal',
    '_signal_class_map'
]

# Provide the old name for backwards compatibility but discourage its use.
signal_class_map = _signal_class_map