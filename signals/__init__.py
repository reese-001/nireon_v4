from __future__ import annotations
import importlib
import inspect
import pkgutil
from typing import Dict, Type, List

# FIX: Explicitly import the modules containing the signal definitions.
# This ensures that all subclasses of EpistemicSignal are loaded into memory
# before we try to discover them.
from . import base
from . import core

from signals.base import EpistemicSignal


def get_all_subclasses(cls: Type) -> List[Type]:
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses

# Now that all signals are imported, this class map will be populated correctly.
signal_class_map: Dict[str, Type[EpistemicSignal]] = {cls.__name__: cls for cls in get_all_subclasses(EpistemicSignal) if not inspect.isabstract(cls)}
signal_class_map['EpistemicSignal'] = EpistemicSignal

globals().update(signal_class_map)
__all__ = list(signal_class_map.keys()) + ['signal_class_map', 'EpistemicSignal']