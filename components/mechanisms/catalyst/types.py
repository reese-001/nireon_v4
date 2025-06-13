from typing import Dict, Any, List, Optional, Tuple, Union, TypeVar
import numpy as np
T = TypeVar('T')
DomainVector = np.ndarray
DomainVectors = Dict[str, DomainVector]
BlendRange = Tuple[float, float]
AntiConstraints = List[str]
DEFAULT_APPLICATION_RATE = 0.1
DEFAULT_BLEND_LOW = 0.1
DEFAULT_BLEND_HIGH = 0.3
DEFAULT_DUPLICATION_COOLDOWN = 5
DEFAULT_DUPLICATION_AGGRESSIVENESS = 0.5
DEFAULT_MAX_BLEND_LOW = 0.8
DEFAULT_MAX_BLEND_HIGH = 0.95
DEFAULT_ANTI_CONSTRAINTS_COUNT = 3
DEFAULT_ANTI_CONSTRAINTS_THRESHOLD = 0.15
DEFAULT_PROMPT_TEMPLATE = None
MIN_BLEND_GAP = 0.01