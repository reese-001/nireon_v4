"""
Explorer Mechanism for NIREON V4

The Explorer mechanism generates idea variations through systematic exploration
and mutation strategies. It supports multiple exploration approaches including
depth-first, breadth-first, and random exploration.

Key Components:
- ExplorerMechanism: Main mechanism class
- ExplorerConfig: Pydantic configuration model
- EXPLORER_METADATA: Canonical metadata definition
"""

from .service import ExplorerMechanism, EXPLORER_METADATA
from .config import ExplorerConfig

__all__ = [
    'ExplorerMechanism',
    'EXPLORER_METADATA', 
    'ExplorerConfig'
]

__version__ = '0.1.0'
__description__ = 'Explorer mechanism for systematic idea generation and variation'