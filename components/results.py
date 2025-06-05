import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class ProcessResult:
    """Result of a component processing operation."""
    success: bool
    component_id: str
    message: str = ""
    output_data: Any = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")

@dataclass
class AnalysisResult:
    """Result of a component analysis operation."""
    success: bool
    component_id: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    message: str = ""
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

class SignalType(Enum):
    """Types of system signals."""
    INFORMATION = "information"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    STATUS_UPDATE = "status_update"
    ADAPTATION_REQUEST = "adaptation_request"

@dataclass
class SystemSignal:
    """A signal emitted by a component to communicate with the system."""
    signal_type: SignalType
    component_id: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not isinstance(self.signal_type, SignalType):
            raise TypeError("signal_type must be a SignalType enum")
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")

class AdaptationActionType(Enum):
    """Types of adaptation actions."""
    CONFIG_UPDATE = "config_update"
    PARAMETER_ADJUST = "parameter_adjust"
    BEHAVIOR_CHANGE = "behavior_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    COMPONENT_RESTART = "component_restart"
    SYSTEM_RECONFIGURE = "system_reconfigure"

@dataclass
class AdaptationAction:
    """An action that should be taken to adapt the system."""
    action_type: AdaptationActionType
    component_id: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority
    estimated_impact: str = "low"  # low, medium, high
    requires_approval: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not isinstance(self.action_type, AdaptationActionType):
            raise TypeError("action_type must be an AdaptationActionType enum")
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")
        if self.estimated_impact not in ["low", "medium", "high"]:
            raise ValueError("estimated_impact must be 'low', 'medium', or 'high'")

@dataclass
class ComponentHealth:
    """Health status of a component."""
    component_id: str
    status: str  # HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checks_performed: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")
        valid_statuses = ["HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN"]
        if self.status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
    
    def is_healthy(self) -> bool:
        """Check if the component is in a healthy state."""
        return self.status == "HEALTHY"
    
    def is_operational(self) -> bool:
        """Check if the component is operational (healthy or degraded but functional)."""
        return self.status in ["HEALTHY", "DEGRADED"]

# Legacy FeatureFlagsManager class (keeping existing functionality)
class FeatureFlagsManager:
    def __init__(self, config: Dict[str, Any]=None):
        self._flags: Dict[str, bool] = {}
        self._descriptions: Dict[str, str] = {}
        self._registered_flags: Set[str] = set()
        
        if config:
            for flag_name, value in config.items():
                if isinstance(value, bool):
                    self._flags[flag_name] = value
                elif isinstance(value, dict) and 'enabled' in value:
                    self._flags[flag_name] = bool(value['enabled'])
                    if 'description' in value:
                        self._descriptions[flag_name] = str(value['description'])
        
        # Register default flags
        self.register_flag('sentinel_enable_progression_adjustment', 
                          default_value=False, 
                          description='Enables progression bonus in Sentinel mechanism')
        self.register_flag('sentinel_enable_edge_trust', 
                          default_value=False, 
                          description='Enables edge trust calculations in Sentinel (if Idea supports graph structure)')
        self.register_flag('enable_exploration', 
                          default_value=True, 
                          description='Enables the Explorer mechanism to generate idea variations')
        self.register_flag('enable_catalyst', 
                          default_value=True, 
                          description='Enables the Catalyst mechanism for cross-domain idea blending')
        self.register_flag('catalyst_anti_constraints', 
                          default_value=False, 
                          description='Enables anti-constraint functionality in Catalyst')
        self.register_flag('catalyst_duplication_check', 
                          default_value=False, 
                          description='Enables duplication detection and adaptation in Catalyst')
        
        logger.info(f'FeatureFlagsManager initialized with {len(self._flags)} flags from config and defaults.')

    def register_flag(self, flag_name: str, default_value: bool = False, description: Optional[str] = None) -> None:
        self._registered_flags.add(flag_name)
        if flag_name not in self._flags:
            self._flags[flag_name] = default_value
        if description:
            self._descriptions[flag_name] = description
        logger.debug(f'Registered feature flag: {flag_name} (default: {default_value})')

    def is_enabled(self, flag_name: str, default: Optional[bool] = None) -> bool:
        if flag_name in self._flags:
            return self._flags[flag_name]
        if default is not None:
            logger.warning(f'Unregistered feature flag: {flag_name}, using provided default: {default}')
            return default
        logger.warning(f'Unregistered feature flag: {flag_name}, defaulting to False')
        return False

    def set_flag(self, flag_name: str, value: bool) -> None:
        if flag_name not in self._registered_flags:
            logger.warning(f'Setting unregistered feature flag: {flag_name}')
            self._registered_flags.add(flag_name)
        self._flags[flag_name] = bool(value)
        logger.info(f'Feature flag {flag_name} set to {value}')

    def get_all_flags(self) -> Dict[str, bool]:
        return dict(self._flags)

    def get_flag_description(self, flag_name: str) -> Optional[str]:
        return self._descriptions.get(flag_name)

    def get_registered_flags(self) -> List[Dict[str, Any]]:
        result = []
        for flag_name in sorted(self._registered_flags):
            flag_info = {
                'name': flag_name,
                'enabled': self._flags.get(flag_name, False)
            }
            if flag_name in self._descriptions:
                flag_info['description'] = self._descriptions[flag_name]
            result.append(flag_info)
        return result

def register_flag(flag_name: str, default_value: bool = False, description: Optional[str] = None) -> None:
    """Utility function for flag registration (legacy compatibility)."""
    logger.info(f'Feature flag registration requested: {flag_name} (default: {default_value})')
    logger.info(f'Description: {description}')