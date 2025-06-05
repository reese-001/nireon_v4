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

# Note: FeatureFlagsManager has been moved to infrastructure/feature_flags.py
# This file now only contains component result classes