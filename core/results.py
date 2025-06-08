import json
import logging
from typing import Any, Dict, List, Optional, Set, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import cached_property
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResultValidationError(ValueError):
    """Raised when result validation fails."""
    pass


@dataclass
class ProcessResult:
    """Enhanced result of a component processing operation."""
    success: bool
    component_id: str
    message: str = ""
    output_data: Any = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)  # New
    data: Any = None  # Backward compatibility alias for output_data
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Add BaseResult functionality as composition instead of inheritance
    _correlation_id: Optional[str] = field(default=None, init=False, repr=False)
    _parent_result_id: Optional[str] = field(default=None, init=False, repr=False)
    _child_results: List[str] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Handle backward compatibility
        if self.data is not None and self.output_data is None:
            self.output_data = self.data
        elif self.output_data is not None:
            self.data = self.output_data
    
    @property
    def result_id(self) -> str:
        """Generate a unique ID for this result."""
        return f"{self.__class__.__name__}_{self.component_id}_{self.timestamp.isoformat()}"
    
    @property
    def correlation_id(self) -> Optional[str]:
        """Get correlation ID for tracking related results."""
        return self._correlation_id
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for tracking related results across components."""
        self._correlation_id = correlation_id
    
    def link_parent(self, parent_result: 'ProcessResult') -> None:
        """Link this result as a child of another result."""
        self._parent_result_id = parent_result.result_id
        parent_result._child_results.append(self.result_id)
    
    @property
    def age(self) -> timedelta:
        """Get the age of this result."""
        return datetime.now(timezone.utc) - self.timestamp
    
    def is_recent(self, max_age_seconds: int = 300) -> bool:
        """Check if this result is recent (default: within 5 minutes)."""
        return self.age.total_seconds() <= max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        # Include computed properties
        data['result_id'] = self.result_id
        data['correlation_id'] = self._correlation_id
        data['parent_result_id'] = self._parent_result_id
        data['child_results'] = self._child_results
        # Handle enums
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessResult':
        """Create from dictionary (deserialization)."""
        # Convert ISO string back to datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Extract non-init fields
        correlation_id = data.pop('correlation_id', None)
        parent_result_id = data.pop('parent_result_id', None)
        child_results = data.pop('child_results', [])
        
        # Remove computed properties
        data.pop('result_id', None)
        
        # Create instance
        instance = cls(**data)
        
        # Restore non-init fields
        if correlation_id:
            instance._correlation_id = correlation_id
        if parent_result_id:
            instance._parent_result_id = parent_result_id
        instance._child_results = child_results
        
        return instance
    
    @property
    def failed(self) -> bool:
        """Check if the operation failed."""
        return not self.success
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a performance metric."""
        self.performance_metrics[name] = value
    
    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a performance metric."""
        return self.performance_metrics.get(name, default)
    
    def chain(self, next_component: Callable[[Any], 'ProcessResult']) -> 'ProcessResult':
        """Chain this result to another component if successful."""
        if not self.success:
            return self
        
        try:
            next_result = next_component(self.output_data)
            next_result.link_parent(self)
            if self._correlation_id:
                next_result.set_correlation_id(self._correlation_id)
            return next_result
        except Exception as e:
            # Create error result
            error_result = ProcessResult(
                success=False,
                component_id=f"chain_from_{self.component_id}",
                message=f"Chain operation failed: {e}",
                error_code="CHAIN_ERROR"
            )
            error_result.link_parent(self)
            return error_result
    
    def map_output(self, transform: Callable[[Any], Any]) -> 'ProcessResult':
        """Transform the output data if successful."""
        if not self.success:
            return self
        
        try:
            new_output = transform(self.output_data)
            return ProcessResult(
                success=True,
                component_id=self.component_id,
                message=f"Transformed: {self.message}",
                output_data=new_output,
                metadata=self.metadata.copy(),
                performance_metrics=self.performance_metrics.copy()
            )
        except Exception as e:
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f"Transform failed: {e}",
                error_code="TRANSFORM_ERROR"
            )
    
    @staticmethod
    def combine(results: List['ProcessResult']) -> 'ProcessResult':
        """Combine multiple results into a single result."""
        if not results:
            return ProcessResult(
                success=False,
                component_id="combined",
                message="No results to combine",
                error_code="NO_RESULTS"
            )
        
        all_success = all(r.success for r in results)
        combined_data = [r.output_data for r in results if r.output_data is not None]
        combined_metrics = {}
        
        for r in results:
            for metric, value in r.performance_metrics.items():
                if metric not in combined_metrics:
                    combined_metrics[metric] = []
                combined_metrics[metric].append(value)
        
        # Average the metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in combined_metrics.items()}
        
        return ProcessResult(
            success=all_success,
            component_id="combined",
            message=f"Combined {len(results)} results",
            output_data=combined_data if combined_data else None,
            performance_metrics=avg_metrics,
            metadata={"source_count": len(results), "success_count": sum(1 for r in results if r.success)}
        )


@dataclass
class AnalysisResult:
    """Enhanced result of a component analysis operation."""
    success: bool
    component_id: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    message: str = ""
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)  # New
    trends: Dict[str, str] = field(default_factory=dict)  # New: metric -> trend direction
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # BaseResult functionality
    _correlation_id: Optional[str] = field(default=None, init=False, repr=False)
    _parent_result_id: Optional[str] = field(default=None, init=False, repr=False)
    _child_results: List[str] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
    
    def add_insight(self, insight: str, confidence_impact: float = 0.0) -> None:
        """Add an insight and optionally adjust confidence."""
        self.insights.append(insight)
        if confidence_impact:
            self.confidence = max(0.0, min(1.0, self.confidence + confidence_impact))
    
    def add_anomaly(self, metric: str, value: Any, expected: Any, severity: str = "medium") -> None:
        """Record an anomaly in the analysis."""
        self.anomalies.append({
            "metric": metric,
            "value": value,
            "expected": expected,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def set_trend(self, metric: str, direction: str) -> None:
        """Set trend direction for a metric (up, down, stable)."""
        if direction not in ["up", "down", "stable"]:
            raise ValueError("Trend direction must be 'up', 'down', or 'stable'")
        self.trends[metric] = direction
    
    @property
    def has_anomalies(self) -> bool:
        """Check if any anomalies were detected."""
        return len(self.anomalies) > 0
    
    @property
    def high_confidence(self) -> bool:
        """Check if this is a high confidence analysis (>= 0.8)."""
        return self.confidence >= 0.8
    
    def merge_with(self, other: 'AnalysisResult') -> 'AnalysisResult':
        """Merge this analysis with another, combining insights and metrics."""
        merged_metrics = {**self.metrics, **other.metrics}
        merged_insights = list(set(self.insights + other.insights))
        merged_recommendations = list(set(self.recommendations + other.recommendations))
        merged_anomalies = self.anomalies + other.anomalies
        merged_trends = {**self.trends, **other.trends}
        
        # Weighted average confidence
        avg_confidence = (self.confidence + other.confidence) / 2
        
        return AnalysisResult(
            success=self.success and other.success,
            component_id=f"{self.component_id}+{other.component_id}",
            metrics=merged_metrics,
            confidence=avg_confidence,
            message=f"Merged analysis from {self.component_id} and {other.component_id}",
            insights=merged_insights,
            recommendations=merged_recommendations,
            anomalies=merged_anomalies,
            trends=merged_trends
        )


class SignalType(Enum):
    """Types of system signals."""
    INFORMATION = "information"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    STATUS_UPDATE = "status_update"
    ADAPTATION_REQUEST = "adaptation_request"
    METRIC_UPDATE = "metric_update"  # New
    HEALTH_CHECK = "health_check"    # New
    
    @property
    def severity_level(self) -> int:
        """Get numeric severity level (higher = more severe)."""
        severity_map = {
            self.INFORMATION: 1,
            self.STATUS_UPDATE: 1,
            self.METRIC_UPDATE: 1,
            self.HEALTH_CHECK: 2,
            self.WARNING: 3,
            self.ADAPTATION_REQUEST: 4,
            self.ERROR: 5,
            self.CRITICAL: 6
        }
        return severity_map.get(self, 0)
    
    def is_error_type(self) -> bool:
        """Check if this is an error-type signal."""
        return self in [self.ERROR, self.CRITICAL]


@dataclass
class SystemSignal:
    """Enhanced signal emitted by a component."""
    signal_type: SignalType
    component_id: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority
    ttl_seconds: Optional[int] = None  # New: Time to live
    requires_acknowledgment: bool = False  # New
    acknowledged: bool = False  # New
    acknowledged_by: Optional[str] = None  # New
    acknowledged_at: Optional[datetime] = None  # New
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # BaseResult functionality
    _correlation_id: Optional[str] = field(default=None, init=False, repr=False)
    _parent_result_id: Optional[str] = field(default=None, init=False, repr=False)
    _child_results: List[str] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        if not isinstance(self.signal_type, SignalType):
            raise TypeError("signal_type must be a SignalType enum")
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Auto-set priority based on signal type if not specified
        if self.priority == 0:
            self.priority = self.signal_type.severity_level
    
    @property
    def result_id(self) -> str:
        """Generate a unique ID for this signal."""
        return f"{self.__class__.__name__}_{self.component_id}_{self.timestamp.isoformat()}"
    
    @property
    def correlation_id(self) -> Optional[str]:
        """Get correlation ID for tracking related results."""
        return self._correlation_id
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for tracking related results across components."""
        self._correlation_id = correlation_id
    
    @property
    def age(self) -> timedelta:
        """Get the age of this signal."""
        return datetime.now(timezone.utc) - self.timestamp
    
    def is_recent(self, max_age_seconds: int = 300) -> bool:
        """Check if this signal is recent (default: within 5 minutes)."""
        return self.age.total_seconds() <= max_age_seconds
    
    @property
    def is_expired(self) -> bool:
        """Check if this signal has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return self.age.total_seconds() > self.ttl_seconds
    
    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge this signal."""
        self.acknowledged = True
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.now(timezone.utc)
    
    @property
    def needs_acknowledgment(self) -> bool:
        """Check if this signal still needs acknowledgment."""
        return self.requires_acknowledgment and not self.acknowledged
    
    def to_event(self) -> Dict[str, Any]:
        """Convert to event format for publishing."""
        return {
            "event_type": f"SIGNAL_{self.signal_type.value.upper()}",
            "component_id": self.component_id,
            "message": self.message,
            "payload": self.payload,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self._correlation_id
        }
    
    @classmethod
    def create_error(cls, component_id: str, message: str, error: Exception) -> 'SystemSignal':
        """Convenience method to create an error signal from an exception."""
        return cls(
            signal_type=SignalType.ERROR,
            component_id=component_id,
            message=message,
            payload={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": None  # Could add traceback if needed
            },
            priority=5,
            requires_acknowledgment=True
        )


class AdaptationActionType(Enum):
    """Types of adaptation actions."""
    CONFIG_UPDATE = "config_update"
    PARAMETER_ADJUST = "parameter_adjust"
    BEHAVIOR_CHANGE = "behavior_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    COMPONENT_RESTART = "component_restart"
    SYSTEM_RECONFIGURE = "system_reconfigure"
    THROTTLE_ADJUST = "throttle_adjust"  # New
    CACHE_CLEAR = "cache_clear"          # New
    FALLBACK_ACTIVATE = "fallback_activate"  # New


@dataclass
class AdaptationAction:
    """Enhanced action for system adaptation."""
    action_type: AdaptationActionType
    component_id: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority
    estimated_impact: str = "low"  # low, medium, high
    requires_approval: bool = False
    approved: bool = False  # New
    approved_by: Optional[str] = None  # New
    executed: bool = False  # New
    execution_result: Optional[ProcessResult] = None  # New
    rollback_action: Optional['AdaptationAction'] = None  # New
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not isinstance(self.action_type, AdaptationActionType):
            raise TypeError("action_type must be an AdaptationActionType enum")
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")
        if self.estimated_impact not in ["low", "medium", "high"]:
            raise ValueError("estimated_impact must be 'low', 'medium', or 'high'")
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
    
    def approve(self, approved_by: str) -> None:
        """Approve this adaptation action."""
        if not self.requires_approval:
            raise ValueError("This action does not require approval")
        self.approved = True
        self.approved_by = approved_by
    
    @property
    def can_execute(self) -> bool:
        """Check if this action can be executed."""
        if self.requires_approval and not self.approved:
            return False
        return not self.executed
    
    def set_execution_result(self, result: ProcessResult) -> None:
        """Set the result of executing this action."""
        self.executed = True
        self.execution_result = result
    
    def create_rollback(self) -> Optional['AdaptationAction']:
        """Create a rollback action if possible."""
        if self.action_type == AdaptationActionType.CONFIG_UPDATE:
            # Assuming parameters contain old_value and new_value
            if "old_value" in self.parameters:
                return AdaptationAction(
                    action_type=AdaptationActionType.CONFIG_UPDATE,
                    component_id=self.component_id,
                    description=f"Rollback: {self.description}",
                    parameters={
                        "config_key": self.parameters.get("config_key"),
                        "new_value": self.parameters.get("old_value"),
                        "old_value": self.parameters.get("new_value")
                    },
                    priority=self.priority + 1,  # Higher priority for rollback
                    estimated_impact=self.estimated_impact
                )
        return None


@dataclass
class ComponentHealth:
    """Enhanced health status of a component."""
    component_id: str
    status: str  # HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checks_performed: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)  # New: detailed health info
    dependencies_health: Dict[str, str] = field(default_factory=dict)  # New
    health_score: float = 1.0  # New: 0.0 to 1.0
    
    def __post_init__(self):
        if not isinstance(self.component_id, str):
            raise TypeError("component_id must be a string")
        valid_statuses = ["HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN", "INITIALIZING", "SHUTDOWN"]
        if self.status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        if self.last_check.tzinfo is None:
            self.last_check = self.last_check.replace(tzinfo=timezone.utc)
        
        # Calculate health score based on status
        status_scores = {
            "HEALTHY": 1.0,
            "DEGRADED": 0.6,
            "UNHEALTHY": 0.2,
            "UNKNOWN": 0.5,
            "INITIALIZING": 0.7,
            "SHUTDOWN": 0.0
        }
        self.health_score = status_scores.get(self.status, 0.5)
    
    def is_healthy(self) -> bool:
        """Check if the component is in a healthy state."""
        return self.status == "HEALTHY"
    
    def is_operational(self) -> bool:
        """Check if the component is operational (healthy or degraded but functional)."""
        return self.status in ["HEALTHY", "DEGRADED", "INITIALIZING"]
    
    def add_check(self, check_name: str, passed: bool, details: Optional[str] = None) -> None:
        """Add a health check result."""
        self.checks_performed.append(check_name)
        if not passed:
            issue = f"{check_name} failed"
            if details:
                issue += f": {details}"
            self.issues.append(issue)
    
    def add_dependency_health(self, dependency_id: str, status: str) -> None:
        """Track health of a dependency."""
        self.dependencies_health[dependency_id] = status
    
    @property
    def has_issues(self) -> bool:
        """Check if there are any health issues."""
        return len(self.issues) > 0
    
    @property
    def all_dependencies_healthy(self) -> bool:
        """Check if all dependencies are healthy."""
        return all(status == "HEALTHY" for status in self.dependencies_health.values())
    
    def calculate_overall_health(self) -> str:
        """Calculate overall health considering dependencies."""
        if self.status == "UNHEALTHY":
            return "UNHEALTHY"
        
        if not self.all_dependencies_healthy:
            return "DEGRADED"
        
        if self.has_issues and self.status == "HEALTHY":
            return "DEGRADED"
        
        return self.status
    
    def to_signal(self) -> SystemSignal:
        """Convert health status to a system signal if unhealthy."""
        if self.is_healthy():
            signal_type = SignalType.STATUS_UPDATE
        elif self.is_operational():
            signal_type = SignalType.WARNING
        else:
            signal_type = SignalType.ERROR
        
        return SystemSignal(
            signal_type=signal_type,
            component_id=self.component_id,
            message=f"Health: {self.status} - {self.message}",
            payload={
                "health_score": self.health_score,
                "issues": self.issues,
                "dependencies_health": self.dependencies_health
            },
            requires_acknowledgment=not self.is_operational()
        )


# Utility classes and functions

class ResultCollector:
    """Collect and analyze results from multiple components."""
    
    def __init__(self):
        self.results: List[Any] = []
        self._by_component: Dict[str, List[Any]] = defaultdict(list)
        self._by_type: Dict[Type[Any], List[Any]] = defaultdict(list)
    
    def add(self, result: Any) -> None:
        """Add a result to the collector."""
        self.results.append(result)
        if hasattr(result, 'component_id'):
            self._by_component[result.component_id].append(result)
        self._by_type[type(result)].append(result)
    
    def get_by_component(self, component_id: str) -> List[Any]:
        """Get all results for a specific component."""
        return self._by_component.get(component_id, [])
    
    def get_by_type(self, result_type: Type) -> List[Any]:
        """Get all results of a specific type."""
        return self._by_type.get(result_type, [])
    
    def get_failures(self) -> List[ProcessResult]:
        """Get all failed process results."""
        return [r for r in self.get_by_type(ProcessResult) if r.failed]
    
    def get_critical_signals(self) -> List[SystemSignal]:
        """Get all critical signals."""
        return [s for s in self.get_by_type(SystemSignal) 
                if s.signal_type == SignalType.CRITICAL]
    
    def get_health_summary(self) -> Dict[str, str]:
        """Get health summary for all components."""
        health_results = self.get_by_type(ComponentHealth)
        return {h.component_id: h.status for h in health_results}
    
    def get_recent(self, max_age_seconds: int = 300) -> List[Any]:
        """Get recent results."""
        recent = []
        for r in self.results:
            if hasattr(r, 'is_recent') and callable(r.is_recent):
                if r.is_recent(max_age_seconds):
                    recent.append(r)
            elif hasattr(r, 'timestamp'):
                age = datetime.now(timezone.utc) - r.timestamp
                if age.total_seconds() <= max_age_seconds:
                    recent.append(r)
        return recent
    
    def clear_old(self, max_age_seconds: int = 3600) -> int:
        """Remove old results and return count removed."""
        old_count = len(self.results)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        
        new_results = []
        for r in self.results:
            if hasattr(r, 'timestamp') and r.timestamp > cutoff:
                new_results.append(r)
        
        self.results = new_results
        
        # Rebuild indexes
        self._by_component.clear()
        self._by_type.clear()
        for result in self.results:
            if hasattr(result, 'component_id'):
                self._by_component[result.component_id].append(result)
            self._by_type[type(result)].append(result)
        
        return old_count - len(self.results)


def create_success_result(component_id: str, message: str = "Success", **kwargs) -> ProcessResult:
    """Convenience function to create a successful ProcessResult."""
    return ProcessResult(
        success=True,
        component_id=component_id,
        message=message,
        **kwargs
    )


def create_error_result(component_id: str, error: Exception, **kwargs) -> ProcessResult:
    """Convenience function to create an error ProcessResult from an exception."""
    return ProcessResult(
        success=False,
        component_id=component_id,
        message=str(error),
        error_code=type(error).__name__,
        **kwargs
    )


def aggregate_health(healths: List[ComponentHealth]) -> ComponentHealth:
    """Aggregate multiple health results into a system-wide health."""
    if not healths:
        return ComponentHealth(
            component_id="system",
            status="UNKNOWN",
            message="No component health data available"
        )
    
    # Count statuses
    status_counts = defaultdict(int)
    all_issues = []
    total_score = 0.0
    
    for health in healths:
        status_counts[health.status] += 1
        all_issues.extend(health.issues)
        total_score += health.health_score
    
    # Determine overall status
    if status_counts["UNHEALTHY"] > 0:
        overall_status = "UNHEALTHY"
    elif status_counts["DEGRADED"] > 0:
        overall_status = "DEGRADED"
    elif status_counts["UNKNOWN"] > 0:
        overall_status = "DEGRADED"
    else:
        overall_status = "HEALTHY"
    
    avg_score = total_score / len(healths) if healths else 0.0
    
    return ComponentHealth(
        component_id="system",
        status=overall_status,
        message=f"Aggregated health from {len(healths)} components",
        health_score=avg_score,
        issues=all_issues,
        details={
            "component_count": len(healths),
            "status_breakdown": dict(status_counts)
        }
    )