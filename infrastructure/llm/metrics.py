# nireon_v4/infrastructure/llm/metrics.py
"""
Comprehensive metrics collection system for LLM operations.
"""
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque
from enum import Enum
import statistics

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MetricSample:
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call."""
    call_id: str
    model: str
    provider: str
    stage: str
    role: str
    prompt_length: int
    response_length: int
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class LLMMetricsCollector:
    """
    Centralized metrics collection for LLM operations.
    Thread-safe and provides various aggregation methods.
    """
    
    def __init__(self, max_samples_per_metric: int = 10000):
        self.max_samples_per_metric = max_samples_per_metric
        self._lock = threading.RLock()
        
        # Raw metrics storage
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, Deque[MetricSample]] = defaultdict(
            lambda: deque(maxlen=max_samples_per_metric)
        )
        
        # LLM-specific metrics
        self._call_metrics: Deque[LLMCallMetrics] = deque(maxlen=max_samples_per_metric)
        
        # Aggregated metrics cache
        self._cached_aggregations: Dict[str, Any] = {}
        self._cache_timestamp = 0
        self._cache_ttl = 60  # Cache for 60 seconds
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric (always increasing)."""
        with self._lock:
            key = self._build_key(name, labels)
            self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric (can go up or down)."""
        with self._lock:
            key = self._build_key(name, labels)
            self._gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram sample."""
        with self._lock:
            key = self._build_key(name, labels)
            sample = MetricSample(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            self._histograms[key].append(sample)
    
    def record_llm_call(self, metrics: LLMCallMetrics):
        """Record comprehensive LLM call metrics."""
        with self._lock:
            self._call_metrics.append(metrics)
            
            # Also record as individual metrics
            labels = {
                'model': metrics.model,
                'provider': metrics.provider,
                'stage': metrics.stage,
                'role': metrics.role
            }
            
            self.record_counter('llm_calls_total', 1.0, labels)
            self.record_histogram('llm_duration_ms', metrics.duration_ms, labels)
            self.record_histogram('llm_prompt_length', metrics.prompt_length, labels)
            self.record_histogram('llm_response_length', metrics.response_length, labels)
            
            if metrics.success:
                self.record_counter('llm_calls_successful', 1.0, labels)
            else:
                error_labels = {**labels, 'error_type': metrics.error_type or 'unknown'}
                self.record_counter('llm_calls_failed', 1.0, error_labels)
    
    def _build_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Build a unique key for metrics with labels."""
        if not labels:
            return name
        
        label_str = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
        return f'{name}{{{label_str}}}'
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        with self._lock:
            key = self._build_key(name, labels)
            return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        with self._lock:
            key = self._build_key(name, labels)
            return self._gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics (min, max, mean, median, p95, p99)."""
        with self._lock:
            key = self._build_key(name, labels)
            samples = self._histograms.get(key, deque())
            
            if not samples:
                return {}
            
            values = [s.value for s in samples]
            values.sort()
            
            n = len(values)
            return {
                'count': n,
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'p95': values[int(0.95 * n)] if n > 0 else 0,
                'p99': values[int(0.99 * n)] if n > 0 else 0,
                'stddev': statistics.stdev(values) if n > 1 else 0
            }
    
    def get_llm_summary(self, time_window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive LLM metrics summary."""
        current_time = time.time()
        
        # Check cache
        if (current_time - self._cache_timestamp) < self._cache_ttl and 'llm_summary' in self._cached_aggregations:
            return self._cached_aggregations['llm_summary']
        
        with self._lock:
            # Filter by time window if specified
            relevant_calls = self._call_metrics
            if time_window_seconds:
                cutoff_time = current_time - time_window_seconds
                relevant_calls = [c for c in self._call_metrics if c.timestamp >= cutoff_time]
            
            if not relevant_calls:
                return {'total_calls': 0, 'time_window_seconds': time_window_seconds}
            
            # Calculate aggregations
            total_calls = len(relevant_calls)
            successful_calls = sum(1 for c in relevant_calls if c.success)
            failed_calls = total_calls - successful_calls
            
            durations = [c.duration_ms for c in relevant_calls]
            prompt_lengths = [c.prompt_length for c in relevant_calls]
            response_lengths = [c.response_length for c in relevant_calls]
            
            # Group by various dimensions
            by_model = defaultdict(int)
            by_provider = defaultdict(int)
            by_stage = defaultdict(int)
            by_error_type = defaultdict(int)
            
            for call in relevant_calls:
                by_model[call.model] += 1
                by_provider[call.provider] += 1
                by_stage[call.stage] += 1
                if not call.success and call.error_type:
                    by_error_type[call.error_type] += 1
            
            summary = {
                'total_calls': total_calls,
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
                'failure_rate': failed_calls / total_calls if total_calls > 0 else 0,
                'time_window_seconds': time_window_seconds,
                'duration_stats': {
                    'mean_ms': statistics.mean(durations),
                    'median_ms': statistics.median(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                    'p95_ms': sorted(durations)[int(0.95 * len(durations))] if durations else 0,
                    'p99_ms': sorted(durations)[int(0.99 * len(durations))] if durations else 0,
                },
                'prompt_length_stats': {
                    'mean': statistics.mean(prompt_lengths),
                    'median': statistics.median(prompt_lengths),
                    'min': min(prompt_lengths),
                    'max': max(prompt_lengths),
                },
                'response_length_stats': {
                    'mean': statistics.mean(response_lengths),
                    'median': statistics.median(response_lengths),
                    'min': min(response_lengths),
                    'max': max(response_lengths),
                },
                'breakdown_by_model': dict(by_model),
                'breakdown_by_provider': dict(by_provider),
                'breakdown_by_stage': dict(by_stage),
                'error_types': dict(by_error_type),
                'timestamp': current_time
            }
            
            # Cache the result
            self._cached_aggregations['llm_summary'] = summary
            self._cache_timestamp = current_time
            
            return summary
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a structured format."""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {
                    name: self.get_histogram_stats(name.split('{')[0], 
                                                  self._parse_labels(name) if '{' in name else None)
                    for name in self._histograms.keys()
                },
                'llm_summary': self.get_llm_summary(),
                'collection_info': {
                    'max_samples_per_metric': self.max_samples_per_metric,
                    'total_call_metrics': len(self._call_metrics),
                    'timestamp': time.time()
                }
            }
    
    def _parse_labels(self, key: str) -> Dict[str, str]:
        """Parse labels from a metric key."""
        if '{' not in key:
            return {}
        
        label_part = key.split('{', 1)[1].rstrip('}')
        labels = {}
        for pair in label_part.split(','):
            if '=' in pair:
                k, v = pair.split('=', 1)
                labels[k] = v
        return labels
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._call_metrics.clear()
            self._cached_aggregations.clear()
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        with self._lock:
            # Export counters
            for key, value in self._counters.items():
                lines.append(f'# TYPE {key.split("{")[0]} counter')
                lines.append(f'{key} {value}')
            
            # Export gauges
            for key, value in self._gauges.items():
                lines.append(f'# TYPE {key.split("{")[0]} gauge')
                lines.append(f'{key} {value}')
            
            # Export histogram summaries
            for key, samples in self._histograms.items():
                if samples:
                    base_name = key.split('{')[0]
                    stats = self.get_histogram_stats(base_name, self._parse_labels(key) if '{' in key else None)
                    
                    lines.append(f'# TYPE {base_name} histogram')
                    for stat_name, stat_value in stats.items():
                        metric_name = f'{base_name}_{stat_name}'
                        if '{' in key:
                            metric_name += key[key.index('{'):]
                        lines.append(f'{metric_name} {stat_value}')
        
        return '\n'.join(lines)

# Global metrics collector instance
_global_metrics_collector: Optional[LLMMetricsCollector] = None

def get_metrics_collector() -> LLMMetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = LLMMetricsCollector()
    return _global_metrics_collector

def record_llm_call_metrics(call_id: str, model: str, provider: str, stage: str, 
                           role: str, prompt_length: int, response_length: int,
                           duration_ms: float, success: bool, error_type: Optional[str] = None):
    """Convenience function to record LLM call metrics."""
    metrics = LLMCallMetrics(
        call_id=call_id,
        model=model,
        provider=provider,
        stage=stage,
        role=role,
        prompt_length=prompt_length,
        response_length=response_length,
        duration_ms=duration_ms,
        success=success,
        error_type=error_type
    )
    get_metrics_collector().record_llm_call(metrics)