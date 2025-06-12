# nireon/infrastructure/llm/circuit_breaker.py
"""
Circuit breaker pattern implementation for LLM backends.
"""
import asyncio
import time
import logging
from enum import Enum
from typing import Any, Dict, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from .exceptions import LLMError, LLMBackendNotAvailableError

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered

@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_state: CircuitState = CircuitState.CLOSED
    
    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    @property
    def success_rate(self) -> float:
        return 1.0 - self.failure_rate

class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Circuit is open, calls fail fast for recovery_timeout period
    - HALF_OPEN: Testing phase, limited calls allowed to test recovery
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,           # Failures before opening
                 recovery_timeout: int = 60,           # Seconds before retry
                 success_threshold: int = 3,           # Successes to close from half-open
                 timeout: float = 30.0,                # Individual call timeout
                 monitoring_window: int = 300):        # Window for failure rate calculation
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.monitoring_window = monitoring_window
        
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.half_open_calls = 0
        self.last_state_change = time.time()
        
        # Sliding window for recent failures
        self.recent_failures: list[float] = []
        
        logger.info(f"Circuit breaker initialized: failure_threshold={failure_threshold}, "
                   f"recovery_timeout={recovery_timeout}s, success_threshold={success_threshold}")
    
    def _record_success(self):
        """Record a successful call."""
        current_time = time.time()
        self.stats.total_calls += 1
        self.stats.successful_calls += 1
        self.stats.last_success_time = current_time
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.success_threshold:
                self._transition_to_closed()
        
        logger.debug(f"Circuit breaker: Success recorded, state={self.state.value}")
    
    def _record_failure(self, error: Exception):
        """Record a failed call."""
        current_time = time.time()
        self.stats.total_calls += 1
        self.stats.failed_calls += 1
        self.stats.last_failure_time = current_time
        
        # Add to recent failures window
        self.recent_failures.append(current_time)
        
        # Clean old failures outside monitoring window
        cutoff_time = current_time - self.monitoring_window
        self.recent_failures = [t for t in self.recent_failures if t > cutoff_time]
        
        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED and len(self.recent_failures) >= self.failure_threshold:
            self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        
        logger.warning(f"Circuit breaker: Failure recorded ({type(error).__name__}), "
                      f"state={self.state.value}, recent_failures={len(self.recent_failures)}")
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        if self.state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker: Transitioning to OPEN state after "
                          f"{len(self.recent_failures)} recent failures")
            self.state = CircuitState.OPEN
            self.stats.state_changes += 1
            self.last_state_change = time.time()
            self.half_open_calls = 0
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        logger.info("Circuit breaker: Transitioning to HALF_OPEN state for testing")
        self.state = CircuitState.HALF_OPEN
        self.stats.state_changes += 1
        self.last_state_change = time.time()
        self.half_open_calls = 0
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        logger.info("Circuit breaker: Transitioning to CLOSED state - service recovered")
        self.state = CircuitState.CLOSED
        self.stats.state_changes += 1
        self.stats.current_state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.half_open_calls = 0
        # Clear recent failures on successful recovery
        self.recent_failures.clear()
    
    def _should_allow_call(self) -> bool:
        """Determine if a call should be allowed based on current state."""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            # Check if enough time has passed to try recovery
            if current_time - self.last_state_change >= self.recovery_timeout:
                self._transition_to_half_open()
                return True
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self.half_open_calls < self.success_threshold
        
        return False
    
    async def call_async(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute an async function with circuit breaker protection."""
        if not self._should_allow_call():
            raise LLMBackendNotAvailableError(
                f"Circuit breaker is {self.state.value} - backend temporarily unavailable",
                details={
                    'state': self.state.value,
                    'recent_failures': len(self.recent_failures),
                    'time_until_retry': max(0, self.recovery_timeout - (time.time() - self.last_state_change))
                }
            )
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            self._record_success()
            return result
            
        except asyncio.TimeoutError as e:
            timeout_error = LLMError(f"Circuit breaker timeout after {self.timeout}s", cause=e)
            self._record_failure(timeout_error)
            raise timeout_error
            
        except Exception as e:
            self._record_failure(e)
            raise
    
    def call_sync(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute a sync function with circuit breaker protection."""
        if not self._should_allow_call():
            raise LLMBackendNotAvailableError(
                f"Circuit breaker is {self.state.value} - backend temporarily unavailable",
                details={
                    'state': self.state.value,
                    'recent_failures': len(self.recent_failures),
                    'time_until_retry': max(0, self.recovery_timeout - (time.time() - self.last_state_change))
                }
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure(e)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        current_time = time.time()
        return {
            'state': self.state.value,
            'total_calls': self.stats.total_calls,
            'successful_calls': self.stats.successful_calls,
            'failed_calls': self.stats.failed_calls,
            'failure_rate': self.stats.failure_rate,
            'success_rate': self.stats.success_rate,
            'recent_failures': len(self.recent_failures),
            'state_changes': self.stats.state_changes,
            'last_failure_time': self.stats.last_failure_time,
            'last_success_time': self.stats.last_success_time,
            'time_in_current_state': current_time - self.last_state_change,
            'half_open_calls': self.half_open_calls if self.state == CircuitState.HALF_OPEN else 0
        }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        logger.info("Circuit breaker: Manual reset to CLOSED state")
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.half_open_calls = 0
        self.last_state_change = time.time()
        self.recent_failures.clear()
    
    def force_open(self):
        """Manually force circuit to OPEN state."""
        logger.warning("Circuit breaker: Manually forced to OPEN state")
        self._transition_to_open()

# Example integration with LLM adapter
class CircuitBreakerLLMAdapter:
    """Wrapper that adds circuit breaker protection to any LLM adapter."""
    
    def __init__(self, llm_adapter, circuit_breaker_config: Optional[Dict[str, Any]] = None):
        self.llm_adapter = llm_adapter
        self.circuit_breaker = CircuitBreaker(**(circuit_breaker_config or {}))
        
    async def call_llm_async(self, *args, **kwargs):
        """Protected async LLM call."""
        return await self.circuit_breaker.call_async(
            self.llm_adapter.call_llm_async, *args, **kwargs
        )
    
    def call_llm_sync(self, *args, **kwargs):
        """Protected sync LLM call."""
        return self.circuit_breaker.call_sync(
            self.llm_adapter.call_llm_sync, *args, **kwargs
        )
    
    def get_stats(self):
        """Get combined stats from adapter and circuit breaker."""
        adapter_stats = getattr(self.llm_adapter, 'get_stats', lambda: {})()
        circuit_stats = self.circuit_breaker.get_stats()
        
        return {
            'adapter_stats': adapter_stats,
            'circuit_breaker_stats': circuit_stats
        }