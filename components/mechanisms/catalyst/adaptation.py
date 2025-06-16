import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from .types import BlendRange, AntiConstraints, MIN_BLEND_GAP
logger = logging.getLogger(__name__)

def handle_duplication_detected(current_blend: BlendRange, max_blend_low: float, max_blend_high: float, duplication_aggressiveness: float, current_step: Optional[int]=None, event_bus: Optional[Any]=None) -> Tuple[BlendRange, Optional[int]]:
    low, high = current_blend
    shift_amount = 0.1 * duplication_aggressiveness
    new_low = min(low + shift_amount, max_blend_low)
    new_high = min(high + shift_amount, max_blend_high)
    if new_low > new_high:
        new_high = new_low
    if new_high - new_low < MIN_BLEND_GAP:
        new_high = new_low + MIN_BLEND_GAP
    new_low = min(max(0.0, new_low), max_blend_low)
    new_high = min(max(0.0, new_high), max_blend_high)
    if current_blend != (new_low, new_high):
        logger.info('Duplication detected → blend range increased to %.2f-%.2f', new_low, new_high)
        if event_bus:
            event_bus.publish('adaptation', {'type': 'catalyst_blend_adjust', 'reason': 'duplication_detected', 'new_blend_range': (new_low, new_high), 'step': current_step, 'timestamp': datetime.now(timezone.utc).isoformat()})
        return ((new_low, new_high), current_step)
    else:
        logger.debug('Duplication detected, but blend range did not change (already at max?).')
        return (current_blend, current_step)

def check_blend_cooldown(current_blend: BlendRange, base_blend: BlendRange, last_duplication_step: Optional[int], current_step: Optional[int], cooldown_steps: int, event_bus: Optional[Any]=None) -> Tuple[BlendRange, Optional[int]]:
    if current_step is not None and last_duplication_step is not None and (current_step - last_duplication_step >= cooldown_steps):
        if current_blend != base_blend:
            logger.debug(f'Blend range reset due to cooldown ({cooldown_steps} steps).')
            if event_bus:
                event_bus.publish('adaptation', {'type': 'catalyst_blend_reset', 'reason': 'cooldown_passed', 'new_blend_range': base_blend, 'step': current_step, 'timestamp': datetime.now(timezone.utc).isoformat()})
            return (base_blend, None)
    return (current_blend, last_duplication_step)

def check_anti_constraints_expiry(active_anti_constraints: AntiConstraints, anti_constraints_expiry: Optional[int], current_step: Optional[int], event_bus: Optional[Any]=None) -> Tuple[AntiConstraints, Optional[int]]:
    if anti_constraints_expiry is not None and current_step is not None and (current_step >= anti_constraints_expiry):
        if active_anti_constraints:
            logger.info('Anti-constraints expired at step %d', current_step)
            if event_bus:
                event_bus.publish('adaptation', {'type': 'catalyst_anti_constraints_expired', 'step': current_step, 'timestamp': datetime.now(timezone.utc).isoformat()})
            return ([], None)
    return (active_anti_constraints, anti_constraints_expiry)

def set_anti_constraints(constraints: List[str], anti_constraints_enabled: bool, anti_constraints_count: int, expiry_step: Optional[int]=None, current_step: Optional[int]=None, event_bus: Optional[Any]=None) -> Tuple[AntiConstraints, Optional[int]]:
    if not anti_constraints_enabled:
        logger.warning('Cannot set anti-constraints: feature is disabled in config')
        if event_bus:
            event_bus.publish('system_status', {'status_type': 'warning', 'component': 'Catalyst', 'message': 'Attempted to set anti-constraints but feature is disabled.', 'severity': 'warning', 'step': current_step, 'timestamp': datetime.now(timezone.utc).isoformat()})
        return ([], None)
    if not isinstance(constraints, list) or not all((isinstance(c, str) for c in constraints)):
        logger.error(f'Invalid input for set_anti_constraints. Expected List[str], got {type(constraints)}')
        if event_bus:
            event_bus.publish('system_status', {'status_type': 'error', 'component': 'Catalyst', 'message': 'Invalid input for set_anti_constraints.', 'severity': 'error', 'step': current_step, 'timestamp': datetime.now(timezone.utc).isoformat()})
        return ([], None)
    max_constraints = anti_constraints_count
    if len(constraints) > max_constraints:
        logger.warning(f'Too many anti-constraints provided ({len(constraints)}), using first {max_constraints}')
        constraints = constraints[:max_constraints]
    logger.info(f"Set {len(constraints)} anti-constraints at step {current_step}, expiring at step {expiry_step or 'N/A'}")
    for c in constraints:
        logger.debug(f'Anti-constraint: {c}')
    if event_bus:
        event_bus.publish('adaptation', {'type': 'catalyst_anti_constraints_set', 'constraints': constraints, 'expiry_step': expiry_step, 'step': current_step, 'timestamp': datetime.now(timezone.utc).isoformat()})
    return (constraints, expiry_step)

def clear_anti_constraints(active_anti_constraints: AntiConstraints, current_step: Optional[int]=None, event_bus: Optional[Any]=None) -> Tuple[AntiConstraints, Optional[int]]:
    if active_anti_constraints:
        logger.info(f'Cleared {len(active_anti_constraints)} anti-constraints at step {current_step}')
        cleared_constraints = active_anti_constraints
        if event_bus:
            event_bus.publish('adaptation', {'type': 'catalyst_anti_constraints_cleared', 'cleared_constraints': cleared_constraints, 'step': current_step, 'timestamp': datetime.now(timezone.utc).isoformat()})
    return ([], None)