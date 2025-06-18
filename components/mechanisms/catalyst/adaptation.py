# nireon_v4\components\mechanisms\catalyst\adaptation.py
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from .types import BlendRange, AntiConstraints, MIN_BLEND_GAP
from .service_helpers.catalyst_event_helper import CatalystEventHelper

logger = logging.getLogger(__name__)

def handle_duplication_detected(current_blend: BlendRange, max_blend_low: float, max_blend_high: float, duplication_aggressiveness: float, current_step: Optional[int] = None, event_helper: Optional[CatalystEventHelper] = None) -> Tuple[BlendRange, Optional[int]]:
    """Increases blend range in response to duplication detection."""
    low, high = current_blend
    shift_amount = 0.1 * duplication_aggressiveness
    new_low = min(low + shift_amount, max_blend_low)
    new_high = min(high + shift_amount, max_blend_high)
    if new_low > new_high:
        new_high = new_low
    
    if (new_high - new_low) < MIN_BLEND_GAP:
        new_high = new_low + MIN_BLEND_GAP

    new_low = min(max(0.0, new_low), max_blend_low)
    new_high = min(max(0.0, new_high), max_blend_high)

    if current_blend != (new_low, new_high):
        logger.info('Duplication detected → blend range increased to %.2f-%.2f', new_low, new_high)
        # Event publication logic would go here if needed, using the event_helper
        return (new_low, new_high), current_step
    else:
        logger.debug('Duplication detected, but blend range did not change (already at max?).')
        return current_blend, current_step

def check_blend_cooldown(current_blend: BlendRange, base_blend: BlendRange, last_duplication_step: Optional[int], current_step: Optional[int], cooldown_steps: int, event_helper: Optional[CatalystEventHelper] = None) -> Tuple[BlendRange, Optional[int]]:
    """Resets the blend range if the cooldown period has passed."""
    if current_step is not None and last_duplication_step is not None and (current_step - last_duplication_step >= cooldown_steps):
        if current_blend != base_blend:
            logger.debug(f'Blend range reset due to cooldown ({cooldown_steps} steps).')
            # Event publication logic would go here if needed
            return base_blend, None
    return current_blend, last_duplication_step


def check_anti_constraints_expiry(active_anti_constraints: AntiConstraints, anti_constraints_expiry: Optional[int], current_step: Optional[int], event_helper: Optional[CatalystEventHelper] = None) -> Tuple[AntiConstraints, Optional[int]]:
    """Checks if active anti-constraints have expired."""
    if anti_constraints_expiry is not None and current_step is not None and current_step >= anti_constraints_expiry:
        if active_anti_constraints:
            logger.info('Anti-constraints expired at step %d', current_step)
            # Event publication logic would go here if needed
            return [], None
    return active_anti_constraints, anti_constraints_expiry