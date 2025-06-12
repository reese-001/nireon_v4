# nireon/domain/mixins/deterministic_rng_mixin.py
import hashlib
import random
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DeterministicRNGMixin:
    """
    Mixin to provide deterministic RNG functionality to frames.
    
    This mixin adds reproducible randomness based on frame identity and creation time.
    Perfect for testing, debugging, and ensuring consistent behavior across runs.
    """
    
    def _generate_frame_seed(self) -> int:
        """
        Generate a deterministic seed using frame identity and creation time.
        
        Returns:
            int: A 32-bit integer seed derived from frame id and creation timestamp
        """
        # Use frame ID and creation timestamp for deterministic seeding
        base_string = f"{self.id}_{self.created_ts}"
        
        # Add schema version for future compatibility
        if hasattr(self, 'schema_version'):
            base_string += f"_{self.schema_version}"
        
        # Generate hash and convert to integer seed
        hash_digest = hashlib.sha256(base_string.encode()).hexdigest()
        seed = int(hash_digest[:8], 16)  # First 8 hex chars → 32-bit int
        
        logger.debug(f"Generated deterministic seed {seed} for frame {self.id}")
        return seed
    
    def get_rng(self) -> random.Random:
        """
        Return a cached, deterministic RNG instance per frame.
        
        The RNG is lazily initialized and cached for performance.
        Multiple calls to this method on the same frame return the same RNG instance.
        
        Returns:
            random.Random: Deterministic Random instance for this frame
        """
        if not hasattr(self, "_rng") or self._rng is None:
            seed = self._generate_frame_seed()
            self._rng = random.Random(seed)
            logger.debug(f"Initialized deterministic RNG for frame {self.id} with seed {seed}")
        
        return self._rng
    
    def reset_rng(self) -> None:
        """
        Reset the RNG instance, forcing regeneration on next get_rng() call.
        
        Useful for testing or when frame properties that affect seeding have changed.
        """
        if hasattr(self, "_rng"):
            self._rng = None
            logger.debug(f"Reset RNG for frame {self.id}")
    
    def get_rng_seed(self) -> int:
        """
        Get the current RNG seed without initializing the RNG instance.
        
        Returns:
            int: The deterministic seed that would be (or was) used for this frame's RNG
        """
        return self._generate_frame_seed()
    
    def set_custom_rng_seed(self, custom_seed: Optional[int] = None) -> None:
        """
        Override the deterministic seed with a custom one.
        
        Args:
            custom_seed: Custom seed to use. If None, reverts to deterministic seeding.
        """
        if custom_seed is not None:
            self._rng = random.Random(custom_seed)
            logger.debug(f"Set custom RNG seed {custom_seed} for frame {self.id}")
        else:
            # Revert to deterministic seeding
            self.reset_rng()
            logger.debug(f"Reverted to deterministic RNG seeding for frame {self.id}")