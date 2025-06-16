import logging
import copy # For deepcopy
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigMerger:
    @staticmethod
    def merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
        context_description: str = "ConfigMerge",
        strict_keys: bool = False, # V4 might default this differently based on config philosophy
        allow_new_keys_in_strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Merges an 'override' dictionary into a 'base' dictionary.
        - Dictionaries are merged recursively.
        - Other types in override replace values in base.
        - If strict_keys is True, override keys not in base raise ValueError,
          unless allow_new_keys_in_strict is also True.
        """
        if not isinstance(base, dict):
            logger.error(
                f"[{context_description}] Base for merge is not a dictionary (type: {type(base)}). "
                f"Returning override if dict, else empty."
            )
            return copy.deepcopy(override) if isinstance(override, dict) else {}
        
        if not isinstance(override, dict):
            logger.warning(
                f"[{context_description}] Override for merge is not a dictionary (type: {type(override)}). "
                f"Returning base."
            )
            return copy.deepcopy(base)

        merged = copy.deepcopy(base)
        logger.debug(
            f"[{context_description}] Starting merge. Base keys: {list(base.keys())}, Override keys: {list(override.keys())}"
        )

        for key, override_value in override.items():
            base_value = merged.get(key) # Use get for safer access if key might not exist in merged yet

            if key not in merged: # Key is new
                if strict_keys and not allow_new_keys_in_strict:
                    raise ValueError(
                        f"[{context_description}] Strict mode: Key '{key}' in override not found in base, and new keys not allowed."
                    )
                merged[key] = copy.deepcopy(override_value)
                logger.debug(
                    f"[{context_description}] Added new key '{key}' with value: {str(override_value)[:80]}"
                    f"{'...' if len(str(override_value)) > 80 else ''}"
                )
            elif isinstance(base_value, dict) and isinstance(override_value, dict):
                # Recursive merge for nested dictionaries
                logger.debug(f"[{context_description}] Recursively merging dict for key '{key}'.")
                merged[key] = ConfigMerger.merge(
                    base_value, 
                    override_value, 
                    context_description=f"{context_description} -> {key}",
                    strict_keys=strict_keys,
                    allow_new_keys_in_strict=allow_new_keys_in_strict
                )
            elif base_value == override_value:
                # V3 had a specific None check here, usually deepcopy handles this fine.
                # If `key not in base` was true but `key in merged` due to prior merges, this path might be hit.
                # Ensuring value is set if it was None due to initial deepcopy of base where key might not have existed.
                if base_value is None and key not in base: # if base itself didn't have the key
                     merged[key] = copy.deepcopy(override_value)
                logger.debug(f"[{context_description}] Key '{key}' has same value in base and override. No change.")
            else: # Override value is different and not a dict-dict merge
                merged[key] = copy.deepcopy(override_value)
                original_base_value_repr = str(base.get(key))[:80] # Get from original base for logging
                if len(str(base.get(key))) > 80: original_base_value_repr += "..."
                
                override_value_repr = str(override_value)[:80]
                if len(str(override_value)) > 80: override_value_repr += "..."
                
                logger.debug(
                    f"[{context_description}] Overridden key '{key}'. Old: {original_base_value_repr}, New: {override_value_repr}"
                )
        
        logger.debug(f"[{context_description}] Merge complete for this level. Result keys: {list(merged.keys())}")
        return merged

def merge_configs(base: Dict[str, Any], *overrides: Dict[str, Any], context: str = "ConfigChain", strict: bool = False) -> Dict[str, Any]:
    """
    Merges multiple override dictionaries into a base dictionary sequentially.
    `allow_new_keys_in_strict` is True by default for chained merges,
    meaning later configs in the chain can introduce new keys even if strict is on for
    the comparison against the immediate base.
    """
    result = base
    for i, override_config in enumerate(overrides):
        result = ConfigMerger.merge(
            result, 
            override_config, 
            context_description=f"{context}_Step{i + 1}", 
            strict_keys=strict,
            allow_new_keys_in_strict=True # Allow new keys in subsequent merges in a chain
        )
    return result

if __name__ == "__main__": # Example usage and tests
    logging.basicConfig(level=logging.DEBUG)
    merger = ConfigMerger()

    base_conf = {"a": 1, "b": {"c": 2, "d": 3}, "e": 5}
    override_conf = {"b": {"c": 20, "f": 6}, "e": 50, "g": 7}

    print("\n--- Test Case 1: Standard Merge ---")
    merged1 = merger.merge(base_conf, override_conf, "TestCase1")
    print("Merged Result 1:", merged1)
    expected1 = {"a": 1, "b": {"c": 20, "d": 3, "f": 6}, "e": 50, "g": 7}
    assert merged1 == expected1, f"Expected {expected1}, got {merged1}"

    print("\n--- Test Case 2: Strict Keys, New Keys Not Allowed (should fail for 'f' or 'g') ---")
    try:
        merger.merge(base_conf, override_conf, "TestCase2", strict_keys=True, allow_new_keys_in_strict=False)
        assert False, "Expected ValueError in strict mode for new keys"
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
        assert "Key 'f' in override not found in base" in str(e) or \
               "Key 'g' in override not found in base" in str(e) # Order of dict items can vary

    print("\nAll ConfigMerger tests (for this file) passed! ✅")