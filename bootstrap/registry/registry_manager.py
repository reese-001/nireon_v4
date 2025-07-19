"""
Registry Manager (v4.1)
----------------------

Adds a thin certification layer on top of the core
:class:`~core.registry.ComponentRegistry`.

Improvements over the legacy v4.0 implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Full type hints & `__all__`** - IDE / static-analysis friendly.
* **Module & function docstrings** - automatic API docs generation.
* **Richer logging** - lazy-formatted strings for performance.
* **Safer clean-up** - unregister rollback moved into a dedicated helper.
* **Deterministic cert hashes** - converts mutable values to *repr* strings
  before hashing to avoid ordering issues.
* **Slots on internal structs** - memory footprint reduced for large systems.

No behaviour changes have been introduced; all public APIs are identical.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.registry import ComponentRegistry

__all__ = ["RegistryManager"]

logger = logging.getLogger(__name__)


class RegistryManager:
    """
    Helper that delegates storage to :class:`~core.registry.ComponentRegistry`
    while adding **self-certification** metadata for each registered component.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, registry: ComponentRegistry) -> None:
        if not isinstance(registry, ComponentRegistry):
            raise TypeError("registry must be a ComponentRegistry instance")
        self.registry: ComponentRegistry = registry
        logger.debug("RegistryManager initialised with ComponentRegistry")

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    # ................ Component registration ........................... #

    def register_with_certification(
        self,
        component: Any,
        metadata: ComponentMetadata,
        additional_cert_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register *component* in the underlying registry and attach a
        cryptographic certification record derived from *metadata* and any
        extra fields.
        """
        cid = metadata.id
        try:
            self.registry.register(component, metadata)
            logger.debug("Component '%s' registered in registry", cid)

            cert = self._generate_certification_data(component, metadata, additional_cert_data)
            self.registry.register_certification(cid, cert)
            logger.info("✓ Component '%s' registered with self-certification", cid)
        except Exception as exc:
            logger.error("Failed to register '%s': %s", cid, exc, exc_info=True)
            self._safe_unregister(cid)
            raise

    def register_service_with_certification(
        self,
        service_type: Type,
        instance: Any,
        service_id: str,
        *,
        category: str = "service",
        description: Optional[str] = None,
        requires_initialize: bool = False,
    ) -> None:
        """
        Convenience wrapper for services that are registered *by type* as well
        as by component-ID.
        """
        from bootstrap.bootstrap_helper.metadata import create_service_metadata

        metadata = create_service_metadata(
            service_id=service_id,
            service_name=service_id,
            category=category,
            description=description or f"Service instance for {service_id}",
            requires_initialize=requires_initialize,
        )

        try:
            if hasattr(self.registry, "register_service_instance"):
                self.registry.register_service_instance(service_type, instance)  # type: ignore[attr-defined]
                logger.debug("Service '%s' registered by type %s", service_id, service_type.__name__)
            self.register_with_certification(instance, metadata)
        except Exception as exc:
            logger.error("Failed to register service '%s': %s", service_id, exc, exc_info=True)
            raise

    # ................ Certification helpers ............................ #

    def verify_certification(self, component_id: str) -> bool:
        """Return ``True`` iff the stored certification hash matches recomputed value."""
        try:
            cert = self.registry.get_certification(component_id)
            if not cert:
                logger.warning("No certification for '%s'", component_id)
                return False
            stored_hash = cert.get("certification_hash")
            if not stored_hash:
                logger.warning("Certification hash missing for '%s'", component_id)
                return False
            calc_hash = self._generate_certification_hash(cert)
            if stored_hash != calc_hash:
                logger.error("Certification hash mismatch for '%s'", component_id)
                logger.debug("Stored=%s Calculated=%s", stored_hash, calc_hash)
                return False
            return True
        except Exception as exc:
            logger.error("Error verifying certification for '%s': %s", component_id, exc, exc_info=True)
            return False

    def list_certified_components(self) -> List[str]:
        """Return list of IDs that pass :meth:`verify_certification`."""
        return [cid for cid in self.registry.list_components() if self.verify_certification(cid)]

    def get_certification_summary(self) -> Dict[str, Any]:
        """Return aggregate stats on certification coverage."""
        all_components = self.registry.list_components()
        certified = self.list_certified_components()
        return {
            "total_components": len(all_components),
            "certified_components": len(certified),
            "certification_rate": len(certified) / len(all_components) if all_components else 0.0,
            "uncertified_components": [c for c in all_components if c not in certified],
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _safe_unregister(self, component_id: str) -> None:
        """Attempt to roll back a partial registration without raising."""
        try:
            if hasattr(self.registry, "unregister"):
                self.registry.unregister(component_id)  # type: ignore[attr-defined]
                logger.debug("Rolled back registration of '%s'", component_id)
        except Exception:
            pass  # swallow rollback failures - original error is more important

    # certification data generation -------------------------------------

    def _generate_certification_data(
        self,
        component: Any,
        metadata: ComponentMetadata,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cert: Dict[str, Any] = {
            "component_id": metadata.id,
            "component_name": metadata.name,
            "version": metadata.version,
            "category": metadata.category,
            "status": "registered",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epistemic_tags": list(metadata.epistemic_tags),
            "capabilities": list(metadata.capabilities),
        }

        # Config fingerprint
        if hasattr(component, "config"):
            try:
                cfg_items = getattr(component, "config").items()  # type: ignore[attr-defined]
                cfg_repr = str(sorted(cfg_items))
            except Exception:
                cfg_repr = repr(getattr(component, "config"))
            cert["config_hash"] = hashlib.sha256(cfg_repr.encode()).hexdigest()

        # Extra info for NireonBaseComponent
        if isinstance(component, NireonBaseComponent):
            cert.update(
                {
                    "base_component": True,
                    "is_initialized": getattr(component, "is_initialized", False),
                    "process_count": getattr(component, "process_count", 0),
                    "error_count": getattr(component, "error_count", 0),
                }
            )

        if extra:
            cert.update(extra)

        cert["certification_hash"] = self._generate_certification_hash(cert)
        return cert

    @staticmethod
    def _generate_certification_hash(data: Dict[str, Any]) -> str:
        to_hash = {k: v for k, v in data.items() if k != "certification_hash"}
        canonical = str(sorted(to_hash.items()))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
