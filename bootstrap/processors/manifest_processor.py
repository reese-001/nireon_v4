"""
Manifest Processor
──────────────────
* JSON‑schema validation (optional when `jsonschema` not installed)
* Parsing of *simple* and *enhanced* manifest formats
* Extraction of `ComponentSpec` objects
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from runtime.utils import detect_manifest_type

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional dependency – jsonschema
# --------------------------------------------------------------------------- #
try:
    import jsonschema as _jsonschema  # type: ignore
except ImportError:  # pragma: no cover
    _jsonschema = None
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Constants / helpers
# --------------------------------------------------------------------------- #
_ENH_SECTIONS: Mapping[str, str] = {
    "shared_services": "shared_service",
    "mechanisms": "mechanism",
    "observers": "observer",
    "managers": "manager",
    "composites": "composite",
    "orchestration_commands": "orchestration_command",
}
_PARAMS_KEY = "parameters"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _append_error(errors: MutableMapping[str, List[str]], key: str, msg: str) -> None:
    errors.setdefault(key, []).append(msg)
    logger.error("%s", msg)


# ===========================================================================
# Core classes
# ===========================================================================


class ManifestProcessor:
    """Validate & parse manifest files."""

    def __init__(self, strict_mode: bool = True) -> None:
        self.strict_mode = strict_mode
        self.package_root = Path(__file__).resolve().parents[2]
        self._schema_cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def process_manifest(
        self, manifest_path: Path, manifest_data: Dict[str, Any]
    ) -> "ManifestProcessingResult":
        logger.info("Processing manifest: %s", manifest_path)
        errors: List[str] = []
        warnings: List[str] = []
        components: List[ComponentSpec] = []

        try:
            mtype = detect_manifest_type(manifest_data)
            logger.debug("Detected manifest type: %s", mtype)

            # 1) Schema validation
            errors.extend(await self._validate_schema(manifest_data, mtype, manifest_path))
            if errors and self.strict_mode:
                return ManifestProcessingResult(False, manifest_path, mtype, errors, warnings, [])

            # 2) Component extraction
            if mtype == "enhanced":
                components = await self._process_enhanced(manifest_data, errors, warnings)
            elif mtype == "simple":
                components = await self._process_simple(manifest_data, errors, warnings)
            else:
                msg = f"Unknown manifest type: {mtype}"
                errors.append(msg)
                if self.strict_mode:
                    raise ValueError(msg)

            return ManifestProcessingResult(not errors, manifest_path, mtype, errors, warnings, components)

        except Exception as exc:
            msg = f"Critical error processing manifest {manifest_path}: {exc}"
            logger.exception(msg)
            return ManifestProcessingResult(False, manifest_path, "unknown", [msg], warnings, [])

    # ------------------------------------------------------------------ #
    # Schema validation
    # ------------------------------------------------------------------ #
    async def _validate_schema(
        self, manifest: Dict[str, Any], mtype: str, path: Path
    ) -> List[str]:
        if not _jsonschema:
            if self.strict_mode:
                return ["jsonschema package required for validation in strict mode"]
            logger.warning("jsonschema not installed – skipping schema validation")
            return []

        schema = await self._load_schema(mtype)
        if not schema:
            msg = f"No schema available for manifest type '{mtype}'"
            logger.warning(msg)
            return [msg] if self.strict_mode else []

        try:
            _jsonschema.validate(manifest, schema)
            logger.debug("✓ Schema validation passed for %s", path)
            return []
        except _jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            loc = ".".join(map(str, exc.absolute_path)) if exc.absolute_path else ""
            return [f"Schema validation failed: {exc.message} at {loc}"]
        except _jsonschema.SchemaError as exc:  # type: ignore[attr-defined]
            return [f"Invalid schema: {exc.message}"]

    async def _load_schema(self, mtype: str) -> Optional[Dict[str, Any]]:
        if mtype in self._schema_cache:
            return self._schema_cache[mtype]

        schema_path = self.package_root / "schemas" / f"{mtype}_manifest.schema.json"
        if not schema_path.exists():
            schema_path = self.package_root / "schemas" / "manifest.schema.json"
            if not schema_path.exists():
                logger.warning("Schema file not found for type '%s'", mtype)
                return None

        try:
            schema = _read_json(schema_path)
            self._schema_cache[mtype] = schema
            logger.debug("Loaded schema: %s", schema_path)
            return schema
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load schema %s: %s", schema_path, exc)
            return None

    # ------------------------------------------------------------------ #
    # Enhanced manifest
    # ------------------------------------------------------------------ #
    async def _process_enhanced(
        self,
        data: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> List["ComponentSpec"]:
        comps: List[ComponentSpec] = []
        for section, ctype in _ENH_SECTIONS.items():
            for cid, spec in (data.get(section) or {}).items():
                if not isinstance(spec, dict):
                    errors.append(f"Component spec for '{cid}' must be a dict")
                    continue
                if not spec.get("enabled", True):
                    logger.debug("Skipping disabled component: %s", cid)
                    continue
                cspec = ComponentSpec(
                    component_id=cid,
                    component_type=ctype,
                    section_name=section,
                    spec_data=spec,
                    manifest_type="enhanced",
                )
                errs = self._validate_component_spec(cspec)
                if errs and self.strict_mode:
                    errors.extend(errs)
                    continue
                errors.extend(errs)
                comps.append(cspec)
        return comps

    # ------------------------------------------------------------------ #
    # Simple manifest
    # ------------------------------------------------------------------ #
    async def _process_simple(
        self,
        data: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> List["ComponentSpec"]:
        comps: List[ComponentSpec] = []
        comp_list = data.get("components", [])
        if not isinstance(comp_list, list):
            errors.append("Simple manifest 'components' must be a list")
            return comps

        for idx, spec in enumerate(comp_list):
            if not isinstance(spec, dict):
                errors.append(f"Component definition {idx} must be a dict")
                continue
            if not spec.get("enabled", True):
                logger.debug("Skipping disabled component at index %d", idx)
                continue
            cspec = ComponentSpec(
                component_id=spec.get("component_id", f"component_{idx}"),
                component_type=spec.get("type", "unknown"),
                section_name="components",
                spec_data=spec,
                manifest_type="simple",
            )
            errs = self._validate_component_spec(cspec)
            if errs and self.strict_mode:
                errors.extend(errs)
                continue
            errors.extend(errs)
            comps.append(cspec)
        return comps

    # ------------------------------------------------------------------ #
    # Component‑spec validation
    # ------------------------------------------------------------------ #
    def _validate_component_spec(self, spec: "ComponentSpec") -> List[str]:
        errs: List[str] = []
        sd = spec.spec_data

        def _need(field: str) -> None:
            if field not in sd:
                errs.append(f"Component '{spec.component_id}' missing required '{field}' field")

        # common
        if not spec.component_id:
            errs.append("Component must have a non‑empty ID")

        if spec.manifest_type == "enhanced":
            if spec.component_type in {"mechanism", "observer", "manager", "composite"}:
                _need("class")
                _need("metadata_definition")
            elif spec.component_type in {"shared_service", "orchestration_command"}:
                _need("class")

        else:  # simple
            _need("component_id")
            _need("type")
            if not (sd.get("factory_key") or sd.get("class")):
                errs.append(
                    f"Simple component '{spec.component_id}' needs either 'factory_key' or 'class'"
                )
        return errs


# ===========================================================================
# Data containers
# ===========================================================================


class ComponentSpec:
    """Represents a component extracted from a manifest."""

    def __init__(
        self,
        component_id: str,
        component_type: str,
        section_name: str,
        spec_data: Dict[str, Any],
        manifest_type: str,
    ) -> None:
        self.component_id = component_id
        self.component_type = component_type
        self.section_name = section_name
        self.spec_data = spec_data
        self.manifest_type = manifest_type

    def __repr__(self) -> str:
        return (
            "ComponentSpec("
            f"id='{self.component_id}', type='{self.component_type}', manifest='{self.manifest_type}')"
        )


class ManifestProcessingResult:
    """Outcome of a single manifest parse/validation run."""

    def __init__(
        self,
        success: bool,
        manifest_path: Path,
        manifest_type: str,
        errors: Sequence[str],
        warnings: Sequence[str],
        components: Sequence[ComponentSpec],
    ) -> None:
        self.success = success
        self.manifest_path = manifest_path
        self.manifest_type = manifest_type
        self.errors = list(errors)
        self.warnings = list(warnings)
        self.components = list(components)

    # Convenience helpers
    @property
    def component_count(self) -> int:  # noqa: D401
        """Number of components extracted."""
        return len(self.components)

    def get_components_by_type(self, ctype: str) -> List[ComponentSpec]:
        return [c for c in self.components if c.component_type == ctype]
