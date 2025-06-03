from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bootstrap.bootstrap_helper.utils import detect_manifest_type

logger = logging.getLogger(__name__)

class ManifestProcessor:
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.package_root = Path(__file__).resolve().parents[2]
        self._schema_cache: Dict[str, Dict] = {}

    async def process_manifest(self, manifest_path: Path, manifest_data: Dict[str, Any]) -> 'ManifestProcessingResult':
        logger.info(f'Processing manifest: {manifest_path}')
        
        errors = []
        warnings = []
        components = []
        
        try:
            # Detect manifest type
            manifest_type = detect_manifest_type(manifest_data)
            logger.debug(f'Detected manifest type: {manifest_type}')
            
            # Validate schema
            schema_errors = await self._validate_schema(manifest_data, manifest_type, manifest_path)
            if schema_errors:
                errors.extend(schema_errors)
                if self.strict_mode:
                    return ManifestProcessingResult(
                        success=False,
                        manifest_path=manifest_path,
                        manifest_type=manifest_type,
                        errors=errors,
                        warnings=warnings,
                        components=[]
                    )
            
            # Process based on manifest type
            if manifest_type == 'enhanced':
                components = await self._process_enhanced_manifest(manifest_data, errors, warnings)
            elif manifest_type == 'simple':
                components = await self._process_simple_manifest(manifest_data, errors, warnings)
            else:
                error_msg = f'Unknown manifest type: {manifest_type}'
                errors.append(error_msg)
                if self.strict_mode:
                    raise ValueError(error_msg)
            
            success = len(errors) == 0
            return ManifestProcessingResult(
                success=success,
                manifest_path=manifest_path,
                manifest_type=manifest_type,
                errors=errors,
                warnings=warnings,
                components=components
            )
            
        except Exception as e:
            error_msg = f'Critical error processing manifest {manifest_path}: {e}'
            logger.error(error_msg, exc_info=True)
            return ManifestProcessingResult(
                success=False,
                manifest_path=manifest_path,
                manifest_type='unknown',
                errors=[error_msg],
                warnings=warnings,
                components=[]
            )

    async def _validate_schema(self, manifest_data: Dict[str, Any], manifest_type: str, manifest_path: Path) -> List[str]:
        """Validate manifest against JSON schema"""
        errors = []
        
        try:
            try:
                import jsonschema
            except ImportError:
                if self.strict_mode:
                    errors.append('jsonschema package required for manifest validation in strict mode')
                    return errors
                else:
                    logger.warning('jsonschema not available - skipping schema validation')
                    return []
            
            schema = await self._load_schema(manifest_type)
            if not schema:
                if self.strict_mode:
                    errors.append(f'No schema available for manifest type: {manifest_type}')
                else:
                    logger.warning(f'No schema available for manifest type: {manifest_type}')
                return errors
            
            try:
                jsonschema.validate(instance=manifest_data, schema=schema)
                logger.debug(f'✓ Schema validation passed for {manifest_path}')
            except jsonschema.ValidationError as e:
                error_msg = f'Schema validation failed: {e.message}'
                if hasattr(e, 'absolute_path') and e.absolute_path:
                    error_msg += f" at path: {'.'.join(str(p) for p in e.absolute_path)}"
                errors.append(error_msg)
                logger.error(f'Schema validation error in {manifest_path}: {error_msg}')
            except jsonschema.SchemaError as e:
                error_msg = f'Invalid schema: {e.message}'
                errors.append(error_msg)
                logger.error(f'Schema error: {error_msg}')
                
        except Exception as e:
            error_msg = f'Error during schema validation: {e}'
            errors.append(error_msg)
            logger.error(error_msg)
        
        return errors

    async def _load_schema(self, manifest_type: str) -> Optional[Dict[str, Any]]:
        """Load JSON schema for manifest type"""
        if manifest_type in self._schema_cache:
            return self._schema_cache[manifest_type]
        
        # Updated schema path construction as mentioned in instructions
        schema_filename = f'{manifest_type}_manifest.schema.json'
        schema_path = self.package_root / 'schemas' / schema_filename
        
        if not schema_path.exists():
            # Fallback to generic schema
            alt_schema_path = self.package_root / 'schemas' / 'manifest.schema.json'
            if alt_schema_path.exists():
                schema_path = alt_schema_path
                logger.debug(f"Schema for type '{manifest_type}' not found, using generic {alt_schema_path}")
            else:
                logger.warning(f'Schema file not found for manifest type: {manifest_type}')
                return None
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            self._schema_cache[manifest_type] = schema
            logger.debug(f'Loaded schema from: {schema_path}')
            return schema
        except Exception as e:
            logger.error(f'Failed to load schema from {schema_path}: {e}')
            return None

    async def _process_enhanced_manifest(self, manifest_data: Dict[str, Any], errors: List[str], warnings: List[str]) -> List['ComponentSpec']:
        """Process enhanced manifest format - handle both dict and list formats"""
        components = []
        
        # Define sections and their types
        sections = {
            'shared_services': 'shared_service',
            'mechanisms': 'mechanism',
            'observers': 'observer', 
            'managers': 'manager',
            'composites': 'composite',
            'orchestration_commands': 'orchestration_command'
        }
        
        for section_name, component_type in sections.items():
            section_data = manifest_data.get(section_name, {})
            if not section_data:
                continue
            
            logger.debug(f'Processing {section_name}: {type(section_data).__name__} with {len(section_data)} items')
            
            # Handle both dict format (shared_services) and list format (mechanisms, observers, managers)
            if isinstance(section_data, dict):
                # Dictionary format: key is component_id, value is spec
                for component_id, spec in section_data.items():
                    try:
                        if not isinstance(spec, dict):
                            error_msg = f"Component spec for '{component_id}' must be a dictionary"
                            errors.append(error_msg)
                            continue
                        
                        if not spec.get('enabled', True):
                            logger.debug(f'Skipping disabled component: {component_id}')
                            continue
                        
                        component_spec = ComponentSpec(
                            component_id=component_id,
                            component_type=component_type,
                            section_name=section_name,
                            spec_data=spec,
                            manifest_type='enhanced'
                        )
                        
                        spec_errors = self._validate_component_spec(component_spec)
                        if spec_errors:
                            errors.extend(spec_errors)
                            if self.strict_mode:
                                continue
                        
                        components.append(component_spec)
                        
                    except Exception as e:
                        error_msg = f"Error processing component '{component_id}': {e}"
                        errors.append(error_msg)
                        logger.error(error_msg)
            
            elif isinstance(section_data, list):
                # List format: each item is a dict with component specification
                for i, item in enumerate(section_data):
                    try:
                        if not isinstance(item, dict):
                            error_msg = f"Item {i} in section '{section_name}' must be a dictionary"
                            errors.append(error_msg)
                            continue
                        
                        # Extract component_id from the item
                        component_id = item.get('id') or item.get('component_id')
                        if not component_id:
                            error_msg = f"Item {i} in section '{section_name}' missing 'id' or 'component_id'"
                            errors.append(error_msg)
                            continue
                        
                        if not item.get('enabled', True):
                            logger.debug(f'Skipping disabled component: {component_id}')
                            continue
                        
                        component_spec = ComponentSpec(
                            component_id=component_id,
                            component_type=component_type,
                            section_name=section_name,
                            spec_data=item,
                            manifest_type='enhanced'
                        )
                        
                        spec_errors = self._validate_component_spec(component_spec)
                        if spec_errors:
                            errors.extend(spec_errors)
                            if self.strict_mode:
                                continue
                        
                        components.append(component_spec)
                        
                    except Exception as e:
                        error_msg = f"Error processing item {i} in section '{section_name}': {e}"
                        errors.append(error_msg)
                        logger.error(error_msg)
            
            else:
                error_msg = f"Section '{section_name}' must be a dictionary or list, got {type(section_data).__name__}"
                errors.append(error_msg)
        
        return components

    async def _process_simple_manifest(self, manifest_data: Dict[str, Any], errors: List[str], warnings: List[str]) -> List['ComponentSpec']:
        """Process simple manifest format"""
        components = []
        
        component_list = manifest_data.get('components', [])
        if not isinstance(component_list, list):
            errors.append("Simple manifest 'components' must be a list")
            return components
        
        for i, comp_def in enumerate(component_list):
            try:
                if not isinstance(comp_def, dict):
                    error_msg = f'Component definition {i} must be a dictionary'
                    errors.append(error_msg)
                    continue
                
                component_id = comp_def.get('component_id', f'component_{i}')
                component_type = comp_def.get('type', 'unknown')
                
                if not comp_def.get('enabled', True):
                    logger.debug(f'Skipping disabled component: {component_id}')
                    continue
                
                component_spec = ComponentSpec(
                    component_id=component_id,
                    component_type=component_type,
                    section_name='components',
                    spec_data=comp_def,
                    manifest_type='simple'
                )
                
                spec_errors = self._validate_component_spec(component_spec)
                if spec_errors:
                    errors.extend(spec_errors)
                    if self.strict_mode:
                        continue
                
                components.append(component_spec)
                
            except Exception as e:
                error_msg = f'Error processing simple component {i}: {e}'
                errors.append(error_msg)
                logger.error(error_msg)
        
        return components

    def _validate_component_spec(self, component_spec: 'ComponentSpec') -> List[str]:
        """Validate component specification"""
        errors = []
        spec_data = component_spec.spec_data
        
        if not component_spec.component_id:
            errors.append('Component must have a non-empty ID')
        
        if component_spec.manifest_type == 'enhanced':
            if component_spec.component_type in ['mechanism', 'observer', 'manager', 'composite']:
                if 'class' not in spec_data:
                    errors.append(f"Component '{component_spec.component_id}' missing required 'class' field")
                if 'metadata_definition' not in spec_data:
                    errors.append(f"Component '{component_spec.component_id}' missing required 'metadata_definition' field")
            elif component_spec.component_type == 'shared_service':
                if 'class' not in spec_data:
                    errors.append(f"Shared service '{component_spec.component_id}' missing required 'class' field")
            elif component_spec.component_type == 'orchestration_command':
                if 'class' not in spec_data:
                    errors.append(f"Orchestration command '{component_spec.component_id}' missing required 'class' field")
        
        elif component_spec.manifest_type == 'simple':
            if 'component_id' not in spec_data:
                errors.append("Simple component missing 'component_id' field")
            if 'type' not in spec_data:
                errors.append(f"Simple component '{component_spec.component_id}' missing 'type' field")
            if not spec_data.get('factory_key') and not spec_data.get('class'):
                errors.append(f"Simple component '{component_spec.component_id}' needs either 'factory_key' or 'class'")
        
        return errors


class ComponentSpec:
    def __init__(self, component_id: str, component_type: str, section_name: str, spec_data: Dict[str, Any], manifest_type: str):
        self.component_id = component_id
        self.component_type = component_type
        self.section_name = section_name
        self.spec_data = spec_data
        self.manifest_type = manifest_type

    def __repr__(self) -> str:
        return f"ComponentSpec(id='{self.component_id}', type='{self.component_type}', manifest='{self.manifest_type}')"


class ManifestProcessingResult:
    def __init__(self, success: bool, manifest_path: Path, manifest_type: str, errors: List[str], warnings: List[str], components: List[ComponentSpec]):
        self.success = success
        self.manifest_path = manifest_path
        self.manifest_type = manifest_type
        self.errors = errors
        self.warnings = warnings
        self.components = components

    @property
    def component_count(self) -> int:
        return len(self.components)

    def get_components_by_type(self, component_type: str) -> List[ComponentSpec]:
        return [c for c in self.components if c.component_type == component_type]