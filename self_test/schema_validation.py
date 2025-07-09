import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from glob import glob
from datetime import datetime

try:
    from jsonschema import validate as js_validate, ValidationError, Draft7Validator
except ImportError:
    print('Error: jsonschema package not installed. Run: pip install jsonschema')
    sys.exit(1)

class NireonValidator:
    def __init__(self, schema_dir: Path=None):
        if schema_dir is None:
            self.schema_dir = self._find_schema_dir()
        else:
            self.schema_dir = Path(schema_dir)
        self.manifest_schema_path = self.schema_dir / 'nireon_manifest.schema.json'
        self.rules_schema_path = self.schema_dir / 'nireon_rules.schema.json'
        self.manifest_schema = self._load_schema(self.manifest_schema_path)
        self.rules_schema = self._load_schema(self.rules_schema_path)
    
    def _find_schema_dir(self) -> Path:
        import sys
        import os
        if hasattr(sys.modules[__name__], '__file__'):
            current_file = Path(sys.modules[__name__].__file__).resolve()
        else:
            current_file = Path.cwd() / 'dummy.py'
        possible_locations = [current_file.parent, current_file.parent.parent, current_file.parent / 'schemas', current_file.parent.parent / 'schemas', Path.cwd(), Path.cwd() / 'schemas', Path.cwd().parent / 'schemas']
        if '__' in __name__:
            package_parts = __name__.split('.')
            if len(package_parts) > 1:
                package_root = current_file.parent
                for _ in range(len(package_parts) - 1):
                    package_root = package_root.parent
                possible_locations.append(package_root / 'schemas')
                possible_locations.append(package_root)
        seen = set()
        unique_locations = []
        for loc in possible_locations:
            loc_resolved = loc.resolve()
            if loc_resolved not in seen:
                seen.add(loc_resolved)
                unique_locations.append(loc)
        for location in unique_locations:
            manifest_path = location / 'nireon_manifest.schema.json'
            rules_path = location / 'nireon_rules.schema.json'
            if manifest_path.exists() and rules_path.exists():
                print(f'Found schemas in: {location}')
                return location
        raise FileNotFoundError('Could not find NIREON schema files. Searched in:\n' + '\n'.join((f'  - {loc}' for loc in unique_locations)) + "\n\nPlease ensure 'nireon_manifest.schema.json' and 'nireon_rules.schema.json' " + 'are in one of these locations or specify --schema-dir')
    
    def _load_schema(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f'Schema file not found: {path}')
        try:
            with path.open('r', encoding='utf-8') as f:
                schema = json.load(f)
                Draft7Validator.check_schema(schema)
                return schema
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in schema file {path}: {e}')
        except Exception as e:
            raise ValueError(f'Error loading schema {path}: {e}')
    
    def _load_yaml(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f'YAML file not found: {path}')
        try:
            with path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data is None:
                    raise ValueError('Empty YAML file')
                return data
        except yaml.YAMLError as e:
            raise ValueError(f'Invalid YAML in file {path}: {e}')
    
    def validate_file(self, data: Any, schema: Dict[str, Any], file_label: str) -> bool:
        try:
            js_validate(instance=data, schema=schema)
            print(f'✅ {file_label}: VALID')
            return True
        except ValidationError as e:
            print(f'❌ {file_label}: INVALID')
            print(f'   Error: {e.message}')
            if e.absolute_path:
                path_str = '.'.join((str(p) for p in e.absolute_path))
                print(f'   Location: {path_str}')
            if e.schema_path:
                schema_path_str = '.'.join((str(p) for p in e.schema_path))
                print(f'   Schema rule: {schema_path_str}')
            if hasattr(e, 'instance') and isinstance(e.instance, (str, int, float, bool, type(None))):
                print(f'   Value: {repr(e.instance)}')
            return False
        except Exception as e:
            print(f'❌ {file_label}: ERROR during validation')
            print(f'   {type(e).__name__}: {e}')
            return False
    
    def validate_manifest(self, manifest_path: Path) -> bool:
        try:
            data = self._load_yaml(manifest_path)
            return self.validate_file(data, self.manifest_schema, f'Manifest [{manifest_path}]')
        except Exception as e:
            print(f'❌ Failed to load manifest [{manifest_path}]: {e}')
            return False
    
    def validate_rules(self, rules_paths: List[Path]) -> List[Tuple[Path, bool]]:
        results = []
        for rule_path in rules_paths:
            try:
                data = self._load_yaml(rule_path)
                is_valid = self.validate_file(data, self.rules_schema, f'Rules [{rule_path}]')
                results.append((rule_path, is_valid))
            except Exception as e:
                print(f'❌ Failed to load rules [{rule_path}]: {e}')
                results.append((rule_path, False))
        return results
    
    def concatenate_rules(self, rules_paths: List[Path], output_dir: Path, only_if_valid: bool = True) -> Path:
        """Concatenate all rule files into a single timestamped file."""
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'concatenated_rules_{timestamp}.yaml'
        output_path = output_dir / output_filename
        
        concatenated_content = []
        
        # Add header
        concatenated_content.append(f"# Concatenated NIREON Rules")
        concatenated_content.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        concatenated_content.append(f"# Source files: {len(rules_paths)}")
        concatenated_content.append("")
        
        # Process each rule file
        for rule_path in rules_paths:
            try:
                # Read the file content
                with rule_path.open('r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Add file separator
                concatenated_content.append(f"# {'='*60}")
                concatenated_content.append(f"# Source: {rule_path}")
                concatenated_content.append(f"# {'='*60}")
                concatenated_content.append("")
                concatenated_content.append(content)
                concatenated_content.append("")  # Add spacing between files
                
            except Exception as e:
                print(f"⚠️  Warning: Could not read {rule_path} for concatenation: {e}")
                if only_if_valid:
                    print("   Skipping concatenation due to error.")
                    return None
        
        # Write concatenated content
        try:
            with output_path.open('w', encoding='utf-8') as f:
                f.write('\n'.join(concatenated_content))
            
            print(f"\n📄 Concatenated rules saved to: {output_path}")
            print(f"   Total size: {output_path.stat().st_size:,} bytes")
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving concatenated file: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(
        description='Validate NIREON YAML configuration files against JSON schemas.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --manifest app_manifest.yaml --rules core.yaml
  %(prog)s --manifest manifest.yaml --rules rules/*.yaml
  %(prog)s --schema-dir ./schemas --manifest config/manifest.yaml --rules config/rules/*.yaml
  
Windows examples:
  python -m self_test.validate_configs --manifest configs\\manifests\\standard.yaml --rules configs\\reactor\\rules\\*.yaml
  python -m self_test.validate_configs --schema-dir .\\schemas --manifest configs\\manifests\\standard.yaml --rules configs\\reactor\\rules\\*.yaml

Concatenation:
  The script will automatically concatenate all rule files and save them to ./rules directory with a timestamp.
  Use --no-concat to disable this feature, or --concat-dir to specify a different output directory.
        ''')
    
    parser.add_argument('--manifest', required=True, type=Path, help='Path to manifest YAML file')
    parser.add_argument('--rules', nargs='+', required=True, type=Path, help='Path(s) to reactor rule YAML file(s). Supports wildcards.')
    parser.add_argument('--schema-dir', type=Path, help='Directory containing the JSON schema files (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show more detailed output')
    parser.add_argument('--no-concat', action='store_true', help='Disable concatenation of rule files')
    parser.add_argument('--concat-dir', type=Path, default=Path('./self_test/rules'), help='Directory to save concatenated rules (default: ./rules)')
    parser.add_argument('--concat-always', action='store_true', help='Concatenate rules even if validation fails')
    
    args = parser.parse_args()
    
    rules_paths = []
    for pattern in args.rules:
        pattern_str = str(pattern)
        if '*' in pattern_str or '?' in pattern_str:
            from glob import glob
            matched_files = glob(pattern_str, recursive=False)
            if not matched_files:
                print(f"Warning: No files matched pattern '{pattern_str}'")
            else:
                rules_paths.extend([Path(f) for f in matched_files])
        else:
            rules_paths.append(pattern)
    
    if not rules_paths:
        print('Error: No rules files specified or found')
        sys.exit(1)
    
    try:
        validator = NireonValidator(schema_dir=args.schema_dir)
    except FileNotFoundError as e:
        print(f'Error: {e}')
        sys.exit(1)
    
    print('=' * 60)
    print('NIREON Configuration Validator')
    print('=' * 60)
    
    if args.verbose:
        print(f'\nSchema directory: {validator.schema_dir}')
        print(f'Manifest schema: {validator.manifest_schema_path}')
        print(f'Rules schema: {validator.rules_schema_path}')
    
    print(f'\nValidating {len(rules_paths) + 1} file(s)...\n')
    
    all_valid = True
    
    manifest_valid = validator.validate_manifest(args.manifest)
    all_valid &= manifest_valid
    
    print()
    
    rules_results = validator.validate_rules(rules_paths)
    for path, is_valid in rules_results:
        all_valid &= is_valid
    
    # Concatenate rules if enabled
    if not args.no_concat:
        print()  # Add spacing
        only_if_valid = not args.concat_always
        if all_valid or args.concat_always:
            validator.concatenate_rules(rules_paths, args.concat_dir, only_if_valid=only_if_valid)
        else:
            print("\n⚠️  Skipping concatenation due to validation failures.")
            print("   Use --concat-always to concatenate anyway.")
    
    print('\n' + '=' * 60)
    print('VALIDATION SUMMARY')
    print('=' * 60)
    
    valid_count = sum((1 for _, valid in [(args.manifest, manifest_valid)] + rules_results if valid))
    total_count = 1 + len(rules_results)
    
    print(f'\nTotal files checked: {total_count}')
    print(f'Valid files: {valid_count}')
    print(f'Invalid files: {total_count - valid_count}')
    
    if all_valid:
        print('\n✅ All validations PASSED!')
        sys.exit(0)
    else:
        print('\n❌ Some validations FAILED!')
        sys.exit(1)

if __name__ == '__main__':
    main()