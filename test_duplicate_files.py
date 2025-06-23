#!/usr/bin/env python3
"""
Verify which reactor rules directory is actually being loaded
"""
import asyncio
import sys
from pathlib import Path

# Find project root
def _find_project_root(markers=['bootstrap', 'domain', 'core', 'configs']):
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if all((candidate / m).is_dir() for m in markers):
            return candidate
    return None

PROJECT_ROOT = _find_project_root()
if PROJECT_ROOT is None:
    print('ERROR: Could not determine the NIREON V4 project root.')
    sys.exit(1)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=== REACTOR RULES DIRECTORY INVESTIGATION ===\n")

# Check for duplicate rules directories
print("1. Checking for rules directories in the project:")
print("-" * 60)

rules_dirs = []
for pattern in ['**/reactor/rules', '**/configs/reactor/rules']:
    for path in PROJECT_ROOT.glob(pattern):
        if path.is_dir():
            rules_dirs.append(path)
            print(f"  📁 Found: {path.relative_to(PROJECT_ROOT)}")
            # List files in this directory
            yaml_files = list(path.glob('*.yaml'))
            if yaml_files:
                print(f"     Files: {[f.name for f in yaml_files]}")
            else:
                print(f"     Files: (empty)")

if len(rules_dirs) > 1:
    print("\n⚠️  WARNING: Multiple rules directories found! This can cause confusion.")

# Check if there's a stray core.yaml in the wrong place
print("\n\n2. Looking for all core.yaml files:")
print("-" * 60)
core_yamls = list(PROJECT_ROOT.rglob('core.yaml'))
for core_yaml in core_yamls:
    print(f"  📄 {core_yaml.relative_to(PROJECT_ROOT)}")
    # Check if it contains the simple_test_loop_finisher rule
    with open(core_yaml, 'r') as f:
        content = f.read()
        if 'simple_test_loop_finisher' in content:
            print(f"     ✓ Contains simple_test_loop_finisher rule")
        if 'route_business_idea_to_quantifier' in content:
            print(f"     ✓ Contains quantifier routing rule")

# Check the actual reactor configuration
print("\n\n3. Checking reactor configuration:")
print("-" * 60)

config_file = PROJECT_ROOT / 'configs' / 'default' / 'global_app_config.yaml'
if config_file.exists():
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    reactor_config = config.get('reactor', {})
    rules_path = reactor_config.get('rules_path', 'Not specified')
    print(f"  Configured rules_path: {rules_path}")
else:
    print(f"  ❌ Config file not found: {config_file}")

# Now let's trace what the actual bootstrap process would do
print("\n\n4. Simulating ReactorSetupPhase path resolution:")
print("-" * 60)

from pathlib import Path

# This mimics what ReactorSetupPhase does
rules_path_from_config = "configs/reactor/rules"
resolved_path = Path(rules_path_from_config)
if not resolved_path.is_absolute():
    resolved_path = PROJECT_ROOT / resolved_path

print(f"  Config says: {rules_path_from_config}")
print(f"  Resolves to: {resolved_path}")
print(f"  Exists: {resolved_path.exists()}")
if resolved_path.exists():
    yaml_files = list(resolved_path.glob('*.yaml'))
    print(f"  Contains: {[f.name for f in yaml_files]}")

# Check if the wrong directory might be imported
print("\n\n5. Checking Python import paths that might interfere:")
print("-" * 60)

# Look for imports of reactor.rules
import os
for root, dirs, files in os.walk(PROJECT_ROOT):
    for file in files:
        if file.endswith('.py'):
            filepath = Path(root) / file
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'reactor.rules' in content or 'reactor/rules' in content:
                        rel_path = filepath.relative_to(PROJECT_ROOT)
                        print(f"  ⚠️  {rel_path} imports from reactor.rules")
            except:
                pass

print("\n\n=== RECOMMENDATION ===")
if len(rules_dirs) > 1:
    wrong_dir = None
    for d in rules_dirs:
        if 'configs' not in str(d):
            wrong_dir = d
            break
    
    if wrong_dir:
        print(f"\n🔥 DELETE the duplicate rules directory: {wrong_dir.relative_to(PROJECT_ROOT)}")
        print(f"   This directory is likely causing the system to load the wrong rules!")
        print(f"\n✅ KEEP only: configs/reactor/rules/")
        print(f"   This is the correct location referenced in the configuration.")