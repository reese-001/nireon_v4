#!/usr/bin/env python3
"""
Quick diagnostic to check if llm_config.yaml is loading correctly
"""

import yaml
from pathlib import Path
import json

def check_llm_config():
    config_path = Path("configs/default/llm_config.yaml")
    
    print(f"Checking config at: {config_path.absolute()}")
    print(f"File exists: {config_path.exists()}")
    
    if not config_path.exists():
        print("ERROR: Config file not found!")
        return
        
    print(f"File size: {config_path.stat().st_size} bytes")
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            
        print(f"\nFirst 200 characters of file:")
        print(content[:200])
        
        # Try to parse YAML
        config = yaml.safe_load(content)
        
        print(f"\nYAML parsed successfully!")
        print(f"Top-level keys: {list(config.keys())}")
        
        if 'llm' in config:
            print(f"\nKeys under 'llm': {list(config['llm'].keys())}")
            
            if 'models' in config['llm']:
                print(f"\nModels found: {list(config['llm']['models'].keys())}")
                print("✅ Config structure looks correct!")
            else:
                print("\n❌ ERROR: 'models' key not found under 'llm'!")
                print("Available keys:", list(config['llm'].keys()))
        else:
            print("\n❌ ERROR: 'llm' key not found at top level!")
            
        # Pretty print the structure
        print("\n\nFull config structure (without values):")
        print(json.dumps(_get_structure(config), indent=2))
            
    except yaml.YAMLError as e:
        print(f"\n❌ YAML parsing error: {e}")
        print("This suggests a syntax error in the YAML file")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def _get_structure(obj, max_depth=3, current_depth=0):
    """Get structure of nested dict without values"""
    if current_depth >= max_depth:
        return "..."
    
    if isinstance(obj, dict):
        return {k: _get_structure(v, max_depth, current_depth + 1) for k, v in obj.items()}
    elif isinstance(obj, list):
        if obj:
            return [_get_structure(obj[0], max_depth, current_depth + 1), "..."]
        return []
    else:
        return type(obj).__name__

if __name__ == "__main__":
    check_llm_config()