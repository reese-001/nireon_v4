#!/usr/bin/env python3
"""
Test the payload template to see what's going wrong.
"""
import json
import yaml
from string import Template
from pathlib import Path

def test_payload_template():
    """Test the payload template independently."""
    
    # Find the config file (from tests/llm_subsystem/ directory)
    script_dir = Path(__file__).parent
    config_paths = [
        script_dir / "../../configs/default/llm_config.yaml",
        script_dir.parent.parent / "configs/default/llm_config.yaml",
        Path("configs/default/llm_config.yaml"),
        Path("llm_config.yaml")
    ]
    
    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break
    
    if not config_file:
        print("❌ Could not find llm_config.yaml")
        print("Searched in:")
        for path in config_paths:
            print(f"  - {path.resolve()}")
        return
    
    print(f"📁 Using config file: {config_file.resolve()}")
    
    # Load the config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'models' not in config or 'nano_default' not in config['models']:
        print("❌ nano_default model not found in config")
        return
        
    nano_config = config['models']['nano_default']
    template_str = nano_config.get('payload_template', '')
    
    print("\n=== TEMPLATE STRING ===")
    print(repr(template_str))
    print("\n=== TEMPLATE STRING (formatted) ===")
    print(template_str)
    
    # Test variables
    template_vars = {
        'model_name_for_api': 'gpt-4o-mini',
        'system_prompt': 'You are a helpful assistant.',
        'prompt': 'Say hello',
        'temperature': 0.7,
        'top_p': 1.0,
        'max_tokens': 100
    }
    
    print("\n=== TEMPLATE VARIABLES ===")
    for k, v in template_vars.items():
        print(f"{k}: {v} ({type(v)})")
    
    try:
        # Convert {{ variable }} syntax to $variable syntax for Python Template
        import re
        def replace_braces(match):
            var_name = match.group(1).strip()
            return f"${var_name}"
        
        converted_template_str = re.sub(r'\{\{\s*(\w+)\s*\}\}', replace_braces, template_str)
        
        print(f"\n=== CONVERTED TEMPLATE ({{ }} -> $ syntax) ===")
        print(converted_template_str)
        
        # Apply template
        template = Template(converted_template_str)
        payload_str = template.safe_substitute(**template_vars)
        
        print("\n=== GENERATED PAYLOAD STRING ===")
        print(repr(payload_str))
        print("\n=== FORMATTED PAYLOAD STRING ===")
        print(payload_str)
        
        # Try to parse JSON
        payload = json.loads(payload_str)
        
        print("\n=== PARSED JSON ===")
        print(json.dumps(payload, indent=2))
        print("\n✅ SUCCESS: Template generates valid JSON")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Error type: {type(e)}")
        
        # Try to find the exact position of the error
        if "line" in str(e) and "column" in str(e):
            lines = payload_str.split('\n')
            print(f"\nPayload string has {len(lines)} lines")
            for i, line in enumerate(lines, 1):
                print(f"Line {i}: {repr(line)}")
                
        # Show character-by-character analysis around the error
        try:
            # Extract line and column from error message
            import re
            match = re.search(r'line (\d+) column (\d+)', str(e))
            if match:
                error_line = int(match.group(1))
                error_col = int(match.group(2))
                
                print(f"\n🔍 ERROR AT LINE {error_line}, COLUMN {error_col}")
                lines = payload_str.split('\n')
                if error_line <= len(lines):
                    line = lines[error_line - 1]
                    print(f"Line content: {repr(line)}")
                    if error_col <= len(line):
                        print(f"Character at error: {repr(line[error_col-1:error_col+5])}")
                        print(f"Position marker: {' ' * (error_col-1)}^")
        except:
            pass

if __name__ == "__main__":
    test_payload_template()