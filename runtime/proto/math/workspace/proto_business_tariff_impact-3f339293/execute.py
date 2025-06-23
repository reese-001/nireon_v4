
import sys
import os
import json
import time
import traceback
from pathlib import Path

# --- User-provided Code Block ---
# The code is loaded from a JSON-safe string to avoid escaping issues
user_code = "import numpy as np\nimport matplotlib.pyplot as plt\n\ndef calculate_tariff_impact(cogs_percentage, tariff_percentage, scenarios):\n    revenue = 100  # Assume revenue is 100 for simplicity\n    cogs = cogs_percentage / 100 * revenue\n    tariff_cost = tariff_percentage / 100 * revenue\n    \n    margins = {}\n    for scenario in scenarios:\n        price_increase = tariff_cost * scenario\n        new_revenue = revenue + price_increase\n        new_cogs = cogs\n        gross_margin = (new_revenue - new_cogs) / new_revenue * 100\n        margins[f\"scenario_{int(scenario * 100)}%\"] = gross_margin\n    \n    # Visualization\n    plt.bar(margins.keys(), margins.values(), color=['blue', 'orange', 'green'])\n    plt.title('Impact of Tariff on Gross Margin')\n    plt.ylabel('Gross Margin (%)')\n    plt.xlabel('Tariff Cost Absorption Scenarios')\n    plt.savefig('tariff_impact.png')\n    plt.close()\n    \n    return margins\n\n# Example call (not part of the function)\n# result = calculate_tariff_impact(70, 25, [0, 0.5, 1])"
exec(user_code, globals())
# ------------------------------

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            inputs_file_path = sys.argv[1]
            with open(inputs_file_path, 'r', encoding='utf-8') as f:
                inputs = json.load(f)
        else:
            inputs = {}

        entry_point = locals().get('calculate_tariff_impact')
        if not callable(entry_point):
            raise NameError("Entry point function 'calculate_tariff_impact' not found or not callable.")

        result = entry_point(**inputs)
        
        # --- FIX: Add requirements.txt to the ignore list ---
        ignore_list = ['execute.py', 'inputs.json', 'requirements.txt']
        artifacts = [
            str(p) for p in Path(".").iterdir() 
            if p.is_file() and p.name not in ignore_list and not p.name.startswith('.')
        ]
        
        output = {
            "success": True,
            "result": result,
            "artifacts": artifacts
        }
        # Use an f-string to embed the JSON to avoid potential escaping issues with print
        print(f"RESULT_JSON:{json.dumps(output, default=str)}")
        
    except Exception as e:
        error_output = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"RESULT_JSON:{json.dumps(error_output)}")
