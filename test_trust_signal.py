"""
Diagnostic script to understand TrustAssessmentSignal structure
"""

import json
from pathlib import Path
import re

def analyze_reactor_logs():
    """Analyze reactor logs to understand signal structure"""
    
    print("=== TRUST ASSESSMENT SIGNAL DIAGNOSTIC ===\n")
    
    # Look for the most recent log file
    runtime_dir = Path("runtime")
    log_files = list(runtime_dir.glob("*.log"))
    
    if not log_files:
        print("No log files found in runtime directory")
        return
    
    # Read the most recent log
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    print(f"Analyzing log file: {latest_log}\n")
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with error handling
        with open(latest_log, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            print("Warning: Some characters in the log file were replaced due to encoding issues\n")
    
    # Look for TrustAssessmentSignal processing
    trust_signal_pattern = r'Processing signal TrustAssessmentSignal.*?(?=Processing signal|$)'
    matches = re.findall(trust_signal_pattern, content, re.DOTALL)
    
    print(f"Found {len(matches)} TrustAssessmentSignal processing entries\n")
    
    # Look for REL Expression Debug sections
    rel_debug_pattern = r'=== REL Expression Debug ===(.*?)=== End REL Debug ==='
    rel_matches = re.findall(rel_debug_pattern, content, re.DOTALL)
    
    # Analyze REL debug info for TrustAssessmentSignal
    trust_rel_debugs = []
    for match in rel_matches:
        if 'TrustAssessmentSignal' in match:
            trust_rel_debugs.append(match)
    
    if trust_rel_debugs:
        print("=== TrustAssessmentSignal Structure from REL Debug ===")
        # Get the most recent one
        latest_debug = trust_rel_debugs[-1]
        
        # Extract key information
        lines = latest_debug.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['Signal type:', 'signal.payload type:', 
                                                   'signal.payload keys:', "signal.payload['payload']",
                                                   'trust_score:', 'is_stable:']):
                print(line.strip())
    
    # Look for actual trust scores in the logs
    print("\n=== Trust Scores Found in Logs ===")
    trust_score_pattern = r'trust_score=(\d+\.\d+)'
    scores = re.findall(trust_score_pattern, content)
    for score in scores[-10:]:  # Last 10 scores
        print(f"  trust_score={score}")
    
    # Check what rules matched
    print("\n=== Rules Matching TrustAssessmentSignal ===")
    rule_match_pattern = r"✅ Rule '([^']+)' MATCHED signal 'TrustAssessmentSignal'"
    rule_matches = re.findall(rule_match_pattern, content)
    for rule in set(rule_matches):
        print(f"  - {rule}")
    
    # Check if quantifier rule was evaluated
    print("\n=== Quantifier Rule Evaluation ===")
    quantifier_patterns = [
        r"Rule 'route_business_idea_to_quantifier[^']*'.*?for signal 'TrustAssessmentSignal'[^\\n]*",
        r"Rule 'route_high_trust_to_quantifier[^']*'.*?for signal 'TrustAssessmentSignal'[^\\n]*",
        r"quantifier.*?skipped.*?Reason: ([^\\n]+)"
    ]
    
    for pattern in quantifier_patterns:
        matches = re.findall(pattern, content)
        if matches:
            print(f"Pattern: {pattern}")
            for match in matches[-5:]:  # Last 5 matches
                print(f"  {match}")

if __name__ == "__main__":
    analyze_reactor_logs()
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Add the debug rule from 'debug_trust_signal_rule' to see exact signal structure")
    print("2. Check that 'quantifier_agent_primary' is enabled in standard.yaml")
    print("3. Verify the rule priority - it might be running after another rule that consumes the signal")
    print("4. Try the simpler rule from 'corrected_quantifier_rule' that uses direct payload access")