#!/usr/bin/env python3
"""
Debug trust score storage issue
"""

import sqlite3
import json
from pathlib import Path

def debug_trust_scores(db_path="runtime/nireon_ideas.db"):
    conn = sqlite3.connect(db_path)
    
    print("=== DEBUGGING TRUST SCORE STORAGE ===\n")
    
    # 1. Raw payload inspection
    print("1. RAW TRUST ASSESSMENT PAYLOADS")
    print("-" * 80)
    
    cursor = conn.execute("""
        SELECT 
            event_id,
            trust_score,
            payload
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
        LIMIT 3
    """)
    
    for row in cursor:
        print(f"\nEvent ID: {row[0]}")
        print(f"DB trust_score column: {row[1]}")
        
        payload = json.loads(row[2])
        print(f"Payload keys: {list(payload.keys())}")
        
        # Look for trust score in various places
        if 'trust_score' in payload:
            print(f"  payload['trust_score']: {payload['trust_score']}")
        if 'assessment_details' in payload:
            details = payload['assessment_details']
            if 'trust_score' in details:
                print(f"  payload['assessment_details']['trust_score']: {details['trust_score']}")
        
        # Show relevant payload fields
        for key in ['target_id', 'target_type', 'is_stable', 'rejection_reason']:
            if key in payload:
                print(f"  payload['{key}']: {payload[key]}")
    
    # 2. Extract trust scores from payload JSON
    print("\n\n2. TRUST SCORES EXTRACTED FROM JSON")
    print("-" * 80)
    
    cursor = conn.execute("""
        SELECT 
            json_extract(payload, '$.trust_score') as payload_trust,
            json_extract(payload, '$.assessment_details.trust_score') as details_trust,
            json_extract(payload, '$.is_stable') as is_stable,
            json_extract(payload, '$.target_id') as idea_id,
            substr(json_extract(payload, '$.idea_text'), 1, 50) as idea_preview
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
    """)
    
    print(f"{'Payload Trust':>15} | {'Details Trust':>15} | {'Stable':>8} | {'Idea ID':>40}")
    print("-" * 85)
    
    for row in cursor:
        print(f"{row[0] or 'None':>15} | {row[1] or 'None':>15} | {row[2] or 'None':>8} | {row[3] or 'None':>40}")
    
    # 3. Check if trust > 6.0 exists in payload
    print("\n\n3. HIGH TRUST IDEAS (FROM PAYLOAD)")
    print("-" * 80)
    
    cursor = conn.execute("""
        SELECT 
            json_extract(payload, '$.trust_score') as trust,
            json_extract(payload, '$.is_stable') as stable,
            json_extract(payload, '$.target_id') as idea_id,
            datetime(created_at, 'unixepoch') as time
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
        AND CAST(json_extract(payload, '$.trust_score') AS REAL) > 6.0
    """)
    
    count = 0
    for row in cursor:
        count += 1
        print(f"Trust: {row[0]}, Stable: {row[1]}, Idea: {row[2]}, Time: {row[3]}")
    
    if count == 0:
        print("No ideas with trust > 6.0 found in payloads")
    else:
        print(f"\nFound {count} ideas with trust > 6.0")
    
    # 4. Check reactor rule conditions
    print("\n\n4. CHECKING REACTOR RULE CONDITIONS")
    print("-" * 80)
    
    cursor = conn.execute("""
        SELECT 
            json_extract(payload, '$.trust_score') as trust,
            json_extract(payload, '$.is_stable') as stable,
            CASE 
                WHEN CAST(json_extract(payload, '$.trust_score') AS REAL) > 6.0 
                     AND json_extract(payload, '$.is_stable') = 1
                THEN 'YES - Should trigger quantifier'
                ELSE 'NO - Should not trigger'
            END as should_trigger_quantifier,
            json_extract(payload, '$.target_id') as idea_id
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
        ORDER BY created_at DESC
    """)
    
    for row in cursor:
        print(f"Trust: {row[0] or 'None':>20}, Stable: {row[1] or 'None':>5}, Trigger?: {row[2]:>25}, Idea: {row[3] or 'None'}")
    conn.close()

if __name__ == "__main__":
    debug_trust_scores()