#!/usr/bin/env python3
"""
Check what data the Quantifier is receiving
"""

import sqlite3
import json
from datetime import datetime

def check_quantifier_input(db_path: str = "runtime/nireon_ideas.db"):
    """Check the trust assessment signal payloads"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=== CHECKING QUANTIFIER INPUT DATA ===\n")
    
    # 1. Check recent trust assessment payloads
    print("1. RECENT TRUST ASSESSMENT SIGNAL PAYLOADS")
    print("-" * 80)
    
    cursor.execute("""
        SELECT 
            event_id,
            target_id,
            trust_score,
            payload
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
          AND trust_score > 6.0
          AND timestamp > datetime('now', '-2 hours')
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        print(f"\nSignal: {row['event_id']}")
        print(f"Target: {row['target_id']}")
        print(f"Trust: {row['trust_score']}")
        
        payload = json.loads(row['payload'])
        print("Payload keys:", list(payload.keys()))
        
        # Check for idea_text
        idea_text = payload.get('idea_text', '')
        if idea_text:
            print(f"✓ Has idea_text: {idea_text[:100]}...")
        else:
            print("✗ Missing idea_text!")
            
        # Check assessment_details
        if 'assessment_details' in payload:
            details = payload['assessment_details']
            print(f"  assessment_details keys: {list(details.keys())}")
            if 'metadata' in details:
                meta = details['metadata']
                print(f"    metadata keys: {list(meta.keys())}")
    
    # 2. Check if ideas exist in ideas table
    print("\n\n2. CHECKING IF IDEAS EXIST IN DATABASE")
    print("-" * 80)
    
    cursor.execute("""
        SELECT 
            e.target_id,
            e.trust_score,
            i.idea_id,
            SUBSTR(i.text, 1, 100) as idea_text_preview
        FROM events e
        LEFT JOIN ideas i ON e.target_id = i.idea_id
        WHERE e.signal_type = 'TrustAssessmentSignal'
          AND e.trust_score > 6.0
          AND e.timestamp > datetime('now', '-2 hours')
        ORDER BY e.timestamp DESC
        LIMIT 10
    """)
    
    for row in cursor.fetchall():
        if row['idea_id']:
            print(f"✓ Idea {row['target_id'][:8]}... found: {row['idea_text_preview']}...")
        else:
            print(f"✗ Idea {row['target_id'][:8]}... NOT FOUND in ideas table!")
    
    # 3. Check what reactor is passing to Quantifier
    print("\n\n3. CHECKING REACTOR INPUT_DATA_MAPPING")
    print("-" * 80)
    print("\nThe reactor rule 'route_high_trust_to_quantifier' maps:")
    print('  idea_id: "payload.target_id"')
    print('  idea_text: "payload.idea_text"')
    print('  assessment_details: "payload"')
    print("\nThis means Quantifier receives:")
    print("  data['idea_id'] = signal.payload.target_id")
    print("  data['idea_text'] = signal.payload.idea_text")
    print("  data['assessment_details'] = signal.payload")
    
    # 4. Check for completed proto executions
    print("\n\n4. PROTO EXECUTION RESULTS")
    print("-" * 80)
    
    cursor.execute("""
        SELECT 
            source_node_id,
            json_extract(payload, '$.status') as status,
            json_extract(payload, '$.proto_block_id') as proto_id,
            timestamp
        FROM events
        WHERE signal_type IN ('ProtoResultSignal', 'MathProtoResultSignal', 'ProtoErrorSignal')
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    
    results = cursor.fetchall()
    if results:
        for row in results:
            print(f"{row['timestamp']}: {row['source_node_id']} - {row['status']} (proto: {row['proto_id']})")
    else:
        print("No proto execution results found")
    
    conn.close()
    
    print("\n" + "="*80)
    print("LIKELY ISSUE:")
    print("="*80)
    print("The TrustAssessmentSignal payload is missing 'idea_text', which the")
    print("Quantifier needs to analyze the idea. The Sentinel is not including")
    print("the idea text in the signal payload.")
    print("\nFIX: Update Sentinel to include idea_text in the TrustAssessmentSignal payload")


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "runtime/nireon_ideas.db"
    check_quantifier_input(db_path)