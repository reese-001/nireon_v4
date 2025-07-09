#!/usr/bin/env python3
"""
Debug and fix trust assessment signal storage issues.
This script analyzes the trust assessment data flow and identifies where stability information is lost.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

def analyze_trust_assessment_storage(db_path: str = "runtime/nireon_ideas.db"):
    """Analyze how trust assessment signals are stored"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=== ANALYZING TRUST ASSESSMENT STORAGE ===\n")
    
    # 1. Check table schema
    print("1. CHECKING TABLE SCHEMA")
    print("-" * 80)
    cursor.execute("PRAGMA table_info(events)")
    columns = cursor.fetchall()
    print("Events table columns:")
    for col in columns:
        print(f"  {col['name']:20} {col['type']:15} {'NOT NULL' if col['notnull'] else 'NULL'}")
    
    # 2. Analyze raw trust assessment signals
    print("\n2. RAW TRUST ASSESSMENT SIGNALS")
    print("-" * 80)
    cursor.execute("""
        SELECT event_id, payload, trust_score, context_tags
        FROM events 
        WHERE signal_type = 'TrustAssessmentSignal'
        ORDER BY id DESC
        LIMIT 5
    """)
    
    signals = cursor.fetchall()
    for signal in signals:
        event_id = signal['event_id']
        payload = json.loads(signal['payload']) if signal['payload'] else {}
        
        print(f"\nEvent ID: {event_id}")
        print(f"DB trust_score column: {signal['trust_score']}")
        
        # Analyze payload structure
        print("Payload structure:")
        _print_dict_structure(payload, indent=2)
        
        # Extract trust assessment data from various possible locations
        trust_data = extract_trust_assessment_data(payload)
        print(f"\nExtracted trust data:")
        for key, value in trust_data.items():
            print(f"  {key}: {value}")
    
    # 3. Find where stability data might be
    print("\n3. SEARCHING FOR STABILITY DATA")
    print("-" * 80)
    
    # Check different payload paths
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN json_extract(payload, '$.is_stable') IS NOT NULL THEN 1 ELSE 0 END) as has_is_stable,
            SUM(CASE WHEN json_extract(payload, '$.assessment_details.is_stable') IS NOT NULL THEN 1 ELSE 0 END) as has_nested_stable,
            SUM(CASE WHEN json_extract(payload, '$.payload.is_stable') IS NOT NULL THEN 1 ELSE 0 END) as has_double_nested,
            SUM(CASE WHEN json_extract(payload, '$.stable') IS NOT NULL THEN 1 ELSE 0 END) as has_stable
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
    """)
    
    result = cursor.fetchone()
    print(f"Total TrustAssessmentSignals: {result['total']}")
    print(f"Has $.is_stable: {result['has_is_stable']}")
    print(f"Has $.assessment_details.is_stable: {result['has_nested_stable']}")
    print(f"Has $.payload.is_stable: {result['has_double_nested']}")
    print(f"Has $.stable: {result['has_stable']}")
    
    # 4. Trace the data flow
    print("\n4. TRACING DATA FLOW")
    print("-" * 80)
    
    # Look for Sentinel output pattern
    cursor.execute("""
        SELECT event_id, source_node_id, payload
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
          AND source_node_id LIKE '%sentinel%'
        LIMIT 3
    """)
    
    sentinel_signals = cursor.fetchall()
    print(f"\nFound {len(sentinel_signals)} signals from Sentinel")
    
    for signal in sentinel_signals:
        payload = json.loads(signal['payload'])
        print(f"\nSentinel signal {signal['event_id']}:")
        
        # Check for IdeaAssessment structure
        if 'assessment_details' in payload:
            details = payload['assessment_details']
            if isinstance(details, dict):
                print("  Has assessment_details dict")
                if 'is_stable' in details:
                    print(f"    is_stable: {details['is_stable']}")
                if 'rejection_reason' in details:
                    print(f"    rejection_reason: {details['rejection_reason']}")
                if 'axis_scores' in details:
                    print(f"    axis_scores: {len(details['axis_scores'])} axes")
    
    # 5. Proposed fix
    print("\n5. PROPOSED FIX")
    print("-" * 80)
    print("The issue appears to be that stability data is nested in the payload but not ")
    print("extracted to the database columns. The fix requires:")
    print("1. Update the event repository to extract is_stable from the correct path")
    print("2. Migrate existing data to populate the is_stable column")
    print("3. Ensure Sentinel properly includes is_stable in the signal")
    
    conn.close()


def extract_trust_assessment_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract trust assessment data from various possible locations in payload"""
    
    result = {
        'trust_score': None,
        'is_stable': None,
        'target_id': None,
        'target_type': None,
        'rejection_reason': None
    }
    
    # Direct fields
    if 'trust_score' in payload:
        result['trust_score'] = payload['trust_score']
    if 'is_stable' in payload:
        result['is_stable'] = payload['is_stable']
    if 'target_id' in payload:
        result['target_id'] = payload['target_id']
    if 'target_type' in payload:
        result['target_type'] = payload['target_type']
    
    # Check assessment_details
    if 'assessment_details' in payload:
        details = payload['assessment_details']
        if isinstance(details, dict):
            if 'is_stable' in details:
                result['is_stable'] = details['is_stable']
            if 'rejection_reason' in details:
                result['rejection_reason'] = details['rejection_reason']
            if 'trust_score' in details and result['trust_score'] is None:
                result['trust_score'] = details['trust_score']
    
    # Check nested payload (double wrapping)
    if 'payload' in payload and isinstance(payload['payload'], dict):
        nested = payload['payload']
        for key in ['trust_score', 'is_stable', 'target_id', 'target_type']:
            if key in nested and result[key] is None:
                result[key] = nested[key]
    
    # Alternative field names
    if result['target_id'] is None:
        result['target_id'] = payload.get('idea_id')
    
    return result


def _print_dict_structure(d: Dict[str, Any], indent: int = 0):
    """Print dictionary structure with types"""
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}: dict ({len(value)} keys)")
            if indent < 3:  # Limit depth
                _print_dict_structure(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}: list ({len(value)} items)")
        else:
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            print(f"{prefix}{key}: {type(value).__name__} = {value_str}")


def fix_trust_assessment_storage(db_path: str = "runtime/nireon_ideas.db"):
    """Apply fixes to properly store trust assessment data"""
    
    print("\n=== APPLYING TRUST ASSESSMENT STORAGE FIXES ===\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Add missing columns if needed
    print("1. Ensuring columns exist...")
    cursor.execute("PRAGMA table_info(events)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    columns_to_add = [
        ('is_stable', 'BOOLEAN'),
        ('target_id', 'TEXT'),
        ('target_type', 'TEXT'),
        ('assessment_rationale', 'TEXT')
    ]
    
    for col_name, col_type in columns_to_add:
        if col_name not in existing_columns:
            print(f"  Adding column: {col_name} {col_type}")
            cursor.execute(f'ALTER TABLE events ADD COLUMN {col_name} {col_type}')
    
    # 2. Update existing records
    print("\n2. Updating existing TrustAssessmentSignal records...")
    
    # Count records to update
    cursor.execute("""
        SELECT COUNT(*) FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
          AND is_stable IS NULL
    """)
    count = cursor.fetchone()[0]
    print(f"  Found {count} records to update")
    
    if count > 0:
        # Update in batches
        batch_size = 100
        updated = 0
        
        while updated < count:
            cursor.execute("""
                SELECT id, payload FROM events
                WHERE signal_type = 'TrustAssessmentSignal'
                  AND is_stable IS NULL
                LIMIT ?
            """, (batch_size,))
            
            rows = cursor.fetchall()
            if not rows:
                break
            
            for row in rows:
                event_id = row[0]
                payload = json.loads(row[1]) if row[1] else {}
                
                # Extract trust assessment data
                data = extract_trust_assessment_data(payload)
                
                # Update the record
                cursor.execute("""
                    UPDATE events
                    SET is_stable = ?,
                        target_id = ?,
                        target_type = ?,
                        trust_score = COALESCE(trust_score, ?)
                    WHERE id = ?
                """, (
                    data['is_stable'],
                    data['target_id'],
                    data['target_type'],
                    data['trust_score'],
                    event_id
                ))
                
                updated += 1
            
            conn.commit()
            print(f"  Updated {updated}/{count} records...")
    
    # 3. Create missing indexes
    print("\n3. Creating indexes for better performance...")
    indexes = [
        ('idx_events_trust_stable', 'events(trust_score, is_stable)', 'signal_type = "TrustAssessmentSignal"'),
        ('idx_events_high_trust', 'events(trust_score DESC)', 'trust_score > 6.0')
    ]
    
    for idx_name, idx_cols, where_clause in indexes:
        try:
            if where_clause:
                cursor.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_cols} WHERE {where_clause}')
            else:
                cursor.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_cols}')
            print(f"  Created index: {idx_name}")
        except sqlite3.OperationalError:
            print(f"  Index {idx_name} already exists")
    
    # 4. Verify the fix
    print("\n4. Verifying the fix...")
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN is_stable IS NOT NULL THEN 1 ELSE 0 END) as has_stable,
            SUM(CASE WHEN is_stable = 1 THEN 1 ELSE 0 END) as stable_count,
            SUM(CASE WHEN is_stable = 0 THEN 1 ELSE 0 END) as unstable_count,
            SUM(CASE WHEN trust_score > 6.0 AND is_stable = 1 THEN 1 ELSE 0 END) as high_trust_stable
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
    """)
    
    result = cursor.fetchone()
    print(f"  Total TrustAssessmentSignals: {result[0]}")
    print(f"  Has stability data: {result[1]} ({result[1]/result[0]*100:.1f}%)")
    print(f"  Stable ideas: {result[2]}")
    print(f"  Unstable ideas: {result[3]}")
    print(f"  High trust + stable: {result[4]}")
    
    conn.commit()
    conn.close()
    
    print("\nFix applied successfully!")


def check_reactor_rules(db_path: str = "runtime/nireon_ideas.db"):
    """Check if reactor rules would trigger based on the data"""
    
    print("\n=== CHECKING REACTOR RULE CONDITIONS ===\n")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get sample high trust events
    cursor.execute("""
        SELECT event_id, trust_score, is_stable, target_id, payload
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
          AND trust_score > 6.0
        ORDER BY trust_score DESC
        LIMIT 10
    """)
    
    events = cursor.fetchall()
    
    print(f"Found {len(events)} high trust (>6.0) events\n")
    
    for event in events:
        trust = event['trust_score']
        stable = event['is_stable']
        target_id = event['target_id']
        
        # Check rule conditions
        print(f"Event: {event['event_id']}")
        print(f"  Trust: {trust:.2f}, Stable: {stable}, Target: {target_id}")
        
        # Rule: route_high_trust_to_quantifier
        # Conditions: trust_score > 6.0
        if trust > 6.0:
            print("  ✓ Would trigger: route_high_trust_to_quantifier")
        
        # Rule: low_trust_completion  
        # Conditions: trust_score <= 6.0 or is_stable == False
        if trust <= 6.0 or stable is False:
            print("  ✓ Would trigger: low_trust_completion")
        
        # Check payload structure for rule compatibility
        payload = json.loads(event['payload']) if event['payload'] else {}
        if 'target_id' in payload:
            print("  ✓ Has target_id in payload (required for quantifier)")
        else:
            print("  ✗ Missing target_id in payload")
        
        print()
    
    conn.close()


if __name__ == "__main__":
    import sys
    
    # Default database path
    db_path = "runtime/nireon_ideas.db"
    
    # Check if custom path provided
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    # Run analysis
    analyze_trust_assessment_storage(db_path)
    
    # Apply fixes
    print("\nDo you want to apply the fixes? (y/n): ", end="")
    if input().lower() == 'y':
        fix_trust_assessment_storage(db_path)
        
        # Check reactor rules
        check_reactor_rules(db_path)