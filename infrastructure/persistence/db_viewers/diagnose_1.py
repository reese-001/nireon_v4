#!/usr/bin/env python3
"""
Diagnose why Quantifier Agent is getting stuck
"""

import sqlite3
import json
from datetime import datetime
from collections import defaultdict

def diagnose_quantifier_issue(db_path: str = "runtime/nireon_ideas.db"):
    """Run diagnostic queries to identify the issue"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=== QUANTIFIER AGENT DIAGNOSTICS ===\n")
    
    # 1. Check for duplicate trust assessments
    print("1. CHECKING FOR DUPLICATE TRUST ASSESSMENTS")
    print("-" * 80)
    
    cursor.execute("""
        SELECT 
            target_id,
            COUNT(*) as signal_count,
            GROUP_CONCAT(event_id, ', ') as event_ids,
            MIN(timestamp) as first_time,
            MAX(timestamp) as last_time,
            trust_score
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
          AND trust_score > 6.0
          AND timestamp > datetime('now', '-1 hour')
        GROUP BY target_id, trust_score
        HAVING COUNT(*) > 1
        ORDER BY signal_count DESC
        LIMIT 10
    """)
    
    duplicates = cursor.fetchall()
    if duplicates:
        print(f"Found {len(duplicates)} ideas with duplicate trust assessments:")
        for row in duplicates:
            print(f"  Idea {row['target_id'][:8]}...: {row['signal_count']} signals")
            print(f"    Trust score: {row['trust_score']}")
            print(f"    Time span: {row['first_time']} to {row['last_time']}")
    else:
        print("No duplicate trust assessments found")
    
    # 2. Check quantifier-related signals
    print("\n2. QUANTIFIER-RELATED SIGNALS")
    print("-" * 80)
    
    # Check if Quantifier is emitting any signals
    cursor.execute("""
        SELECT 
            signal_type,
            COUNT(*) as count,
            MIN(timestamp) as first,
            MAX(timestamp) as last
        FROM events
        WHERE source_node_id = 'quantifier_agent_primary'
          AND timestamp > datetime('now', '-1 hour')
        GROUP BY signal_type
    """)
    
    quantifier_signals = cursor.fetchall()
    if quantifier_signals:
        print("Signals from Quantifier Agent:")
        for row in quantifier_signals:
            print(f"  {row['signal_type']}: {row['count']} signals")
    else:
        print("❌ No signals from quantifier_agent_primary found!")
    
    # 3. Check ProtoTaskSignal generation
    print("\n3. PROTO TASK SIGNALS")
    print("-" * 80)
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT json_extract(payload, '$.proto_block.id')) as unique_protos
        FROM events
        WHERE signal_type = 'ProtoTaskSignal'
          AND timestamp > datetime('now', '-1 hour')
    """)
    
    proto_stats = cursor.fetchone()
    print(f"ProtoTaskSignals: {proto_stats['total']} total, {proto_stats['unique_protos']} unique")
    
    # 4. Check for GenerativeLoopFinishedSignal
    print("\n4. LOOP COMPLETION SIGNALS")
    print("-" * 80)
    
    cursor.execute("""
        SELECT 
            source_node_id,
            COUNT(*) as count,
            json_extract(payload, '$.status') as status,
            json_extract(payload, '$.quantifier_triggered') as quantifier_triggered
        FROM events
        WHERE signal_type = 'GenerativeLoopFinishedSignal'
          AND timestamp > datetime('now', '-1 hour')
        GROUP BY source_node_id, status
    """)
    
    completion_signals = cursor.fetchall()
    if completion_signals:
        print("Loop completion signals:")
        for row in completion_signals:
            print(f"  {row['source_node_id']}: {row['count']} signals")
            print(f"    Status: {row['status']}")
            print(f"    Quantifier triggered: {row['quantifier_triggered']}")
    else:
        print("❌ No GenerativeLoopFinishedSignal found!")
    
    # 5. Analyze recent high-trust ideas flow
    print("\n5. RECENT HIGH-TRUST IDEA FLOW")
    print("-" * 80)
    
    cursor.execute("""
        SELECT 
            e1.target_id as idea_id,
            e1.trust_score,
            e1.timestamp as assessed_at,
            e2.signal_type as next_signal,
            e2.source_node_id as next_source,
            e2.timestamp as next_at,
            CAST((julianday(e2.timestamp) - julianday(e1.timestamp)) * 86400 AS INTEGER) as seconds_delay
        FROM events e1
        LEFT JOIN events e2 ON 
            e2.timestamp > e1.timestamp 
            AND e2.timestamp < datetime(e1.timestamp, '+5 minutes')
            AND (e2.payload LIKE '%' || e1.target_id || '%' OR e2.target_id = e1.target_id)
        WHERE e1.signal_type = 'TrustAssessmentSignal'
          AND e1.trust_score > 6.0
          AND e1.timestamp > datetime('now', '-30 minutes')
        ORDER BY e1.timestamp DESC, e2.timestamp
        LIMIT 20
    """)
    
    flows = cursor.fetchall()
    current_idea = None
    for flow in flows:
        if flow['idea_id'] != current_idea:
            current_idea = flow['idea_id']
            print(f"\nIdea {current_idea[:8]}... (trust={flow['trust_score']:.2f}):")
            print(f"  Assessed at: {flow['assessed_at']}")
        
        if flow['next_signal']:
            print(f"  → {flow['seconds_delay']}s later: {flow['next_signal']} from {flow['next_source']}")
    
    # 6. Check for stuck processes
    print("\n\n6. POTENTIAL STUCK POINTS")
    print("-" * 80)
    
    # Ideas assessed but not processed further
    cursor.execute("""
        WITH high_trust_ideas AS (
            SELECT DISTINCT target_id, trust_score, timestamp
            FROM events
            WHERE signal_type = 'TrustAssessmentSignal'
              AND trust_score > 6.0
              AND timestamp > datetime('now', '-1 hour')
        )
        SELECT 
            hti.target_id,
            hti.trust_score,
            hti.timestamp,
            COUNT(e.event_id) as followup_signals
        FROM high_trust_ideas hti
        LEFT JOIN events e ON 
            e.timestamp > hti.timestamp
            AND (e.payload LIKE '%' || hti.target_id || '%' OR e.target_id = hti.target_id)
            AND e.signal_type != 'TrustAssessmentSignal'
        GROUP BY hti.target_id
        HAVING followup_signals = 0
    """)
    
    stuck_ideas = cursor.fetchall()
    if stuck_ideas:
        print(f"❌ Found {len(stuck_ideas)} high-trust ideas with no follow-up processing:")
        for idea in stuck_ideas:
            print(f"  Idea {idea['target_id'][:8]}...: trust={idea['trust_score']:.2f}, assessed at {idea['timestamp']}")
    
    # 7. Check reactor rule execution
    print("\n7. REACTOR RULE MATCHING")
    print("-" * 80)
    
    # This is approximate - looking for patterns in signal timing
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT target_id) as unique_ideas,
            COUNT(*) as total_signals,
            AVG(trust_score) as avg_trust,
            MIN(trust_score) as min_trust,
            MAX(trust_score) as max_trust
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
          AND trust_score > 6.0
          AND timestamp > datetime('now', '-1 hour')
    """)
    
    stats = cursor.fetchone()
    print(f"High-trust assessments in last hour:")
    print(f"  Unique ideas: {stats['unique_ideas']}")
    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Trust scores: {stats['min_trust']:.2f} - {stats['max_trust']:.2f} (avg: {stats['avg_trust']:.2f})")
    
    conn.close()
    
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    if not quantifier_signals:
        print("❌ ISSUE: Quantifier Agent is not emitting any signals")
        print("   This suggests it's getting stuck during processing")
    
    if proto_stats['total'] == 0:
        print("❌ ISSUE: No ProtoTaskSignals found")
        print("   The Quantifier may be failing to generate proto requests")
    
    if stuck_ideas:
        print(f"❌ ISSUE: {len(stuck_ideas)} high-trust ideas are not being processed")
        print("   The reactor rules may not be triggering properly")
    
    if duplicates:
        print(f"⚠️  WARNING: Duplicate trust assessments may be causing multiple triggers")


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "runtime/nireon_ideas.db"
    diagnose_quantifier_issue(db_path)