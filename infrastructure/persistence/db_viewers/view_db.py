#!/usr/bin/env python3
"""
Enhanced NIREON database viewer with better analysis
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class EnhancedDBViewer:
    def __init__(self, db_path="runtime/nireon_ideas.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_event_summary(self):
        """Get overall event statistics"""
        with self.get_connection() as conn:
            # Total events by type
            df = pd.read_sql_query("""
                SELECT signal_type, COUNT(*) as count
                FROM events
                GROUP BY signal_type
                ORDER BY count DESC
            """, conn)
            
            print("=== Event Summary ===")
            print(df.to_string(index=False))
            print(f"\nTotal Events: {df['count'].sum()}")
            
            return df
    
    def show_trust_distribution(self):
        """Show distribution of trust scores"""
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    trust_score,
                    COUNT(*) as count,
                    json_extract(payload, '$.is_stable') as is_stable
                FROM events
                WHERE signal_type = 'TrustAssessmentSignal'
                AND trust_score IS NOT NULL
                GROUP BY trust_score, is_stable
                ORDER BY trust_score DESC
            """, conn)
            
            print("\n=== Trust Score Distribution ===")
            print(df.to_string(index=False))
            
            # Plot histogram
            trust_scores = pd.read_sql_query("""
                SELECT trust_score
                FROM events
                WHERE signal_type = 'TrustAssessmentSignal'
                AND trust_score IS NOT NULL
            """, conn)
            
            if not trust_scores.empty:
                plt.figure(figsize=(10, 6))
                plt.hist(trust_scores['trust_score'], bins=20, edgecolor='black')
                plt.axvline(x=6.0, color='red', linestyle='--', label='Quantifier Threshold')
                plt.xlabel('Trust Score')
                plt.ylabel('Count')
                plt.title('Trust Score Distribution')
                plt.legend()
                plt.savefig('trust_distribution.png')
                print("Trust distribution saved to trust_distribution.png")
    
    def show_idea_journey(self, limit=5):
        """Show the complete journey of recent ideas"""
        with self.get_connection() as conn:
            # Get recent ideas
            recent_ideas = pd.read_sql_query("""
                SELECT DISTINCT related_idea_id
                FROM events
                WHERE signal_type = 'IdeaGeneratedSignal'
                ORDER BY created_at DESC
                LIMIT ?
            """, conn, params=(limit,))
            
            print(f"\n=== Journey of {limit} Recent Ideas ===")
            
            for idea_id in recent_ideas['related_idea_id']:
                print(f"\n--- Idea: {idea_id} ---")
                
                events = pd.read_sql_query("""
                    SELECT 
                        datetime(created_at, 'unixepoch') as time,
                        signal_type,
                        source_node_id,
                        trust_score,
                        json_extract(payload, '$.is_stable') as is_stable,
                        json_extract(payload, '$.generation_method') as method
                    FROM events
                    WHERE related_idea_id = ?
                    ORDER BY created_at
                """, conn, params=(idea_id,))
                
                for _, event in events.iterrows():
                    print(f"{event['time']} | {event['signal_type']:30} | Trust: {event['trust_score']} | Stable: {event['is_stable']}")
    
    def show_component_activity(self):
        """Show which components are most active"""
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    source_node_id,
                    signal_type,
                    COUNT(*) as event_count
                FROM events
                WHERE source_node_id IS NOT NULL
                GROUP BY source_node_id, signal_type
                ORDER BY event_count DESC
                LIMIT 20
            """, conn)
            
            print("\n=== Component Activity ===")
            print(df.to_string(index=False))
    
    def show_reactor_rules_fired(self):
        """Analyze which reactor rules are firing"""
        with self.get_connection() as conn:
            # Look for trigger patterns
            df = pd.read_sql_query("""
                WITH event_sequences AS (
                    SELECT 
                        e1.signal_type as trigger_signal,
                        e2.signal_type as result_signal,
                        e2.source_node_id as triggered_component,
                        COUNT(*) as times_fired
                    FROM events e1
                    JOIN events e2 ON 
                        e2.created_at > e1.created_at AND 
                        e2.created_at < e1.created_at + 0.1  -- Within 100ms
                    WHERE e1.signal_type != e2.signal_type
                    GROUP BY trigger_signal, result_signal, triggered_component
                )
                SELECT * FROM event_sequences
                WHERE times_fired > 1
                ORDER BY times_fired DESC
            """, conn)
            
            print("\n=== Reactor Rule Patterns ===")
            print("(Signals that trigger other signals)")
            print(df.to_string(index=False))
    
    def show_quantifier_activity(self):
        """Check quantifier agent activity"""
        with self.get_connection() as conn:
            # Check for quantifier triggers
            df = pd.read_sql_query("""
                SELECT 
                    datetime(created_at, 'unixepoch') as time,
                    signal_type,
                    json_extract(payload, '$.idea_text') as idea_preview,
                    trust_score
                FROM events
                WHERE source_node_id = 'quantifier_agent_primary'
                   OR (signal_type = 'TrustAssessmentSignal' AND trust_score > 6.0)
                ORDER BY created_at DESC
                LIMIT 10
            """, conn)
            
            print("\n=== Quantifier Agent Activity ===")
            if df.empty:
                print("No quantifier activity found")
                
                # Check why
                high_trust = pd.read_sql_query("""
                    SELECT COUNT(*) as count
                    FROM events
                    WHERE signal_type = 'TrustAssessmentSignal'
                    AND trust_score > 6.0
                    AND json_extract(payload, '$.is_stable') = 1
                """, conn)
                
                print(f"Ideas meeting quantifier criteria: {high_trust['count'].iloc[0]}")
            else:
                print(df.to_string(index=False))
    
    def show_proto_activity(self):
        """Show Proto execution activity"""
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    datetime(created_at, 'unixepoch') as time,
                    signal_type,
                    json_extract(payload, '$.dialect') as dialect,
                    json_extract(payload, '$.proto_block_id') as proto_id,
                    json_extract(payload, '$.success') as success,
                    json_extract(payload, '$.execution_time_sec') as exec_time
                FROM events
                WHERE signal_type IN ('ProtoTaskSignal', 'ProtoResultSignal', 'ProtoErrorSignal')
                ORDER BY created_at DESC
            """, conn)
            
            print("\n=== Proto Execution Activity ===")
            if df.empty:
                print("No Proto executions found")
            else:
                print(df.to_string(index=False))

def main():
    viewer = EnhancedDBViewer()
    
    # Original summary
    viewer.get_event_summary()
    
    # New analyses
    viewer.show_trust_distribution()
    viewer.show_idea_journey()
    viewer.show_component_activity()
    viewer.show_reactor_rules_fired()
    viewer.show_quantifier_activity()
    viewer.show_proto_activity()

if __name__ == "__main__":
    main()