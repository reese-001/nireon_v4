"""
Quick fix for event_aggregations table schema
"""

import sqlite3
import sys

def fix_event_aggregations_schema(db_path: str = "runtime/nireon_ideas.db"):
    """Add missing column to event_aggregations table"""
    
    print(f"Fixing event_aggregations table in {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(event_aggregations)")
        columns = {row[1] for row in cursor.fetchall()}
        
        if 'high_trust_stable_count' not in columns:
            print("Adding high_trust_stable_count column...")
            cursor.execute('''
                ALTER TABLE event_aggregations 
                ADD COLUMN high_trust_stable_count INTEGER DEFAULT 0
            ''')
            conn.commit()
            print("✓ Column added successfully")
        else:
            print("✓ Column already exists")
            
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "runtime/nireon_ideas.db"
    fix_event_aggregations_schema(db_path)