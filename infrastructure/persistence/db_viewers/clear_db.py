#!/usr/bin/env python3
"""
Clear Nireon database tables for a fresh start
"""

import sqlite3
import sys
from datetime import datetime

def clear_nireon_database(db_path: str = "runtime/nireon_ideas.db", keep_schema: bool = True):
    """Clear all data from Nireon database tables"""
    
    print(f"Clearing database: {db_path}")
    print(f"Keep schema: {keep_schema}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"\nFound {len(tables)} tables: {', '.join(tables)}")
    
    if keep_schema:
        # Clear data but keep table structure
        print("\nClearing data from tables...")
        
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table}")
                count = cursor.rowcount
                print(f"  ✓ Cleared {count} rows from {table}")
            except Exception as e:
                print(f"  ✗ Error clearing {table}: {e}")
        
        # Reset autoincrement counters
        cursor.execute("DELETE FROM sqlite_sequence")
        print("  ✓ Reset autoincrement counters")
        
        # Commit the transaction before VACUUM
        conn.commit()
        
        # Vacuum to reclaim space - must be done outside of a transaction
        conn.isolation_level = None  # This enables autocommit mode
        try:
            cursor.execute("VACUUM")
            print("  ✓ Database vacuumed")
        except Exception as e:
            print(f"  ✗ Error during VACUUM: {e}")
        finally:
            # Restore normal transaction mode
            conn.isolation_level = ""
        
    else:
        # Drop all tables
        print("\nDropping all tables...")
        
        for table in tables:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"  ✓ Dropped {table}")
            except Exception as e:
                print(f"  ✗ Error dropping {table}: {e}")
        
        conn.commit()
    
    # Show database statistics
    cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
    size_bytes = cursor.fetchone()[0]
    size_mb = size_bytes / (1024 * 1024)
    print(f"\nDatabase size: {size_mb:.2f} MB")
    
    conn.close()
    print("\nDatabase cleared successfully!")
    
    # Create a backup marker
    backup_marker = f"cleared_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(f"runtime/{backup_marker}", "w") as f:
            f.write(f"Database cleared at {datetime.now()}\n")
            f.write(f"Tables cleared: {', '.join(tables)}\n")
        print(f"Created backup marker: runtime/{backup_marker}")
    except:
        pass


def verify_idea_text_in_signals(db_path: str = "runtime/nireon_ideas.db"):
    """Check if recent TrustAssessmentSignals have idea_text"""
    
    print("\nVerifying idea_text in TrustAssessmentSignals...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN json_extract(payload, '$.idea_text') IS NOT NULL THEN 1 ELSE 0 END) as has_text,
            MAX(timestamp) as latest
        FROM events
        WHERE signal_type = 'TrustAssessmentSignal'
    """)
    
    result = cursor.fetchone()
    if result[0] > 0:
        print(f"  Total signals: {result[0]}")
        print(f"  With idea_text: {result[1]} ({result[1]/result[0]*100:.1f}%)")
        print(f"  Latest: {result[2]}")
    else:
        print("  No TrustAssessmentSignals found")
    
    conn.close()


if __name__ == "__main__":
    import os
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "runtime/nireon_ideas.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    print("This will DELETE ALL DATA from the Nireon database!")
    print(f"Database: {db_path}")
    print("\nOptions:")
    print("1. Clear all data (keep table structure)")
    print("2. Drop all tables (complete reset)")
    print("3. Cancel")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        clear_nireon_database(db_path, keep_schema=True)
        verify_idea_text_in_signals(db_path)
    elif choice == "2":
        clear_nireon_database(db_path, keep_schema=False)
        print("\nNote: You'll need to restart Nireon to recreate the tables.")
    else:
        print("Cancelled.")