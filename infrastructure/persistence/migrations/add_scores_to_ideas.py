import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def migrate_ideas_table_add_scores(db_path: str = 'runtime/nireon_ideas.db'):
    """Add trust_score and novelty_score columns to ideas table"""
    
    print(f"Migrating ideas table in {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check existing columns
        cursor.execute("PRAGMA table_info(ideas)")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Add trust_score column if not exists
        if 'trust_score' not in columns:
            print("Adding trust_score column...")
            cursor.execute("""
                ALTER TABLE ideas 
                ADD COLUMN trust_score REAL
            """)
            
        # Add novelty_score column if not exists
        if 'novelty_score' not in columns:
            print("Adding novelty_score column...")
            cursor.execute("""
                ALTER TABLE ideas 
                ADD COLUMN novelty_score REAL
            """)
            
        # Add is_stable column for completeness
        if 'is_stable' not in columns:
            print("Adding is_stable column...")
            cursor.execute("""
                ALTER TABLE ideas 
                ADD COLUMN is_stable BOOLEAN
            """)
            
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ideas_trust_score 
            ON ideas(trust_score DESC) 
            WHERE trust_score IS NOT NULL
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ideas_novelty_score 
            ON ideas(novelty_score DESC) 
            WHERE novelty_score IS NOT NULL
        """)
        
        # Populate scores from existing trust assessments
        print("Populating scores from existing assessments...")
        cursor.execute("""
            UPDATE ideas
            SET trust_score = (
                SELECT e.trust_score
                FROM events e
                WHERE e.signal_type = 'TrustAssessmentSignal'
                  AND (e.target_id = ideas.idea_id OR e.related_idea_id = ideas.idea_id)
                  AND e.trust_score IS NOT NULL
                ORDER BY e.created_at DESC
                LIMIT 1
            ),
            novelty_score = (
                SELECT e.novelty_score
                FROM events e
                WHERE e.signal_type = 'TrustAssessmentSignal'
                  AND (e.target_id = ideas.idea_id OR e.related_idea_id = ideas.idea_id)
                  AND e.novelty_score IS NOT NULL
                ORDER BY e.created_at DESC
                LIMIT 1
            ),
            is_stable = (
                SELECT e.is_stable
                FROM events e
                WHERE e.signal_type = 'TrustAssessmentSignal'
                  AND (e.target_id = ideas.idea_id OR e.related_idea_id = ideas.idea_id)
                  AND e.is_stable IS NOT NULL
                ORDER BY e.created_at DESC
                LIMIT 1
            )
            WHERE EXISTS (
                SELECT 1 FROM events e
                WHERE e.signal_type = 'TrustAssessmentSignal'
                  AND (e.target_id = ideas.idea_id OR e.related_idea_id = ideas.idea_id)
            )
        """)
        
        updated_count = cursor.rowcount
        conn.commit()
        
        print(f"✓ Migration completed successfully")
        print(f"✓ Updated {updated_count} ideas with scores")
        
        # Show statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_ideas,
                COUNT(trust_score) as has_trust,
                AVG(trust_score) as avg_trust,
                MAX(trust_score) as max_trust,
                COUNT(CASE WHEN trust_score > 6.0 THEN 1 END) as high_trust_count
            FROM ideas
        """)
        stats = cursor.fetchone()
        print(f"\nIdea Statistics:")
        print(f"  Total ideas: {stats[0]}")
        print(f"  Ideas with scores: {stats[1]}")
        print(f"  Average trust: {stats[2]:.2f}" if stats[2] else "  Average trust: N/A")
        print(f"  Max trust: {stats[3]:.2f}" if stats[3] else "  Max trust: N/A")
        print(f"  High trust ideas: {stats[4]}")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()