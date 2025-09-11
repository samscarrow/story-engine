#!/usr/bin/env python3
"""
APEX Integration Helper - Creates database views and procedures for APEX integration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    import oracledb
except ImportError:
    print("oracledb not available - run: pip install oracledb")
    sys.exit(1)


def create_apex_integration():
    """Create database views and procedures to support APEX applications."""

    load_dotenv(".env.oracle")

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    dsn = os.getenv("DB_DSN")
    wallet_location = os.getenv("DB_WALLET_LOCATION")

    wallet_path = str(Path(wallet_location).resolve())
    os.environ["TNS_ADMIN"] = wallet_path

    try:
        print("Connecting to Oracle database...")
        conn = oracledb.connect(
            user=user, password=password, dsn=dsn, config_dir=wallet_path
        )

        cursor = conn.cursor()

        print("Creating APEX-friendly database objects...")

        # 1. Dashboard Views
        dashboard_views = [
            # Workflow Summary View
            """
            CREATE OR REPLACE VIEW apex_workflow_summary AS
            SELECT 
                workflow_name,
                COUNT(*) as total_executions,
                MAX(timestamp) as last_execution,
                MIN(timestamp) as first_execution,
                ROUND(AVG(
                    CASE 
                        WHEN JSON_EXISTS(metadata, '$.execution_time')
                        THEN JSON_VALUE(metadata, '$.execution_time' RETURNING NUMBER)
                        ELSE NULL 
                    END
                ), 2) as avg_execution_time_sec,
                COUNT(
                    CASE 
                        WHEN JSON_VALUE(metadata, '$.status') = 'success' 
                        THEN 1 
                    END
                ) as successful_runs,
                COUNT(
                    CASE 
                        WHEN JSON_VALUE(metadata, '$.status') = 'error' 
                        THEN 1 
                    END
                ) as failed_runs
            FROM workflow_outputs 
            GROUP BY workflow_name
            ORDER BY last_execution DESC
            """,
            # Recent Activity View
            """
            CREATE OR REPLACE VIEW apex_recent_activity AS
            SELECT 
                id,
                workflow_name,
                timestamp,
                JSON_VALUE(metadata, '$.status') as status,
                JSON_VALUE(metadata, '$.model_used') as model_used,
                JSON_VALUE(metadata, '$.execution_time' RETURNING NUMBER) as execution_time,
                CASE 
                    WHEN DBMS_LOB.GETLENGTH(output_data) > 1000 
                    THEN SUBSTR(output_data, 1, 997) || '...'
                    ELSE output_data 
                END as output_preview
            FROM workflow_outputs 
            WHERE timestamp >= SYSDATE - 7  -- Last 7 days
            ORDER BY timestamp DESC
            """,
            # System Health View
            """
            CREATE OR REPLACE VIEW apex_system_health AS
            SELECT 
                'Workflows' as component,
                COUNT(*) as total_records,
                MAX(timestamp) as last_activity,
                ROUND(
                    COUNT(CASE WHEN timestamp >= SYSDATE - 1 THEN 1 END) / 
                    NULLIF(COUNT(*), 0) * 100, 
                    1
                ) as activity_last_24h_pct
            FROM workflow_outputs
            UNION ALL
            SELECT 
                'Characters' as component,
                COUNT(*) as total_records,
                MAX(updated_at) as last_activity,
                ROUND(
                    COUNT(CASE WHEN updated_at >= SYSDATE - 1 THEN 1 END) / 
                    NULLIF(COUNT(*), 0) * 100, 
                    1
                ) as activity_last_24h_pct
            FROM characters
            UNION ALL
            SELECT 
                'World States' as component,
                COUNT(*) as total_records,
                MAX(created_at) as last_activity,
                ROUND(
                    COUNT(CASE WHEN created_at >= SYSDATE - 1 THEN 1 END) / 
                    NULLIF(COUNT(*), 0) * 100, 
                    1
                ) as activity_last_24h_pct
            FROM world_states
            UNION ALL
            SELECT 
                'Scenes' as component,
                COUNT(*) as total_records,
                MAX(created_at) as last_activity,
                ROUND(
                    COUNT(CASE WHEN created_at >= SYSDATE - 1 THEN 1 END) / 
                    NULLIF(COUNT(*), 0) * 100, 
                    1
                ) as activity_last_24h_pct
            FROM scenes
            """,
        ]

        # 2. Character Management Views
        character_views = [
            # Character Details View
            """
            CREATE OR REPLACE VIEW apex_character_details AS
            SELECT 
                c.id,
                c.name,
                c.created_at,
                c.updated_at,
                JSON_VALUE(c.persona_data, '$.personality.traits[0]') as primary_trait,
                JSON_VALUE(c.persona_data, '$.background.occupation') as occupation,
                JSON_VALUE(c.persona_data, '$.background.origin') as origin,
                (SELECT COUNT(*) FROM world_states ws WHERE ws.character_id = c.id) as world_states_count,
                (SELECT COUNT(*) FROM scenes s WHERE JSON_EXISTS(s.character_ids, '$[*] ? (@ == "' || c.id || '")')) as scenes_count
            FROM characters c
            ORDER BY c.updated_at DESC
            """,
            # Character Relationships View (if we had relationships table)
            """
            CREATE OR REPLACE VIEW apex_character_network AS
            SELECT 
                c1.id as character_id,
                c1.name as character_name,
                c2.id as related_character_id, 
                c2.name as related_character_name,
                'appears_together' as relationship_type,
                COUNT(*) as interaction_count
            FROM characters c1
            JOIN scenes s ON JSON_EXISTS(s.character_ids, '$[*] ? (@ == "' || c1.id || '")')
            JOIN characters c2 ON c2.id != c1.id 
                AND JSON_EXISTS(s.character_ids, '$[*] ? (@ == "' || c2.id || '")')
            GROUP BY c1.id, c1.name, c2.id, c2.name
            HAVING COUNT(*) > 0
            ORDER BY interaction_count DESC
            """,
        ]

        # 3. Story Analytics Views
        analytics_views = [
            # Quality Metrics View
            """
            CREATE OR REPLACE VIEW apex_quality_metrics AS
            SELECT 
                workflow_name,
                TO_CHAR(timestamp, 'YYYY-MM') as month_year,
                AVG(JSON_VALUE(metadata, '$.quality_score' RETURNING NUMBER)) as avg_quality,
                COUNT(*) as story_count,
                AVG(JSON_VALUE(metadata, '$.execution_time' RETURNING NUMBER)) as avg_generation_time,
                JSON_VALUE(metadata, '$.model_used') as model_used
            FROM workflow_outputs 
            WHERE JSON_EXISTS(metadata, '$.quality_score')
            GROUP BY workflow_name, TO_CHAR(timestamp, 'YYYY-MM'), JSON_VALUE(metadata, '$.model_used')
            ORDER BY month_year DESC, avg_quality DESC
            """,
            # Performance Trends View
            """
            CREATE OR REPLACE VIEW apex_performance_trends AS
            SELECT 
                TO_CHAR(timestamp, 'YYYY-MM-DD') as date_key,
                COUNT(*) as daily_stories,
                AVG(JSON_VALUE(metadata, '$.execution_time' RETURNING NUMBER)) as avg_time,
                MIN(JSON_VALUE(metadata, '$.execution_time' RETURNING NUMBER)) as min_time,
                MAX(JSON_VALUE(metadata, '$.execution_time' RETURNING NUMBER)) as max_time,
                COUNT(CASE WHEN JSON_VALUE(metadata, '$.status') = 'success' THEN 1 END) as success_count,
                COUNT(CASE WHEN JSON_VALUE(metadata, '$.status') = 'error' THEN 1 END) as error_count
            FROM workflow_outputs 
            WHERE timestamp >= SYSDATE - 30  -- Last 30 days
            GROUP BY TO_CHAR(timestamp, 'YYYY-MM-DD')
            ORDER BY date_key DESC
            """,
        ]

        # Create all views
        all_views = dashboard_views + character_views + analytics_views

        for i, view_sql in enumerate(all_views, 1):
            try:
                cursor.execute(view_sql)
                # Extract view name from SQL
                view_name = (
                    view_sql.split("VIEW ")[1].split(" ")[0].replace("apex_", "")
                )
                print(f"   ✓ Created view: apex_{view_name}")
            except oracledb.Error as e:
                if "ORA-00955" in str(e):  # Already exists
                    view_name = (
                        view_sql.split("VIEW ")[1].split(" ")[0].replace("apex_", "")
                    )
                    print(f"   - View apex_{view_name} already exists")
                else:
                    print(f"   ✗ Error creating view: {e}")

        # 4. Utility Procedures for APEX
        procedures = [
            # Procedure to trigger story generation
            """
            CREATE OR REPLACE PROCEDURE apex_trigger_story_generation(
                p_workflow_name IN VARCHAR2,
                p_parameters IN CLOB,
                p_result OUT VARCHAR2
            )
            AS
            BEGIN
                -- This would integrate with your Python story engine
                -- For now, just log the request
                INSERT INTO workflow_outputs (workflow_name, output_data, metadata)
                VALUES (
                    p_workflow_name || '_triggered',
                    p_parameters,
                    JSON_OBJECT(
                        'status' VALUE 'triggered_from_apex',
                        'timestamp' VALUE SYSTIMESTAMP,
                        'trigger_source' VALUE 'apex_ui'
                    )
                );
                COMMIT;
                p_result := 'Story generation triggered successfully';
            EXCEPTION
                WHEN OTHERS THEN
                    p_result := 'Error: ' || SQLERRM;
            END;
            """,
            # Procedure to clean up old data
            """
            CREATE OR REPLACE PROCEDURE apex_cleanup_old_data(
                p_days_to_keep IN NUMBER DEFAULT 90,
                p_result OUT VARCHAR2
            )
            AS
                v_deleted_count NUMBER;
            BEGIN
                DELETE FROM workflow_outputs 
                WHERE timestamp < SYSDATE - p_days_to_keep
                AND workflow_name LIKE '%test%';
                
                v_deleted_count := SQL%ROWCOUNT;
                COMMIT;
                
                p_result := 'Deleted ' || v_deleted_count || ' old test records';
            EXCEPTION
                WHEN OTHERS THEN
                    p_result := 'Error: ' || SQLERRM;
                    ROLLBACK;
            END;
            """,
        ]

        for proc_sql in procedures:
            try:
                cursor.execute(proc_sql)
                proc_name = proc_sql.split("PROCEDURE ")[1].split("(")[0].strip()
                print(f"   ✓ Created procedure: {proc_name}")
            except oracledb.Error as e:
                if "ORA-00955" in str(e):
                    proc_name = proc_sql.split("PROCEDURE ")[1].split("(")[0].strip()
                    print(f"   - Procedure {proc_name} already exists")
                else:
                    print(f"   ✗ Error creating procedure: {e}")

        # 5. Create sample data for testing
        print("\nCreating sample data for APEX testing...")

        sample_data = [
            # Sample workflow outputs
            """
            INSERT INTO workflow_outputs (workflow_name, output_data, metadata)
            SELECT * FROM (
                SELECT 'character_generation' as workflow_name, 
                       '{"character": {"name": "Aragorn", "class": "Ranger"}}' as output_data,
                       JSON_OBJECT('status' VALUE 'success', 'execution_time' VALUE 2.3, 'model_used' VALUE 'qwen3-8b', 'quality_score' VALUE 8.5) as metadata
                FROM dual
                UNION ALL
                SELECT 'scene_generation',
                       '{"scene": "A dark forest clearing where heroes gather"}',
                       JSON_OBJECT('status' VALUE 'success', 'execution_time' VALUE 4.1, 'model_used' VALUE 'story-unhinged', 'quality_score' VALUE 9.2)
                FROM dual
                UNION ALL
                SELECT 'plot_structure',
                       '{"structure": "three_act", "acts": ["Setup", "Confrontation", "Resolution"]}',
                       JSON_OBJECT('status' VALUE 'success', 'execution_time' VALUE 1.8, 'model_used' VALUE 'qwen3-8b', 'quality_score' VALUE 7.8)
                FROM dual
            ) WHERE NOT EXISTS (
                SELECT 1 FROM workflow_outputs WHERE workflow_name = 'character_generation'
            )
            """,
            # Sample characters
            """
            INSERT INTO characters (id, name, persona_data)
            SELECT * FROM (
                SELECT 'char_aragorn' as id,
                       'Aragorn' as name,
                       JSON_OBJECT(
                           'personality' VALUE JSON_OBJECT('traits' VALUE JSON_ARRAY('brave', 'noble', 'reluctant_leader')),
                           'background' VALUE JSON_OBJECT('occupation' VALUE 'Ranger', 'origin' VALUE 'Gondor'),
                           'appearance' VALUE JSON_OBJECT('height' VALUE 'tall', 'build' VALUE 'lean')
                       ) as persona_data
                FROM dual
                UNION ALL
                SELECT 'char_gandalf',
                       'Gandalf',
                       JSON_OBJECT(
                           'personality' VALUE JSON_OBJECT('traits' VALUE JSON_ARRAY('wise', 'patient', 'mysterious')),
                           'background' VALUE JSON_OBJECT('occupation' VALUE 'Wizard', 'origin' VALUE 'Valinor'),
                           'appearance' VALUE JSON_OBJECT('height' VALUE 'tall', 'build' VALUE 'thin')
                       )
                FROM dual
            ) WHERE NOT EXISTS (
                SELECT 1 FROM characters WHERE id = 'char_aragorn'
            )
            """,
        ]

        for sample_sql in sample_data:
            try:
                cursor.execute(sample_sql)
                print("   ✓ Created sample data")
            except oracledb.Error as e:
                print(f"   - Sample data might already exist: {e}")

        conn.commit()
        cursor.close()
        conn.close()

        print("\n=== APEX INTEGRATION SETUP COMPLETE ===")
        print("\nDatabase objects created:")
        print("📊 Dashboard Views:")
        print("   - apex_workflow_summary")
        print("   - apex_recent_activity")
        print("   - apex_system_health")
        print("\n👥 Character Views:")
        print("   - apex_character_details")
        print("   - apex_character_network")
        print("\n📈 Analytics Views:")
        print("   - apex_quality_metrics")
        print("   - apex_performance_trends")
        print("\n⚙️ Utility Procedures:")
        print("   - apex_trigger_story_generation")
        print("   - apex_cleanup_old_data")

        print("\nNext Steps:")
        print("1. Access APEX at your provided URL")
        print("2. Create new application workspace")
        print("3. Use these views as data sources for APEX pages")
        print("4. Build interactive dashboards and forms")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("Oracle APEX Integration Setup")
    print("=" * 40)
    print()
    print("This will create database views and procedures for APEX:")
    print("- Dashboard and analytics views")
    print("- Character management views")
    print("- Utility procedures for story operations")
    print("- Sample data for testing")
    print()

    success = create_apex_integration()
    sys.exit(0 if success else 1)
