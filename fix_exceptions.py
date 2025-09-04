#!/usr/bin/env python3
"""
Fix all bare except clauses and exception swallowing in the codebase
Ensures no silent failures and proper error handling
"""

import os
import re
import shutil
from pathlib import Path

# Files to fix
FILES_TO_FIX = [
    "core/orchestration/llm_orchestrator.py",
    "core/character_engine/character_simulation_engine_v2.py",
    "core/character_engine/multi_character_simulation.py", 
    "core/character_engine/complex_group_dynamics.py",
    "core/story_engine/iterative_story_system.py"
]

# Patterns to fix
FIXES = [
    {
        "pattern": r"(\s+)except:\n(\s+)return False",
        "replacement": r"\1except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:\n\2logger.warning(f'Connection error: {e}')\n\2return False\n\1except Exception as e:\n\2logger.error(f'Unexpected error: {e}')\n\2return False"
    },
    {
        "pattern": r"(\s+)except:\n(\s+)pass",
        "replacement": r"\1except json.JSONDecodeError as e:\n\2logger.warning(f'JSON parsing error: {e}')\n\2pass\n\1except Exception as e:\n\2logger.error(f'Unexpected error: {e}')\n\2raise"
    },
    {
        "pattern": r"(\s+)except:\n(\s+)return None",
        "replacement": r"\1except (KeyError, ValueError, json.JSONDecodeError) as e:\n\2logger.warning(f'Data extraction error: {e}')\n\2return None\n\1except Exception as e:\n\2logger.error(f'Unexpected error: {e}')\n\2raise"
    },
    {
        "pattern": r"(\s+)except:\n(\s+)return \{\}",
        "replacement": r"\1except (KeyError, ValueError) as e:\n\2logger.warning(f'Data processing error: {e}')\n\2return {}\n\1except Exception as e:\n\2logger.error(f'Unexpected error: {e}')\n\2raise"
    }
]

def add_imports(content):
    """Add necessary imports if not present"""
    imports_needed = []
    
    if "import logging" not in content:
        imports_needed.append("import logging")
    if "import aiohttp" not in content and "aiohttp" in content:
        imports_needed.append("import aiohttp")
    if "import asyncio" not in content and "asyncio" in content:
        imports_needed.append("import asyncio")
    if "import json" not in content and "json" in content:
        imports_needed.append("import json")
    
    if imports_needed:
        # Find the right place to insert imports
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
            elif line and not line.startswith('#') and not line.startswith('"""'):
                break
        
        for imp in imports_needed:
            lines.insert(insert_pos, imp)
            insert_pos += 1
        
        content = '\n'.join(lines)
    
    # Add logger if not present
    if "logger = logging.getLogger" not in content:
        # Add after imports
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
            elif line and not line.startswith('#') and not line.startswith('"""'):
                if insert_pos > 0:
                    lines.insert(insert_pos, "")
                    lines.insert(insert_pos + 1, "logger = logging.getLogger(__name__)")
                    lines.insert(insert_pos + 2, "")
                break
        content = '\n'.join(lines)
    
    return content

def fix_file(filepath):
    """Fix exception handling in a single file"""
    print(f"Fixing {filepath}...")
    
    # Create backup
    backup_path = filepath + ".backup"
    shutil.copy2(filepath, backup_path)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        for fix in FIXES:
            content = re.sub(fix["pattern"], fix["replacement"], content)
        
        # Add necessary imports
        content = add_imports(content)
        
        # Write fixed content
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Fixed exception handling in {filepath}")
            return True
        else:
            print(f"  ℹ️  No changes needed in {filepath}")
            os.remove(backup_path)
            return False
            
    except Exception as e:
        print(f"  ❌ Error fixing {filepath}: {e}")
        # Restore backup
        shutil.move(backup_path, filepath)
        return False

def main():
    """Fix all files"""
    print("🔧 Fixing Exception Handling Issues")
    print("=" * 60)
    
    fixed_count = 0
    error_count = 0
    
    for filepath in FILES_TO_FIX:
        if os.path.exists(filepath):
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"  ⚠️  File not found: {filepath}")
            error_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Fixed {fixed_count} files")
    if error_count > 0:
        print(f"⚠️  {error_count} files had errors or were not found")
    
    print("\n📝 Recommendations:")
    print("1. Use llm_orchestrator_strict.py as the primary orchestrator")
    print("2. Review the fixed files to ensure proper error handling")
    print("3. Run the mock data detector again to verify fixes")
    print("4. Test with actual LLM providers to ensure no silent failures")

if __name__ == "__main__":
    main()