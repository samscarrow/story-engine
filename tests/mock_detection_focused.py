#!/usr/bin/env python3
"""
Focused Mock Data Detection for Autonomous Agent Pipeline

Direct analysis of the most critical mock data patterns without full audit overhead.
"""

import re
from pathlib import Path
from typing import Dict, Any

def find_mock_data_issues(project_root: str = "..") -> Dict[str, Any]:
    """
    Direct scan for mock data patterns in the autonomous agent pipeline
    """
    
    project_path = Path(project_root)
    issues = {
        'critical': [],
        'high': [],
        'medium': [],
        'files_scanned': 0
    }
    
    # Critical patterns that indicate mock data in production
    critical_patterns = [
        r'Mock\s*\(',  # Mock() usage
        r'AsyncMock\s*\(',  # AsyncMock() usage
        r'mock_orchestrator',  # Mock orchestrator usage
        r'return Mock',  # Returning mock objects
        r'self\.mock_',  # Mock instance variables
        r'# Note:.*mock.*implementation',  # Comments about mock implementations
    ]
    
    # High severity patterns
    high_patterns = [
        r'workflow_results\s*=\s*\{',  # Hardcoded workflow results
        r'return\s*\{[^}]*"stages_completed"',  # Hardcoded simulation results
        r'success_rate.*0\.\d+',  # Hardcoded success rates
        r'characters_created.*\[.*\]',  # Hardcoded character lists
        r'scenes_designed.*\[.*\]',  # Hardcoded scene lists
    ]
    
    # Medium patterns
    medium_patterns = [
        r'# Uncomment.*run.*real',  # Comments about running with real LLMs
        r'For now.*simulated',  # Simulated implementations
        r'mock.*response',  # Mock response references
    ]
    
    # Scan key files
    key_files = [
        "examples/autonomous_agent_usage_example.py",
        "examples/autonomous_persona_demonstration.py", 
        "core/orchestration/autonomous_persona_agents.py",
        "core/orchestration/recursive_simulation_engine.py",
        "core/orchestration/unified_llm_orchestrator.py",
        "tests/test_breaking_points_simplified.py"
    ]
    
    for file_path in key_files:
        full_path = project_path / file_path
        
        if not full_path.exists():
            continue
            
        issues['files_scanned'] += 1
        
        try:
            content = full_path.read_text(encoding='utf-8')
            
            # Check critical patterns
            for pattern in critical_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues['critical'].append({
                        'file': str(file_path),
                        'line': line_num,
                        'pattern': pattern,
                        'match': match.group(0),
                        'context': get_line_context(content, line_num)
                    })
            
            # Check high patterns
            for pattern in high_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues['high'].append({
                        'file': str(file_path),
                        'line': line_num,
                        'pattern': pattern,
                        'match': match.group(0),
                        'context': get_line_context(content, line_num)
                    })
            
            # Check medium patterns
            for pattern in medium_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues['medium'].append({
                        'file': str(file_path),
                        'line': line_num,
                        'pattern': pattern,
                        'match': match.group(0),
                        'context': get_line_context(content, line_num)
                    })
                    
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
    
    return issues

def get_line_context(content: str, line_num: int, context: int = 1) -> str:
    """Get context lines around a specific line number"""
    lines = content.split('\n')
    start = max(0, line_num - context - 1)
    end = min(len(lines), line_num + context)
    
    context_lines = []
    for i in range(start, end):
        marker = ">>> " if i == line_num - 1 else "    "
        context_lines.append(f"{marker}{i+1}: {lines[i]}")
    
    return '\n'.join(context_lines)

def analyze_specific_mock_implementations():
    """
    Analyze specific files for mock implementations
    """
    
    print("🔍 ANALYZING SPECIFIC FILES FOR MOCK DATA...")
    
    # Check the usage example specifically
    usage_example = Path("../examples/autonomous_agent_usage_example.py")
    if usage_example.exists():
        content = usage_example.read_text()
        
        print(f"\n📄 {usage_example.name}:")
        
        # Look for mock orchestrator setup
        if 'providers = {' in content and 'mock_provider' in content:
            print("   🔴 CRITICAL: Uses mock provider configuration")
        
        # Look for hardcoded results
        if 'workflow_results = {' in content:
            print("   🔴 CRITICAL: Contains hardcoded workflow results")
            
            # Extract the results section
            start = content.find('workflow_results = {')
            end = content.find('}', start + 1000)  # Look within reasonable range
            if end != -1:
                results_section = content[start:end + 1]
                if "'stages_completed'" in results_section:
                    print("   🔴 CRITICAL: Hardcoded stages_completed list")
                if "'characters_created'" in results_section:
                    print("   🔴 CRITICAL: Hardcoded characters_created list")
        
        # Look for actual LLM calls
        if 'await' in content and 'llm' in content.lower():
            print("   ✅ Contains async LLM-like calls")
        else:
            print("   🟡 WARNING: Limited async LLM interaction detected")
    
    # Check demonstration file
    demo_file = Path("../examples/autonomous_persona_demonstration.py")
    if demo_file.exists():
        content = demo_file.read_text()
        
        print(f"\n📄 {demo_file.name}:")
        
        if 'Mock' in content:
            print("   🔴 CRITICAL: Contains Mock objects")
        
        if 'asyncio.run(demonstrate_autonomous_agents())' in content:
            # Check if it's commented out
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'asyncio.run(demonstrate_autonomous_agents())' in line:
                    if line.strip().startswith('#'):
                        print("   🔴 CRITICAL: Main demonstration is commented out")
                    else:
                        print("   ✅ Main demonstration can be executed")
    
    # Check the actual agents
    agents_file = Path("../core/orchestration/autonomous_persona_agents.py")
    if agents_file.exists():
        content = agents_file.read_text()
        
        print(f"\n📄 {agents_file.name}:")
        
        # Look for actual LLM generation calls
        if 'await self.orchestrator.generate_with_persona' in content:
            print("   ✅ Makes actual LLM orchestrator calls")
        else:
            print("   🔴 CRITICAL: No actual LLM orchestrator calls found")
        
        # Look for template generation
        if 'def generate_base_template' in content:
            # Check if it returns hardcoded content
            method_start = content.find('def generate_base_template')
            method_end = content.find('\n    def ', method_start + 1)
            if method_end == -1:
                method_end = len(content)
            
            method_content = content[method_start:method_end]
            
            if 'return """' in method_content or "return '''" in method_content:
                print("   🔴 CRITICAL: generate_base_template returns static content")
            elif 'await' in method_content:
                print("   ✅ generate_base_template uses async calls")
            else:
                print("   🟡 WARNING: generate_base_template implementation unclear")

def main():
    print("🚨 FOCUSED MOCK DATA DETECTION")
    print("="*50)
    
    # Run focused detection
    issues = find_mock_data_issues()
    
    print("\n📊 SCAN RESULTS:")
    print(f"   Files Scanned: {issues['files_scanned']}")
    print(f"   Critical Issues: {len(issues['critical'])}")
    print(f"   High Issues: {len(issues['high'])}")
    print(f"   Medium Issues: {len(issues['medium'])}")
    
    if issues['critical']:
        print("\n🔴 CRITICAL MOCK DATA ISSUES:")
        for issue in issues['critical']:
            print(f"   📍 {issue['file']}:{issue['line']}")
            print(f"      Pattern: {issue['pattern']}")
            print(f"      Match: {issue['match']}")
            print(f"      Context:\n{issue['context']}")
            print()
    
    if issues['high']:
        print("\n🟡 HIGH PRIORITY ISSUES:")
        for issue in issues['high'][:5]:  # Show first 5
            print(f"   📍 {issue['file']}:{issue['line']}")
            print(f"      Match: {issue['match']}")
            print()
    
    if not issues['critical'] and not issues['high']:
        print("\n✅ No critical mock data patterns detected!")
    
    # Run specific analysis
    analyze_specific_mock_implementations()
    
    # Overall assessment
    total_issues = len(issues['critical']) + len(issues['high'])
    
    print("\n📈 MOCK DATA ASSESSMENT:")
    if issues['critical']:
        print("   🔴 CRITICAL: Production code contains mock implementations")
        print("   📝 ACTION REQUIRED: Remove mock objects and implement real LLM integration")
    elif len(issues['high']) > 5:
        print("   🟡 WARNING: Multiple hardcoded responses detected")  
        print("   📝 RECOMMENDATION: Replace hardcoded data with dynamic generation")
    elif total_issues > 0:
        print("   🟠 MODERATE: Some suspicious patterns detected")
        print("   📝 SUGGESTION: Review flagged implementations")
    else:
        print("   ✅ CLEAN: No significant mock data issues detected")
    
    return issues

if __name__ == "__main__":
    main()