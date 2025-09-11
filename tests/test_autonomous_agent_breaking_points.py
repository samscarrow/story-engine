#!/usr/bin/env python3
"""
Autonomous Agent Pipeline Breaking Point Tests

Tests the autonomous persona agent system for failure modes, edge cases,
and breaking points to identify weaknesses and improve robustness.

Test Categories:
1. Template Creation Failures
2. Recursive Simulation Depth Limits
3. Agent Collaboration Deadlocks
4. Memory/Performance Bottlenecks
5. Malformed Input Handling
6. Network/LLM Provider Failures
7. Template Evolution Edge Cases
8. Concurrent Access Race Conditions
"""

import asyncio
import logging
import time
from unittest.mock import Mock, AsyncMock
from typing import Dict, List
from pathlib import Path
import tempfile
import shutil

# Import the components we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from story_engine.core.orchestration.autonomous_persona_agents import (
    CharacterSimulatorAgent,
    SceneArchitectAgent
)
from story_engine.core.orchestration.recursive_simulation_engine import (
    RecursiveSimulationEngine,
    SimulationContext,
    SimulationPriority
)
from story_engine.core.orchestration.unified_llm_orchestrator import (
    UnifiedLLMOrchestrator,
    LLMPersona
)

# Setup test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BreakingPointTester:
    """
    Comprehensive test suite for identifying breaking points in the autonomous agent system
    """
    
    def __init__(self):
        self.temp_dir = None
        self.mock_orchestrator = None
        self.test_results = {
            'template_creation_failures': [],
            'recursion_limits': [],
            'deadlock_scenarios': [],
            'memory_issues': [],
            'malformed_input': [],
            'provider_failures': [],
            'evolution_edge_cases': [],
            'race_conditions': []
        }
        
    async def setup(self):
        """Setup test environment with mocked dependencies"""
        # Create temporary directory for test templates
        self.temp_dir = tempfile.mkdtemp(prefix="agent_test_")
        
        # Create mock LLM orchestrator
        self.mock_orchestrator = Mock(spec=UnifiedLLMOrchestrator)
        self.mock_orchestrator.generate_with_persona = AsyncMock()
        
        logger.info(f"Test setup complete. Temp dir: {self.temp_dir}")
    
    async def cleanup(self):
        """Cleanup test environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        logger.info("Test cleanup complete")
    
    async def test_template_creation_breaking_points(self):
        """
        Test various template creation failure scenarios
        """
        logger.info("Testing template creation breaking points...")
        
        failures = []
        
        # Test 1: Extremely large context data
        try:
            agent = CharacterSimulatorAgent(
                self.mock_orchestrator,
                template_repository_path=f"{self.temp_dir}/char_sim"
            )
            
            # Create context with massive data
            huge_context = {
                'character_concept': {
                    'massive_backstory': 'A' * 100000,  # 100k character string
                    'complex_traits': ['trait'] * 10000,  # 10k traits
                    'detailed_relationships': {f'person_{i}': f'relationship_{i}' * 1000 for i in range(1000)}
                }
            }
            
            context = SimulationContext(
                agent_id='test_agent',
                simulation_id='breaking_test_1',
                context_data=huge_context,
                parent_simulation_id=None,
                depth=0
            )
            
            # This should handle gracefully or fail predictably
            template = await agent.generate_base_template(context)
            
            if len(template) > 50000:  # Templates shouldn't be excessively large
                failures.append({
                    'test': 'massive_context',
                    'issue': 'Template too large',
                    'template_size': len(template)
                })
                
        except Exception as e:
            failures.append({
                'test': 'massive_context',
                'issue': 'Exception during template creation',
                'error': str(e)
            })
        
        # Test 2: Malformed/corrupted context data
        try:
            malformed_contexts = [
                {'character_concept': None},
                {'character_concept': {'name': {'invalid': 'structure'}}},
                {'character_concept': {'traits': 'should_be_list_not_string'}},
                {},  # Empty context
                {'circular_ref': None}  # Will add circular reference
            ]
            
            # Add circular reference
            malformed_contexts[-1]['circular_ref'] = malformed_contexts[-1]
            
            for i, malformed_context in enumerate(malformed_contexts):
                try:
                    context = SimulationContext(
                        agent_id='test_agent',
                        simulation_id=f'malformed_test_{i}',
                        context_data=malformed_context,
                        parent_simulation_id=None,
                        depth=0
                    )
                    
                    template = await agent.generate_base_template(context)
                    
                    # Should have some basic validation
                    if not template or len(template) < 10:
                        failures.append({
                            'test': f'malformed_context_{i}',
                            'issue': 'Template too short or empty',
                            'context': str(malformed_context)[:200]
                        })
                        
                except Exception as e:
                    failures.append({
                        'test': f'malformed_context_{i}',
                        'issue': 'Unhandled exception',
                        'error': str(e),
                        'context': str(malformed_context)[:200]
                    })
        
        except Exception as e:
            failures.append({
                'test': 'malformed_contexts',
                'issue': 'Setup failure',
                'error': str(e)
            })
        
        # Test 3: Filesystem permission issues
        try:
            # Create agent with read-only template directory
            readonly_dir = f"{self.temp_dir}/readonly"
            Path(readonly_dir).mkdir()
            Path(readonly_dir).chmod(0o444)  # Read-only
            
            agent = CharacterSimulatorAgent(
                self.mock_orchestrator,
                template_repository_path=readonly_dir
            )
            
            context = SimulationContext(
                agent_id='test_agent',
                simulation_id='readonly_test',
                context_data={'character_concept': {'name': 'Test'}},
                parent_simulation_id=None,
                depth=0
            )
            
            template = await agent.generate_base_template(context)
            
            # Should handle read-only gracefully
            if not hasattr(agent, '_handle_readonly_repository'):
                failures.append({
                    'test': 'readonly_filesystem',
                    'issue': 'No handling for read-only template repository'
                })
                
        except PermissionError:
            # Expected - but should be handled gracefully
            failures.append({
                'test': 'readonly_filesystem',
                'issue': 'PermissionError not handled gracefully'
            })
        except Exception as e:
            failures.append({
                'test': 'readonly_filesystem',
                'issue': 'Unexpected exception',
                'error': str(e)
            })
        
        self.test_results['template_creation_failures'] = failures
        logger.info(f"Template creation tests complete. Found {len(failures)} issues.")
        
        return failures
    
    async def test_recursive_simulation_limits(self):
        """
        Test recursive simulation depth and complexity limits
        """
        logger.info("Testing recursive simulation breaking points...")
        
        failures = []
        
        # Mock LLM response that always spawns more simulations
        self.mock_orchestrator.generate_with_persona.return_value = Mock(
            content="This simulation needs to spawn 5 sub-simulations for detailed analysis.",
            metadata={'persona': 'CHARACTER_SIMULATOR'}
        )
        
        try:
            engine = RecursiveSimulationEngine(self.mock_orchestrator)
            
            # Test 1: Extreme recursion depth
            try:
                # Create a scenario that will try to recurse very deeply
                deep_context = {
                    'force_recursion': True,
                    'target_depth': 50,  # Way beyond reasonable limits
                    'spawn_count': 3    # Each level spawns 3 more
                }
                
                await engine.submit_simulation(
                    persona=LLMPersona.CHARACTER_SIMULATOR,
                    context_data=deep_context,
                    priority=SimulationPriority.HIGH
                )
                
                # Wait briefly then check if depth limiting is working
                await asyncio.sleep(1)
                
                status = engine.get_status()
                active_simulations = status.get('active_simulations', 0)
                
                # Should have reasonable limits (not exponential explosion)
                if active_simulations > 100:
                    failures.append({
                        'test': 'extreme_recursion_depth',
                        'issue': 'No depth limiting - simulation explosion',
                        'active_simulations': active_simulations
                    })
                
            except Exception as e:
                failures.append({
                    'test': 'extreme_recursion_depth',
                    'issue': 'Exception during deep recursion',
                    'error': str(e)
                })
            
            # Test 2: Memory consumption during recursion
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Submit many recursive simulations
                simulation_ids = []
                for i in range(20):
                    sim_id = await engine.submit_simulation(
                        persona=LLMPersona.SCENE_ARCHITECT,
                        context_data={
                            'recursive_scenario': f'scenario_{i}',
                            'complexity': 'high',
                            'spawn_subsimulations': True
                        }
                    )
                    simulation_ids.append(sim_id)
                
                await asyncio.sleep(2)  # Let simulations run
                
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Flag excessive memory usage
                if memory_increase > 500:  # More than 500MB increase
                    failures.append({
                        'test': 'recursive_memory_consumption',
                        'issue': 'Excessive memory usage during recursion',
                        'memory_increase_mb': memory_increase,
                        'simulations_count': len(simulation_ids)
                    })
                    
            except ImportError:
                logger.warning("psutil not available - skipping memory test")
            except Exception as e:
                failures.append({
                    'test': 'recursive_memory_consumption',
                    'issue': 'Exception during memory test',
                    'error': str(e)
                })
            
            # Test 3: Circular simulation dependencies
            try:
                # Create simulations that reference each other
                circular_context_a = {
                    'depends_on_simulation': 'simulation_b',
                    'simulation_type': 'character_analysis'
                }
                
                circular_context_b = {
                    'depends_on_simulation': 'simulation_a',
                    'simulation_type': 'scene_design'
                }
                
                sim_a = await engine.submit_simulation(
                    persona=LLMPersona.CHARACTER_SIMULATOR,
                    context_data=circular_context_a
                )
                
                sim_b = await engine.submit_simulation(
                    persona=LLMPersona.SCENE_ARCHITECT,
                    context_data=circular_context_b
                )
                
                # Wait and check for deadlock detection
                await asyncio.sleep(3)
                
                status = engine.get_status()
                
                # Should detect and handle circular dependencies
                if 'deadlock_detected' not in status:
                    failures.append({
                        'test': 'circular_dependencies',
                        'issue': 'No deadlock detection mechanism',
                        'sim_a': sim_a,
                        'sim_b': sim_b
                    })
                    
            except Exception as e:
                failures.append({
                    'test': 'circular_dependencies',
                    'issue': 'Exception during circular dependency test',
                    'error': str(e)
                })
                
        except Exception as e:
            failures.append({
                'test': 'recursive_simulation_setup',
                'issue': 'Failed to setup recursive simulation test',
                'error': str(e)
            })
        
        self.test_results['recursion_limits'] = failures
        logger.info(f"Recursion limit tests complete. Found {len(failures)} issues.")
        
        return failures
    
    async def test_agent_collaboration_deadlocks(self):
        """
        Test scenarios where agents might deadlock waiting for each other
        """
        logger.info("Testing agent collaboration deadlocks...")
        
        failures = []
        
        try:
            # Create agents
            char_agent = CharacterSimulatorAgent(
                self.mock_orchestrator,
                template_repository_path=f"{self.temp_dir}/char_deadlock"
            )
            
            scene_agent = SceneArchitectAgent(
                self.mock_orchestrator,
                template_repository_path=f"{self.temp_dir}/scene_deadlock"
            )
            
            # Test 1: Mutual dependency deadlock
            try:
                # Agent A needs result from Agent B
                # Agent B needs result from Agent A
                
                context_a = SimulationContext(
                    agent_id='char_agent',
                    simulation_id='deadlock_a',
                    context_data={
                        'requires_scene_from': 'deadlock_b',
                        'character_name': 'TestChar'
                    },
                    parent_simulation_id=None,
                    depth=0
                )
                
                context_b = SimulationContext(
                    agent_id='scene_agent',
                    simulation_id='deadlock_b',
                    context_data={
                        'requires_character_from': 'deadlock_a',
                        'scene_name': 'TestScene'
                    },
                    parent_simulation_id=None,
                    depth=0
                )
                
                # Start both simulations simultaneously
                task_a = asyncio.create_task(char_agent.run_simulation(context_a.context_data, depth=0))
                task_b = asyncio.create_task(scene_agent.run_simulation(context_b.context_data, depth=0))
                
                # Wait with timeout to detect deadlock
                try:
                    await asyncio.wait_for(
                        asyncio.gather(task_a, task_b),
                        timeout=10.0  # Should complete within 10 seconds
                    )
                except asyncio.TimeoutError:
                    failures.append({
                        'test': 'mutual_dependency_deadlock',
                        'issue': 'Agents deadlocked waiting for each other',
                        'timeout_seconds': 10
                    })
                    
                    # Cancel tasks
                    task_a.cancel()
                    task_b.cancel()
                    
            except Exception as e:
                failures.append({
                    'test': 'mutual_dependency_deadlock',
                    'issue': 'Exception during deadlock test',
                    'error': str(e)
                })
            
            # Test 2: Resource contention deadlock
            try:
                # Multiple agents trying to modify the same template simultaneously
                shared_template_path = f"{self.temp_dir}/shared_template.poml"
                
                # Create initial template
                Path(shared_template_path).write_text("""
                <template>
                    <system>Base template</system>
                    <user>{{context}}</user>
                </template>
                """)
                
                # Create contexts that will modify the same template
                modification_contexts = []
                for i in range(5):
                    context = SimulationContext(
                        agent_id=f'agent_{i}',
                        simulation_id=f'modify_{i}',
                        context_data={
                            'modify_template': shared_template_path,
                            'modification': f'modification_{i}',
                            'concurrent': True
                        },
                        parent_simulation_id=None,
                        depth=0
                    )
                    modification_contexts.append(context)
                
                # Start all modifications simultaneously
                modification_tasks = []
                for context in modification_contexts:
                    if hasattr(char_agent, 'modify_shared_template'):
                        task = asyncio.create_task(
                            char_agent.modify_shared_template(context)
                        )
                        modification_tasks.append(task)
                
                if modification_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*modification_tasks),
                            timeout=15.0
                        )
                    except asyncio.TimeoutError:
                        failures.append({
                            'test': 'resource_contention_deadlock',
                            'issue': 'Resource contention caused deadlock',
                            'concurrent_modifications': len(modification_tasks)
                        })
                        
                        # Cancel remaining tasks
                        for task in modification_tasks:
                            if not task.done():
                                task.cancel()
                else:
                    # If method doesn't exist, note it as potential issue
                    failures.append({
                        'test': 'resource_contention_deadlock',
                        'issue': 'No shared resource modification method found',
                        'note': 'Agents may not handle concurrent template modifications'
                    })
                    
            except Exception as e:
                failures.append({
                    'test': 'resource_contention_deadlock',
                    'issue': 'Exception during resource contention test',
                    'error': str(e)
                })
                
        except Exception as e:
            failures.append({
                'test': 'collaboration_deadlock_setup',
                'issue': 'Failed to setup collaboration deadlock test',
                'error': str(e)
            })
        
        self.test_results['deadlock_scenarios'] = failures
        logger.info(f"Collaboration deadlock tests complete. Found {len(failures)} issues.")
        
        return failures
    
    async def test_llm_provider_failure_handling(self):
        """
        Test how the system handles LLM provider failures and network issues
        """
        logger.info("Testing LLM provider failure handling...")
        
        failures = []
        
        try:
            # Test 1: Network timeout simulation
            async def timeout_mock(*args, **kwargs):
                await asyncio.sleep(60)  # Simulate network timeout
                raise asyncio.TimeoutError("Network timeout")
            
            self.mock_orchestrator.generate_with_persona = timeout_mock
            
            agent = CharacterSimulatorAgent(
                self.mock_orchestrator,
                template_repository_path=f"{self.temp_dir}/timeout_test"
            )
            
            context = SimulationContext(
                agent_id='timeout_agent',
                simulation_id='timeout_test',
                context_data={'character_concept': {'name': 'TimeoutChar'}},
                parent_simulation_id=None,
                depth=0
            )
            
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    agent.run_simulation(context.context_data, depth=0),
                    timeout=10.0
                )
                end_time = time.time()
                
                # Should have failed fast, not waited full timeout
                if end_time - start_time > 30:
                    failures.append({
                        'test': 'network_timeout',
                        'issue': 'No fast failure on network timeout',
                        'wait_time': end_time - start_time
                    })
                    
            except asyncio.TimeoutError:
                # Expected - but should be handled gracefully by agent
                failures.append({
                    'test': 'network_timeout',
                    'issue': 'Timeout not handled gracefully by agent'
                })
            except Exception as e:
                failures.append({
                    'test': 'network_timeout',
                    'issue': 'Unexpected exception during timeout',
                    'error': str(e)
                })
            
            # Test 2: LLM provider returning malformed responses
            malformed_responses = [
                None,  # Null response
                "",    # Empty response
                "Not JSON at all",  # Non-JSON when JSON expected
                {"incomplete": "response"},  # Missing required fields
                {"content": None},  # Null content
                {"content": "Response", "metadata": "should_be_dict"},  # Wrong types
            ]
            
            for i, malformed_response in enumerate(malformed_responses):
                try:
                    self.mock_orchestrator.generate_with_persona = AsyncMock(return_value=Mock(
                        content=malformed_response.get('content') if isinstance(malformed_response, dict) else malformed_response,
                        metadata=malformed_response.get('metadata', {}) if isinstance(malformed_response, dict) else {}
                    ))
                    
                    result = await agent.run_simulation(
                        {'character_concept': {'name': f'MalformedTest{i}'}},
                        depth=0
                    )
                    
                    # Should handle malformed responses gracefully
                    if not result or 'error' not in result:
                        failures.append({
                            'test': f'malformed_response_{i}',
                            'issue': 'Malformed LLM response not handled',
                            'response': str(malformed_response)[:200]
                        })
                        
                except Exception as e:
                    failures.append({
                        'test': f'malformed_response_{i}',
                        'issue': 'Exception on malformed response',
                        'error': str(e),
                        'response': str(malformed_response)[:200]
                    })
            
            # Test 3: Provider authentication failures
            async def auth_failure_mock(*args, **kwargs):
                raise Exception("Authentication failed: Invalid API key")
            
            self.mock_orchestrator.generate_with_persona = auth_failure_mock
            
            try:
                result = await agent.run_simulation(
                    {'character_concept': {'name': 'AuthTest'}},
                    depth=0
                )
                
                # Should handle auth failures gracefully
                if not result or 'authentication_error' not in result:
                    failures.append({
                        'test': 'authentication_failure',
                        'issue': 'Authentication failure not handled gracefully'
                    })
                    
            except Exception as e:
                failures.append({
                    'test': 'authentication_failure', 
                    'issue': 'Authentication failure caused unhandled exception',
                    'error': str(e)
                })
                
        except Exception as e:
            failures.append({
                'test': 'provider_failure_setup',
                'issue': 'Failed to setup provider failure tests',
                'error': str(e)
            })
        
        self.test_results['provider_failures'] = failures
        logger.info(f"Provider failure tests complete. Found {len(failures)} issues.")
        
        return failures
    
    async def test_template_evolution_edge_cases(self):
        """
        Test edge cases in template evolution and learning
        """
        logger.info("Testing template evolution edge cases...")
        
        failures = []
        
        try:
            agent = CharacterSimulatorAgent(
                self.mock_orchestrator,
                template_repository_path=f"{self.temp_dir}/evolution_test"
            )
            
            # Test 1: Template evolution with contradictory feedback
            try:
                # Simulate contradictory performance feedback
                context = SimulationContext(
                    agent_id='evolution_agent',
                    simulation_id='contradictory_test',
                    context_data={'character_concept': {'name': 'TestChar'}},
                    parent_simulation_id=None,
                    depth=0
                )
                
                base_template = await agent.generate_base_template(context)
                
                # Apply contradictory modifications
                contradictory_feedbacks = [
                    {"performance": "excellent", "reason": "very detailed character"},
                    {"performance": "poor", "reason": "too much detail, be concise"},
                    {"performance": "excellent", "reason": "perfect amount of detail"},
                    {"performance": "poor", "reason": "not enough detail"}
                ]
                
                evolved_template = base_template
                for feedback in contradictory_feedbacks:
                    try:
                        evolved_template = await agent.modify_template(
                            evolved_template,
                            context,
                            feedback['reason']
                        )
                    except Exception as e:
                        failures.append({
                            'test': 'contradictory_feedback',
                            'issue': 'Cannot handle contradictory feedback',
                            'error': str(e),
                            'feedback': feedback
                        })
                        break
                
                # Check if template became unstable
                if len(evolved_template) < len(base_template) * 0.5 or len(evolved_template) > len(base_template) * 3:
                    failures.append({
                        'test': 'contradictory_feedback',
                        'issue': 'Template became unstable with contradictory feedback',
                        'base_length': len(base_template),
                        'evolved_length': len(evolved_template)
                    })
                    
            except Exception as e:
                failures.append({
                    'test': 'contradictory_feedback',
                    'issue': 'Exception during contradictory feedback test',
                    'error': str(e)
                })
            
            # Test 2: Evolution with extreme success/failure patterns
            try:
                # Simulate extreme patterns
                extreme_patterns = [
                    {'pattern': 'all_failures', 'success_rate': 0.0, 'count': 100},
                    {'pattern': 'all_successes', 'success_rate': 1.0, 'count': 100},
                    {'pattern': 'alternating', 'success_rate': 0.5, 'count': 1000},
                ]
                
                for pattern in extreme_patterns:
                    if hasattr(agent, 'simulate_performance_pattern'):
                        try:
                            stability_result = await agent.simulate_performance_pattern(
                                pattern['pattern'],
                                pattern['success_rate'],
                                pattern['count']
                            )
                            
                            if not stability_result.get('stable', True):
                                failures.append({
                                    'test': f"extreme_pattern_{pattern['pattern']}",
                                    'issue': 'Template evolution became unstable',
                                    'pattern': pattern
                                })
                                
                        except Exception as e:
                            failures.append({
                                'test': f"extreme_pattern_{pattern['pattern']}",
                                'issue': 'Exception during pattern simulation',
                                'error': str(e)
                            })
                    else:
                        # Note missing functionality
                        failures.append({
                            'test': f"extreme_pattern_{pattern['pattern']}",
                            'issue': 'No performance pattern simulation method',
                            'note': 'Cannot test stability under extreme patterns'
                        })
                        
            except Exception as e:
                failures.append({
                    'test': 'extreme_patterns',
                    'issue': 'Exception during extreme pattern test',
                    'error': str(e)
                })
            
            # Test 3: Template repository corruption
            try:
                # Corrupt the template repository
                template_dir = Path(f"{self.temp_dir}/evolution_test")
                if template_dir.exists():
                    # Create corrupted files
                    (template_dir / "corrupted.poml").write_text("<<INVALID XML>>")
                    (template_dir / "incomplete.poml").write_text("<template><system>")  # Incomplete
                    (template_dir / "empty.poml").write_text("")
                    
                    # Try to load templates
                    try:
                        if hasattr(agent, 'load_templates'):
                            loaded_templates = await agent.load_templates()
                            
                            # Should handle corrupted templates gracefully
                            if not loaded_templates:
                                failures.append({
                                    'test': 'repository_corruption',
                                    'issue': 'All templates rejected due to corruption',
                                    'note': 'No fallback mechanism'
                                })
                        else:
                            failures.append({
                                'test': 'repository_corruption',
                                'issue': 'No template loading method found'
                            })
                            
                    except Exception as e:
                        failures.append({
                            'test': 'repository_corruption',
                            'issue': 'Exception when loading corrupted templates',
                            'error': str(e)
                        })
                        
            except Exception as e:
                failures.append({
                    'test': 'repository_corruption',
                    'issue': 'Exception during corruption test setup',
                    'error': str(e)
                })
                
        except Exception as e:
            failures.append({
                'test': 'template_evolution_setup',
                'issue': 'Failed to setup template evolution tests',
                'error': str(e)
            })
        
        self.test_results['evolution_edge_cases'] = failures
        logger.info(f"Template evolution tests complete. Found {len(failures)} issues.")
        
        return failures
    
    async def test_concurrent_access_race_conditions(self):
        """
        Test for race conditions in concurrent agent operations
        """
        logger.info("Testing concurrent access race conditions...")
        
        failures = []
        
        try:
            # Create multiple agents sharing resources
            agents = []
            for i in range(5):
                agent = CharacterSimulatorAgent(
                    self.mock_orchestrator,
                    template_repository_path=f"{self.temp_dir}/race_test_shared"
                )
                agents.append(agent)
            
            # Test 1: Concurrent template modifications
            try:
                # All agents modify the same base template simultaneously
                base_context = SimulationContext(
                    agent_id='race_test',
                    simulation_id='race_base',
                    context_data={'character_concept': {'name': 'RaceTestChar'}},
                    parent_simulation_id=None,
                    depth=0
                )
                
                base_template = await agents[0].generate_base_template(base_context)
                
                # Concurrent modifications
                modification_tasks = []
                for i, agent in enumerate(agents):
                    context = SimulationContext(
                        agent_id=f'race_agent_{i}',
                        simulation_id=f'race_modify_{i}',
                        context_data={'modification_id': i},
                        parent_simulation_id=None,
                        depth=0
                    )
                    
                    task = asyncio.create_task(
                        agent.modify_template(
                            base_template,
                            context,
                            f"Modification {i}"
                        )
                    )
                    modification_tasks.append(task)
                
                # Wait for all modifications
                try:
                    modified_templates = await asyncio.gather(*modification_tasks)
                    
                    # Check for corruption or inconsistencies
                    unique_templates = set(modified_templates)
                    if len(unique_templates) < len(modified_templates) / 2:
                        failures.append({
                            'test': 'concurrent_template_modification',
                            'issue': 'Too many identical modifications - possible race condition',
                            'unique_count': len(unique_templates),
                            'total_count': len(modified_templates)
                        })
                        
                    # Check for corrupted templates
                    for i, template in enumerate(modified_templates):
                        if not template or len(template) < 10:
                            failures.append({
                                'test': 'concurrent_template_modification',
                                'issue': f'Corrupted template from agent {i}',
                                'template_length': len(template) if template else 0
                            })
                            
                except Exception as e:
                    failures.append({
                        'test': 'concurrent_template_modification',
                        'issue': 'Exception during concurrent modifications',
                        'error': str(e)
                    })
            
            except Exception as e:
                failures.append({
                    'test': 'concurrent_template_modification',
                    'issue': 'Exception setting up concurrent modification test',
                    'error': str(e)
                })
            
            # Test 2: Concurrent simulation runs with shared state
            try:
                simulation_tasks = []
                for i in range(10):
                    context_data = {
                        'character_concept': {'name': f'ConcurrentChar{i}'},
                        'shared_resource': 'global_narrative_state',
                        'concurrent_id': i
                    }
                    
                    task = asyncio.create_task(
                        agents[i % len(agents)].run_simulation(context_data, depth=0)
                    )
                    simulation_tasks.append(task)
                
                # Run all simulations concurrently
                results = await asyncio.gather(*simulation_tasks, return_exceptions=True)
                
                # Check for exceptions or inconsistencies
                exceptions = [r for r in results if isinstance(r, Exception)]
                if exceptions:
                    failures.append({
                        'test': 'concurrent_simulation_runs',
                        'issue': 'Exceptions during concurrent simulations',
                        'exception_count': len(exceptions),
                        'sample_errors': [str(e) for e in exceptions[:3]]
                    })
                
                # Check for result inconsistencies
                successful_results = [r for r in results if not isinstance(r, Exception)]
                if len(successful_results) < len(simulation_tasks) * 0.8:  # Less than 80% success
                    failures.append({
                        'test': 'concurrent_simulation_runs',
                        'issue': 'Low success rate in concurrent simulations',
                        'success_rate': len(successful_results) / len(simulation_tasks),
                        'total_simulations': len(simulation_tasks)
                    })
                    
            except Exception as e:
                failures.append({
                    'test': 'concurrent_simulation_runs',
                    'issue': 'Exception during concurrent simulation test',
                    'error': str(e)
                })
            
            # Test 3: File system race conditions
            try:
                # Multiple agents trying to create the same file
                shared_file_path = f"{self.temp_dir}/race_shared_file.txt"
                
                file_creation_tasks = []
                for i in range(10):
                    async def create_file(agent_id):
                        try:
                            # Simulate file creation race
                            if not Path(shared_file_path).exists():
                                await asyncio.sleep(0.01)  # Small delay to increase race chance
                                Path(shared_file_path).write_text(f"Created by agent {agent_id}")
                                return f"agent_{agent_id}_success"
                            else:
                                return f"agent_{agent_id}_found_existing"
                        except Exception as e:
                            return f"agent_{agent_id}_error_{str(e)}"
                    
                    task = asyncio.create_task(create_file(i))
                    file_creation_tasks.append(task)
                
                creation_results = await asyncio.gather(*file_creation_tasks)
                
                # Check for race condition indicators
                success_count = len([r for r in creation_results if 'success' in r])
                error_count = len([r for r in creation_results if 'error' in r])
                
                if error_count > 0:
                    failures.append({
                        'test': 'filesystem_race_conditions',
                        'issue': 'File system race conditions causing errors',
                        'error_count': error_count,
                        'success_count': success_count,
                        'sample_results': creation_results[:5]
                    })
                    
            except Exception as e:
                failures.append({
                    'test': 'filesystem_race_conditions',
                    'issue': 'Exception during file system race test',
                    'error': str(e)
                })
                
        except Exception as e:
            failures.append({
                'test': 'race_condition_setup',
                'issue': 'Failed to setup race condition tests',
                'error': str(e)
            })
        
        self.test_results['race_conditions'] = failures
        logger.info(f"Race condition tests complete. Found {len(failures)} issues.")
        
        return failures
    
    async def run_all_tests(self):
        """
        Run all breaking point tests and generate comprehensive report
        """
        logger.info("Starting comprehensive breaking point analysis...")
        
        await self.setup()
        
        try:
            # Run all test categories
            test_methods = [
                self.test_template_creation_breaking_points,
                self.test_recursive_simulation_limits,
                self.test_agent_collaboration_deadlocks,
                self.test_llm_provider_failure_handling,
                self.test_template_evolution_edge_cases,
                self.test_concurrent_access_race_conditions
            ]
            
            all_failures = []
            test_results = {}
            
            for test_method in test_methods:
                try:
                    test_name = test_method.__name__
                    logger.info(f"Running {test_name}...")
                    
                    start_time = time.time()
                    failures = await test_method()
                    end_time = time.time()
                    
                    test_results[test_name] = {
                        'failures': failures,
                        'failure_count': len(failures),
                        'execution_time': end_time - start_time,
                        'status': 'completed'
                    }
                    
                    all_failures.extend(failures)
                    
                except Exception as e:
                    test_results[test_method.__name__] = {
                        'failures': [{'test': 'test_execution', 'issue': 'Test method failed', 'error': str(e)}],
                        'failure_count': 1,
                        'execution_time': 0,
                        'status': 'failed'
                    }
                    logger.error(f"Test method {test_method.__name__} failed: {e}")
            
            # Generate comprehensive report
            report = self.generate_breaking_point_report(test_results, all_failures)
            
            return report
            
        finally:
            await self.cleanup()
    
    def generate_breaking_point_report(self, test_results: Dict, all_failures: List[Dict]) -> Dict:
        """
        Generate comprehensive breaking point analysis report
        """
        
        total_tests = sum(len(result['failures']) for result in test_results.values())
        total_failures = len(all_failures)
        
        # Categorize failures by severity
        critical_failures = []
        major_failures = []
        minor_failures = []
        
        for failure in all_failures:
            issue = failure.get('issue', '').lower()
            if any(keyword in issue for keyword in ['deadlock', 'corruption', 'crash', 'exception']):
                critical_failures.append(failure)
            elif any(keyword in issue for keyword in ['timeout', 'memory', 'performance']):
                major_failures.append(failure)
            else:
                minor_failures.append(failure)
        
        # Identify common patterns
        failure_patterns = {}
        for failure in all_failures:
            issue_type = failure.get('test', 'unknown')
            if issue_type not in failure_patterns:
                failure_patterns[issue_type] = []
            failure_patterns[issue_type].append(failure.get('issue', 'Unknown issue'))
        
        # Generate recommendations
        recommendations = []
        
        if critical_failures:
            recommendations.append({
                'priority': 'CRITICAL',
                'issue': 'System stability issues detected',
                'recommendation': 'Implement robust error handling, deadlock detection, and graceful degradation',
                'failures': len(critical_failures)
            })
        
        if major_failures:
            recommendations.append({
                'priority': 'MAJOR',
                'issue': 'Performance and resource management issues',
                'recommendation': 'Add resource limits, timeout handling, and performance monitoring',
                'failures': len(major_failures)
            })
        
        if minor_failures:
            recommendations.append({
                'priority': 'MINOR', 
                'issue': 'Edge case handling improvements needed',
                'recommendation': 'Enhance input validation and edge case handling',
                'failures': len(minor_failures)
            })
        
        report = {
            'summary': {
                'total_test_categories': len(test_results),
                'total_test_cases': total_tests,
                'total_failures': total_failures,
                'failure_rate': total_failures / max(total_tests, 1),
                'critical_failures': len(critical_failures),
                'major_failures': len(major_failures),
                'minor_failures': len(minor_failures)
            },
            'test_results': test_results,
            'failure_breakdown': {
                'critical': critical_failures,
                'major': major_failures,
                'minor': minor_failures
            },
            'failure_patterns': failure_patterns,
            'recommendations': recommendations,
            'detailed_failures': all_failures,
            'generated_at': time.time()
        }
        
        return report

# Utility functions for running tests

async def run_breaking_point_analysis():
    """
    Main function to run breaking point analysis
    """
    tester = BreakingPointTester()
    report = await tester.run_all_tests()
    
    print("\n" + "="*80)
    print("🔍 AUTONOMOUS AGENT BREAKING POINT ANALYSIS")
    print("="*80)
    
    summary = report['summary']
    print("\n📊 SUMMARY:")
    print(f"   Total Test Categories: {summary['total_test_categories']}")
    print(f"   Total Test Cases: {summary['total_test_cases']}")
    print(f"   Total Failures: {summary['total_failures']}")
    print(f"   Failure Rate: {summary['failure_rate']:.1%}")
    
    print("\n🚨 FAILURE BREAKDOWN:")
    print(f"   Critical: {summary['critical_failures']} (System stability issues)")
    print(f"   Major: {summary['major_failures']} (Performance/resource issues)")
    print(f"   Minor: {summary['minor_failures']} (Edge case handling)")
    
    print("\n🔧 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"   [{rec['priority']}] {rec['issue']}")
        print(f"        → {rec['recommendation']}")
        print(f"        → Affects {rec['failures']} failure(s)")
    
    print("\n📋 DETAILED TEST RESULTS:")
    for test_name, result in report['test_results'].items():
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        print(f"   {status_icon} {test_name}: {result['failure_count']} failures in {result['execution_time']:.2f}s")
        
        if result['failures'] and result['failure_count'] > 0:
            print("      Top issues:")
            for failure in result['failures'][:3]:  # Show top 3
                print(f"        • {failure.get('test', 'unknown')}: {failure.get('issue', 'No description')}")
    
    print("\n⚠️  BREAKING POINTS IDENTIFIED:")
    breaking_points = []
    for failure in report['detailed_failures']:
        if any(keyword in failure.get('issue', '').lower() 
               for keyword in ['deadlock', 'explosion', 'corruption', 'crash']):
            breaking_points.append(failure)
    
    if breaking_points:
        for bp in breaking_points[:5]:  # Show top 5 breaking points
            print(f"   🔴 {bp.get('test', 'Unknown')}: {bp.get('issue', 'Unknown issue')}")
    else:
        print("   ✅ No critical breaking points detected!")
    
    print(f"\n📈 SYSTEM ROBUSTNESS SCORE: {max(0, (1 - summary['failure_rate']) * 100):.1f}/100")
    
    return report

if __name__ == "__main__":
    # Run the breaking point analysis
    asyncio.run(run_breaking_point_analysis())
