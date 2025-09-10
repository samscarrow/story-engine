"""
Autonomous Persona Agent Demonstration
Shows how persona agents create, modify, and evolve their own POML templates
"""

import asyncio
import logging
import json
from pathlib import Path

# Import the autonomous persona system
from story_engine.core.orchestration.llm_orchestrator import StrictLLMOrchestrator
from story_engine.core.orchestration.recursive_simulation_engine import (
    create_recursive_simulation_engine,
    SimulationPriority
)
from story_engine.core.orchestration.unified_llm_orchestrator import LLMPersona

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutonomousPersonaDemonstration:
    """Demonstrates the autonomous persona agent system"""
    
    def __init__(self):
        self.setup_complete = False
        self.base_orchestrator = None
        self.simulation_engine = None
        self.repository_path = Path("./persona_repositories")
        
    async def setup_infrastructure(self):
        """Setup the autonomous persona infrastructure"""
        
        logger.info("Setting up autonomous persona infrastructure...")
        
        # 1. Create base orchestrator (normally from your existing setup)
        providers = {
            'kobold': {
                'endpoint': 'http://localhost:5001',
                'api_type': 'openai'
            },
            # Add other providers as needed
        }
        
        self.base_orchestrator = StrictLLMOrchestrator(providers)
        
        # 2. Create recursive simulation engine with autonomous agents
        self.simulation_engine = create_recursive_simulation_engine(
            orchestrator=self.base_orchestrator,
            repository_path=self.repository_path,
            max_concurrent=3,
            max_depth=4,
            default_timeout=120
        )
        
        # 3. Register some callbacks for monitoring
        self.simulation_engine.register_callback(
            'completion_logger',
            self.log_simulation_completion
        )
        
        self.simulation_engine.register_callback(
            'template_tracker',
            self.track_template_usage
        )
        
        # 4. Start the simulation engine
        await self.simulation_engine.start()
        
        self.setup_complete = True
        logger.info("Autonomous persona infrastructure ready!")
    
    async def log_simulation_completion(self, result):
        """Callback to log simulation completions"""
        logger.info(f"🎭 Simulation completed: {result.persona} at depth {result.depth}")
        logger.info(f"   Status: {result.status.value}, Time: {result.execution_time:.2f}s")
        if result.recursive_results:
            logger.info(f"   Triggered {len(result.recursive_results)} recursive simulations")
    
    async def track_template_usage(self, result):
        """Callback to track template creation and usage"""
        if result.result_data and 'template_used' in result.result_data:
            template_preview = result.template_used[:100] + "..." if len(result.template_used) > 100 else result.template_used
            logger.info(f"📝 Template used by {result.persona}: {template_preview}")
    
    async def demonstrate_character_simulator_agent(self):
        """Demonstrate autonomous character simulator agent"""
        
        logger.info("\n" + "="*60)
        logger.info("🎭 CHARACTER SIMULATOR AGENT DEMONSTRATION")
        logger.info("="*60)
        
        # Complex character simulation context
        character_context = {
            'character': {
                'id': 'marcus_aurelius',
                'name': 'Marcus Aurelius',
                'title': 'Emperor of Rome',
                'traits': ['philosophical', 'duty_bound', 'introspective', 'stoic'],
                'emotional_state': {
                    'anger': 0.1,
                    'doubt': 0.6,
                    'fear': 0.3,
                    'compassion': 0.8,
                    'confidence': 0.7
                },
                'backstory': {
                    'origin': 'Roman nobility',
                    'education': 'Philosophy and rhetoric',
                    'career': 'Emperor and philosopher'
                },
                'current_concerns': [
                    'Germanic tribes threatening borders',
                    'Plague affecting the empire',
                    'Personal philosophical growth'
                ]
            },
            'situation': 'A plague has struck Rome. Citizens demand action while advisors suggest contradictory solutions. The emperor must decide between quarantine measures that hurt trade and military action that spreads resources thin.',
            'emphasis': 'philosophical_duty',
            'complexity_factors': [
                'multiple_stakeholders',
                'long_term_consequences', 
                'moral_implications',
                'resource_constraints'
            ],
            'success_criteria': [
                'demonstrates stoic philosophy',
                'addresses practical concerns',
                'shows character growth',
                'maintains imperial dignity'
            ],
            'constraints': {
                'historical_accuracy': 'high',
                'philosophical_consistency': 'required',
                'response_length': 'detailed_but_focused'
            }
        }
        
        # Submit simulation - the agent will create its own template
        request_id = await self.simulation_engine.submit_simulation(
            persona=LLMPersona.CHARACTER_SIMULATOR,
            context_data=character_context,
            priority=SimulationPriority.HIGH,
            metadata={'demonstration': 'character_simulator', 'complexity': 'high'}
        )
        
        logger.info(f"Submitted character simulation request: {request_id}")
        
        # Wait for completion and show results
        await self.wait_and_show_results(request_id, "Character Simulator")
        
        # Now modify the context and see template adaptation
        logger.info("\n📝 TEMPLATE ADAPTATION DEMONSTRATION")
        logger.info("Modifying context to trigger template adaptation...")
        
        modified_context = character_context.copy()
        modified_context['situation'] = 'The same plague crisis, but now a trusted advisor has been found taking bribes from merchants seeking exemptions from quarantine.'
        modified_context['emphasis'] = 'justice_vs_mercy'
        modified_context['additional_complexity'] = 'betrayal_by_trusted_ally'
        
        request_id_2 = await self.simulation_engine.submit_simulation(
            persona=LLMPersona.CHARACTER_SIMULATOR,
            context_data=modified_context,
            priority=SimulationPriority.HIGH,
            metadata={'demonstration': 'template_adaptation', 'parent_scenario': request_id}
        )
        
        await self.wait_and_show_results(request_id_2, "Adapted Character Simulator")
    
    async def demonstrate_scene_architect_agent(self):
        """Demonstrate autonomous scene architect agent"""
        
        logger.info("\n" + "="*60)
        logger.info("🏛️ SCENE ARCHITECT AGENT DEMONSTRATION")
        logger.info("="*60)
        
        # Complex scene creation context
        scene_context = {
            'beat': {
                'name': 'The Emperor\'s Dilemma',
                'purpose': 'Force the protagonist to choose between competing moral imperatives',
                'tension': 9,
                'dramatic_function': 'climactic_decision',
                'expected_outcome': 'character_revelation'
            },
            'characters': [
                {
                    'id': 'marcus_aurelius',
                    'name': 'Marcus Aurelius',
                    'role': 'protagonist_decision_maker',
                    'emotional_state': 'conflicted',
                    'objectives': ['maintain_empire', 'uphold_philosophy', 'protect_citizens']
                },
                {
                    'id': 'advisor_brutus',
                    'name': 'Advisor Brutus',
                    'role': 'pragmatic_counsel',
                    'emotional_state': 'urgent',
                    'objectives': ['practical_solutions', 'empire_stability']
                },
                {
                    'id': 'citizen_delegation',
                    'name': 'Citizen Delegation',
                    'role': 'affected_populace',
                    'emotional_state': 'desperate',
                    'objectives': ['immediate_relief', 'survival']
                }
            ],
            'previous_context': 'The emperor has spent the morning in philosophical meditation, seeking guidance from Stoic principles. Reports of the plague\'s spread have worsened.',
            'environmental_requirements': [
                'imperial_palace_setting',
                'tension_building_atmosphere',
                'multiple_character_positioning',
                'symbolic_elements'
            ],
            'dramatic_requirements': [
                'clear_stakes_for_all_characters',
                'impossible_choice_setup',
                'sensory_details_for_immersion',
                'emotional_resonance'
            ],
            'success_criteria': [
                'creates_dramatic_tension',
                'establishes_clear_character_positions',
                'provides_rich_environmental_detail',
                'sets_up_meaningful_choices'
            ]
        }
        
        # Submit scene architecture request
        request_id = await self.simulation_engine.submit_simulation(
            persona=LLMPersona.SCENE_ARCHITECT,
            context_data=scene_context,
            priority=SimulationPriority.HIGH,
            metadata={'demonstration': 'scene_architect', 'complexity': 'high'}
        )
        
        logger.info(f"Submitted scene architecture request: {request_id}")
        
        await self.wait_and_show_results(request_id, "Scene Architect")
        
        # Demonstrate recursive scene development
        logger.info("\n🔄 RECURSIVE SCENE DEVELOPMENT")
        logger.info("Triggering recursive scene elaboration...")
        
        elaboration_context = scene_context.copy()
        elaboration_context['focus'] = 'character_positioning_and_staging'
        elaboration_context['elaboration_type'] = 'blocking_and_movement'
        elaboration_context['parent_scene'] = request_id
        
        recursive_request = await self.simulation_engine.submit_simulation(
            persona=LLMPersona.SCENE_ARCHITECT,
            context_data=elaboration_context,
            priority=SimulationPriority.NORMAL,
            parent_simulation_id=request_id,
            depth=1,
            metadata={'demonstration': 'recursive_elaboration'}
        )
        
        await self.wait_and_show_results(recursive_request, "Recursive Scene Elaboration")
    
    async def demonstrate_multi_persona_collaboration(self):
        """Demonstrate multiple persona agents working together"""
        
        logger.info("\n" + "="*60)
        logger.info("🤝 MULTI-PERSONA COLLABORATION DEMONSTRATION")
        logger.info("="*60)
        
        # Start with scene architecture
        collaboration_context = {
            'project_type': 'complete_story_segment',
            'narrative_context': {
                'setting': 'Ancient Rome during plague',
                'protagonist': 'Marcus Aurelius',
                'central_conflict': 'duty vs philosophy in crisis',
                'target_audience': 'historical fiction readers'
            },
            'collaboration_phases': [
                'scene_architecture',
                'character_development', 
                'narrative_analysis'
            ]
        }
        
        # Phase 1: Scene Architecture
        scene_request_id = await self.simulation_engine.submit_simulation(
            persona=LLMPersona.SCENE_ARCHITECT,
            context_data={
                'collaboration_project': True,
                'phase': 'scene_architecture',
                **collaboration_context['narrative_context'],
                'beat': {
                    'name': 'Imperial Crisis Council',
                    'purpose': 'Establish the central conflict',
                    'tension': 8
                }
            },
            priority=SimulationPriority.HIGH,
            metadata={'collaboration': True, 'phase': 1}
        )
        
        logger.info(f"Phase 1 - Scene Architecture: {scene_request_id}")
        
        # Wait for scene architecture to complete
        scene_result = await self.wait_for_completion(scene_request_id)
        
        # Phase 2: Character Development (based on scene)
        character_request_id = await self.simulation_engine.submit_simulation(
            persona=LLMPersona.CHARACTER_SIMULATOR,
            context_data={
                'collaboration_project': True,
                'phase': 'character_development',
                'scene_context': scene_result.result_data if scene_result else {},
                'character': {
                    'id': 'marcus_aurelius',
                    'name': 'Marcus Aurelius',
                    'base_traits': ['philosophical', 'duty_bound']
                },
                'development_focus': 'response_to_crisis'
            },
            priority=SimulationPriority.HIGH,
            parent_simulation_id=scene_request_id,
            depth=1,
            metadata={'collaboration': True, 'phase': 2}
        )
        
        logger.info(f"Phase 2 - Character Development: {character_request_id}")
        
        # Wait for character development
        character_result = await self.wait_for_completion(character_request_id)
        
        # Phase 3: Narrative Analysis
        analysis_request_id = await self.simulation_engine.submit_simulation(
            persona=LLMPersona.NARRATIVE_ANALYST,
            context_data={
                'collaboration_project': True,
                'phase': 'narrative_analysis',
                'content_to_analyze': {
                    'scene': scene_result.result_data if scene_result else {},
                    'character_development': character_result.result_data if character_result else {}
                },
                'analysis_focus': ['dramatic_effectiveness', 'character_consistency', 'historical_authenticity']
            },
            priority=SimulationPriority.NORMAL,
            parent_simulation_id=character_request_id,
            depth=2,
            metadata={'collaboration': True, 'phase': 3, 'final_phase': True}
        )
        
        logger.info(f"Phase 3 - Narrative Analysis: {analysis_request_id}")
        
        await self.wait_and_show_results(analysis_request_id, "Collaborative Analysis")
        
        # Show the complete collaboration tree
        logger.info("\n🌳 COLLABORATION TREE")
        collaboration_tree = await self.simulation_engine.get_simulation_tree(scene_request_id)
        self.print_simulation_tree(collaboration_tree)
    
    async def wait_and_show_results(self, request_id: str, context_name: str):
        """Wait for simulation completion and show results"""
        
        logger.info(f"⏳ Waiting for {context_name} simulation to complete...")
        
        result = await self.wait_for_completion(request_id)
        
        if result:
            logger.info(f"✅ {context_name} completed successfully!")
            logger.info(f"   Execution time: {result.execution_time:.2f} seconds")
            
            if result.result_data:
                # Show key information about the result
                result_summary = {
                    'simulation_id': result.simulation_id,
                    'persona': result.persona,
                    'status': result.status.value,
                    'depth': result.depth,
                    'template_length': len(result.template_used),
                    'has_recursive_results': len(result.recursive_results) > 0
                }
                
                logger.info(f"   Result summary: {json.dumps(result_summary, indent=2)}")
                
                # Show first 200 characters of primary result
                primary_result = result.result_data.get('primary_result', {})
                if hasattr(primary_result, 'text'):
                    preview = primary_result.text[:200] + "..." if len(primary_result.text) > 200 else primary_result.text
                elif hasattr(primary_result, 'content'):
                    preview = primary_result.content[:200] + "..." if len(primary_result.content) > 200 else primary_result.content
                else:
                    preview = str(primary_result)[:200] + "..."
                
                logger.info(f"   Result preview: {preview}")
            
        else:
            logger.error(f"❌ {context_name} simulation failed or timed out")
    
    async def wait_for_completion(self, request_id: str, max_wait_seconds: int = 60):
        """Wait for a simulation to complete"""
        
        wait_time = 0
        while wait_time < max_wait_seconds:
            result = await self.simulation_engine.get_simulation_result(request_id)
            if result:
                return result
            
            await asyncio.sleep(1)
            wait_time += 1
        
        logger.warning(f"Simulation {request_id} did not complete within {max_wait_seconds} seconds")
        return None
    
    def print_simulation_tree(self, tree_node: dict, indent: int = 0):
        """Pretty print a simulation tree"""
        
        prefix = "  " * indent
        sim_id = tree_node['simulation_id']
        result = tree_node.get('result')
        
        if result:
            status = result['status']
            persona = result['persona']
            depth = result['depth']
            logger.info(f"{prefix}├─ {sim_id[:8]}... ({persona}, depth={depth}, {status})")
        else:
            logger.info(f"{prefix}├─ {sim_id[:8]}... (pending)")
        
        for child in tree_node.get('children', []):
            self.print_simulation_tree(child, indent + 1)
    
    async def demonstrate_template_evolution(self):
        """Demonstrate how templates evolve and improve over time"""
        
        logger.info("\n" + "="*60)
        logger.info("🧬 TEMPLATE EVOLUTION DEMONSTRATION")
        logger.info("="*60)
        
        # Create similar scenarios to see template reuse and modification
        base_scenario = {
            'character': {
                'id': 'historical_leader',
                'name': 'Generic Historical Leader',
                'traits': ['decisive', 'responsible']
            },
            'situation': 'A crisis requires immediate decision',
            'emphasis': 'leadership'
        }
        
        scenarios = [
            # Scenario 1: Basic leadership
            {**base_scenario, 'scenario_name': 'basic_leadership'},
            
            # Scenario 2: Leadership with moral complexity
            {
                **base_scenario,
                'scenario_name': 'moral_leadership',
                'situation': 'A crisis requires choosing between saving many at cost to few',
                'moral_complexity': 'high'
            },
            
            # Scenario 3: Leadership with time pressure
            {
                **base_scenario,
                'scenario_name': 'urgent_leadership', 
                'situation': 'Immediate decision needed with incomplete information',
                'time_pressure': 'extreme',
                'information_availability': 'limited'
            },
            
            # Scenario 4: Leadership with political implications
            {
                **base_scenario,
                'scenario_name': 'political_leadership',
                'situation': 'Decision affects multiple stakeholder groups with conflicting interests',
                'political_complexity': 'high',
                'stakeholder_count': 'multiple'
            }
        ]
        
        request_ids = []
        
        logger.info("Running scenarios to demonstrate template evolution...")
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\n📊 Running Scenario {i}: {scenario['scenario_name']}")
            
            request_id = await self.simulation_engine.submit_simulation(
                persona=LLMPersona.CHARACTER_SIMULATOR,
                context_data=scenario,
                priority=SimulationPriority.NORMAL,
                metadata={
                    'evolution_demo': True,
                    'scenario_number': i,
                    'scenario_name': scenario['scenario_name']
                }
            )
            
            request_ids.append(request_id)
            logger.info(f"   Submitted: {request_id}")
            
            # Add slight delay to allow for sequential template evolution
            await asyncio.sleep(2)
        
        # Wait for all scenarios to complete
        logger.info("\n⏳ Waiting for all scenarios to complete...")
        
        results = []
        for request_id in request_ids:
            result = await self.wait_for_completion(request_id)
            results.append(result)
        
        # Analyze template evolution
        logger.info("\n📈 TEMPLATE EVOLUTION ANALYSIS")
        
        for i, result in enumerate(results, 1):
            if result:
                template_length = len(result.template_used)
                execution_time = result.execution_time
                
                logger.info(f"Scenario {i}:")
                logger.info(f"  Template length: {template_length} characters")
                logger.info(f"  Execution time: {execution_time:.2f}s")
                logger.info(f"  Status: {result.status.value}")
    
    async def show_engine_status(self):
        """Show comprehensive engine status"""
        
        logger.info("\n" + "="*60)
        logger.info("📊 ENGINE STATUS REPORT")
        logger.info("="*60)
        
        status = self.simulation_engine.get_engine_status()
        
        logger.info(f"Engine Running: {status['engine_running']}")
        logger.info(f"Available Agents: {len(status['available_agents'])}")
        
        for persona in status['available_agents']:
            logger.info(f"  - {persona.value}")
        
        scheduler_status = status['scheduler_status']
        logger.info("\nScheduler Status:")
        logger.info(f"  Pending Requests: {scheduler_status['pending_requests']}")
        logger.info(f"  Running Simulations: {scheduler_status['running_simulations']}")
        logger.info(f"  Completed Results: {scheduler_status['completed_results']}")
        logger.info(f"  Max Concurrent: {scheduler_status['max_concurrent']}")
        
        stats = scheduler_status['statistics']
        logger.info("\nStatistics:")
        logger.info(f"  Total Requests: {stats['total_requests']}")
        logger.info(f"  Completed: {stats['completed']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Average Execution Time: {stats['average_execution_time']:.2f}s")
        
        if stats['persona_usage']:
            logger.info(f"  Persona Usage: {stats['persona_usage']}")
        
        if stats['depth_distribution']:
            logger.info(f"  Depth Distribution: {stats['depth_distribution']}")
        
        logger.info(f"\nSimulation Tree Size: {status['simulation_tree_size']}")
        logger.info(f"Registered Callbacks: {status['registered_callbacks']}")
        logger.info(f"Max Depth: {status['max_depth']}")
        logger.info(f"Default Timeout: {status['default_timeout']}s")

async def run_full_demonstration():
    """Run the complete autonomous persona demonstration"""
    
    print("\n" + "🎭" * 20)
    print("AUTONOMOUS PERSONA AGENTS DEMONSTRATION")
    print("Self-Modifying POML Templates & Recursive Simulations")
    print("🎭" * 20 + "\n")
    
    demo = AutonomousPersonaDemonstration()
    
    try:
        # Setup infrastructure
        await demo.setup_infrastructure()
        
        # Show initial status
        await demo.show_engine_status()
        
        # Run demonstrations
        await demo.demonstrate_character_simulator_agent()
        await demo.demonstrate_scene_architect_agent()
        await demo.demonstrate_multi_persona_collaboration()
        await demo.demonstrate_template_evolution()
        
        # Show final status
        await demo.show_engine_status()
        
        logger.info("\n🎉 DEMONSTRATION COMPLETE!")
        logger.info("The autonomous persona agents have successfully:")
        logger.info("  ✅ Created their own POML templates dynamically")
        logger.info("  ✅ Modified templates based on context")
        logger.info("  ✅ Executed recursive simulations")
        logger.info("  ✅ Collaborated across multiple personas")
        logger.info("  ✅ Evolved templates through experience")
        
        # Cleanup
        await demo.simulation_engine.stop()
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        if demo.simulation_engine:
            await demo.simulation_engine.stop()

if __name__ == "__main__":
    # Run the demonstration
    print("🚀 Starting Autonomous Persona Agent Demonstration...")
    print("Note: This requires a running LLM instance (KoboldCpp, LMStudio, etc.)")
    
    # Uncomment to run the actual demonstration
    # asyncio.run(run_full_demonstration())
    
    print("\n📝 CONCEPT SUMMARY:")
    print("""
    🎭 AUTONOMOUS PERSONA AGENTS:
    
    Each persona is now an autonomous agent that:
    ✅ Creates its own POML templates on-demand
    ✅ Modifies templates based on simulation context  
    ✅ Learns from success/failure to improve templates
    ✅ Spawns recursive sub-simulations as needed
    ✅ Maintains its own template repository
    ✅ Collaborates with other persona agents
    
    🔄 RECURSIVE SIMULATION PROCESS:
    
    1. Agent receives simulation context
    2. Searches for similar existing templates
    3. Creates new template or modifies existing one
    4. Executes simulation with generated template
    5. Evaluates results and updates template performance
    6. Spawns recursive simulations if needed
    7. Stores learned templates for future use
    
    🧬 TEMPLATE EVOLUTION:
    
    Templates improve over time through:
    - Success/failure tracking
    - Context-specific modifications
    - Performance-based selection
    - Recursive refinement
    - Cross-agent collaboration
    
    Each simulation run becomes a subprocess of the persona agent,
    with the agent dynamically creating the perfect template for
    that specific context and learning from the results.
    """)
