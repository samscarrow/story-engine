"""
Real Character Simulation - No Mock Data
Production-ready character simulation using actual LLMs only
"""

import asyncio
import time
from typing import Dict, Any, Optional
from character_simulation_engine_v2 import (
    CharacterState, EmotionalState, CharacterMemory,
    SimulationEngine, LMStudioLLM, OpenAILLM
)

class CharacterSimulation:
    """Main simulation class for real LLM character interactions"""
    
    def __init__(self, llm_provider):
        """Initialize with a real LLM provider"""
        self.engine = SimulationEngine(
            llm_provider=llm_provider,
            max_concurrent=1  # Sequential for narrative coherence
        )
        self.results = []
    
    async def run_scene(self, character: CharacterState, scene: Dict[str, Any]) -> Optional[Dict]:
        """Run a single scene simulation"""
        print(f"\n{'='*70}")
        print(f"📖 {scene['name'].upper()}")
        print(f"{'='*70}")
        print(f"📝 {scene['situation']}")
        
        if 'emphasis' in scene:
            print(f"🎯 Emphasis: {scene['emphasis']}")
        
        if 'memory' in scene and scene['memory']:
            await character.memory.add_event(scene['memory'])
            print(f"💭 Memory: {scene['memory']}")
        
        try:
            print("⏳ Generating response...")
            start = time.time()
            
            result = await self.engine.run_simulation(
                character,
                scene['situation'],
                emphasis=scene.get('emphasis', 'neutral'),
                temperature=scene.get('temperature', 0.7)
            )
            
            elapsed = time.time() - start
            response = result['response']
            
            if isinstance(response, dict):
                print(f"\n⏱️  Response time: {elapsed:.1f}s")
                
                # Display character response
                dialogue = response.get('dialogue', '')
                thought = response.get('thought', '')
                action = response.get('action', '')
                
                print(f"💬 \"{dialogue}\"")
                print(f"🤔 *{thought}*")
                print(f"🎬 {action}")
                
                # Show emotional changes
                if 'emotional_shift' in response:
                    shifts = response['emotional_shift']
                    significant = []
                    for emotion, change in shifts.items():
                        if isinstance(change, (int, float)) and abs(change) > 0.05:
                            significant.append(f"{emotion}: {change:+.2f}")
                    
                    if significant:
                        print(f"📊 Emotional shifts: {', '.join(significant)}")
                
                # Store result
                self.results.append({
                    'scene': scene['name'],
                    'response': response,
                    'time': elapsed,
                    'emotions': {
                        'anger': character.emotional_state.anger,
                        'doubt': character.emotional_state.doubt,
                        'fear': character.emotional_state.fear,
                        'compassion': character.emotional_state.compassion,
                        'confidence': character.emotional_state.confidence
                    }
                })
                
                return response
            else:
                print(f"⚠️  Unexpected response format: {type(response)}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def display_emotional_journey(self):
        """Display the emotional evolution across scenes"""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{'='*70}")
        print("📊 EMOTIONAL JOURNEY")
        print(f"{'='*70}")
        
        print(f"{'Scene':<20} {'Anger':>7} {'Doubt':>7} {'Fear':>7} {'Comp':>7} {'Conf':>7}")
        print("-" * 70)
        
        for result in self.results:
            emotions = result['emotions']
            scene = result['scene'][:19]
            print(f"{scene:<20} {emotions['anger']:>7.2f} {emotions['doubt']:>7.2f} "
                  f"{emotions['fear']:>7.2f} {emotions['compassion']:>7.2f} "
                  f"{emotions['confidence']:>7.2f}")
        
        # Calculate total changes
        if len(self.results) > 1:
            first = self.results[0]['emotions']
            last = self.results[-1]['emotions']
            
            print("\n" + "-" * 70)
            print("TOTAL CHANGES:")
            
            for emotion in ['anger', 'doubt', 'fear', 'compassion', 'confidence']:
                change = last[emotion] - first[emotion]
                if abs(change) > 0.05:
                    direction = "↗" if change > 0 else "↘"
                    print(f"  {emotion.capitalize()}: {direction} {abs(change):.2f}")
        
        # Performance stats
        if self.results:
            avg_time = sum(r['time'] for r in self.results) / len(self.results)
            print(f"\n⏱️  Average response time: {avg_time:.1f}s")
            print(f"📝 Total scenes: {len(self.results)}")


async def create_character(name: str = "Pontius Pilate") -> CharacterState:
    """Create a character for simulation"""
    
    if name == "Pontius Pilate":
        return CharacterState(
            id="pilate",
            name="Pontius Pilate",
            backstory={
                "origin": "Roman equestrian from Samnium",
                "position": "Fifth Prefect of Roman Judaea (26-36 AD)",
                "reputation": "Pragmatic administrator, sometimes harsh",
                "current_situation": "Presiding over the trial of Jesus of Nazareth"
            },
            traits=["pragmatic", "political", "conflicted", "duty-bound", "increasingly desperate"],
            values=["Roman law", "order", "personal survival", "hidden sense of justice"],
            fears=["rebellion", "Caesar's wrath", "divine judgment", "losing control"],
            desires=["peace in province", "clear conscience", "escape from situation"],
            emotional_state=EmotionalState(
                anger=0.2,
                doubt=0.3,
                fear=0.3,
                compassion=0.4,
                confidence=0.7
            ),
            memory=CharacterMemory(),
            current_goal="Navigate trial without causing unrest",
            internal_conflict="Political duty vs moral conscience"
        )
    
    # Add more characters as needed
    return None


async def run_pilate_trial():
    """Run the full Pilate trial sequence"""
    
    print("🎭 PONTIUS PILATE - TRIAL OF JESUS")
    print("=" * 70)
    
    # Setup LLM - try LMStudio first
    print("\n🔌 Connecting to LLM...")
    
    llm = None
    
    # Try LMStudio with Gemma
    try:
        lmstudio = LMStudioLLM(
            endpoint="http://localhost:1234/v1",
            model="google/gemma-3-12b"
        )
        if await lmstudio.health_check():
            print("✅ Connected to Gemma-3-12b via LMStudio")
            llm = lmstudio
    except Exception as e:
        print(f"❌ LMStudio not available: {e}")
    
    # Try OpenAI if configured
    if not llm:
        try:
            import os
            if os.getenv("OPENAI_API_KEY"):
                openai = OpenAILLM()
                print("✅ Connected to OpenAI")
                llm = openai
        except Exception as e:
            print(f"❌ OpenAI not available: {e}")
    
    if not llm:
        print("\n❌ No LLM available. Please:")
        print("1. Start LMStudio and load a model (e.g., Gemma-3-12b), OR")
        print("2. Set OPENAI_API_KEY environment variable")
        return
    
    # Create character and simulation
    pilate = await create_character("Pontius Pilate")
    simulation = CharacterSimulation(llm)
    
    # Define the trial scenes
    scenes = [
        {
            "name": "Initial Confrontation",
            "situation": "Jesus is brought before you by the Sanhedrin. They accuse him of claiming to be King of the Jews, a direct challenge to Caesar. The morning sun casts long shadows in your judgment hall.",
            "emphasis": "duty",
            "temperature": 0.7
        },
        {
            "name": "The Question of Truth",
            "situation": "Jesus speaks: 'My kingdom is not of this world. I came to bear witness to the truth. Everyone who is of the truth hears my voice.' His calmness is unsettling.",
            "emphasis": "doubt",
            "memory": "Jesus spoke of truth and kingdoms beyond Rome",
            "temperature": 0.8
        },
        {
            "name": "Finding No Fault",
            "situation": "After interrogation, you find no grounds for execution. The man seems delusional perhaps, but not dangerous. Yet the crowd outside grows louder, more demanding.",
            "emphasis": "compassion",
            "memory": "I found no fault worthy of death",
            "temperature": 0.7
        },
        {
            "name": "The Passover Choice",
            "situation": "You offer the traditional Passover pardoning: 'Whom shall I release - Barabbas the murderer, or Jesus called Christ?' The crowd roars: 'Barabbas!' You are stunned.",
            "emphasis": "fear",
            "memory": "They chose to free a murderer over this prophet",
            "temperature": 0.8
        },
        {
            "name": "Claudia's Warning",
            "situation": "Your wife Claudia sends an urgent message: 'Have nothing to do with that righteous man. I have suffered much in a dream because of him.' The gods themselves seem to speak.",
            "emphasis": "fear",
            "memory": "Even Claudia's dreams warn against this",
            "temperature": 0.8
        },
        {
            "name": "Blood on Their Heads",
            "situation": "The crowd screams: 'Crucify him! His blood be on us and our children!' They threaten riot. You know Caesar will not tolerate disorder. The choice narrows to political survival.",
            "emphasis": "power",
            "memory": "They demanded his blood be on their heads",
            "temperature": 0.7
        },
        {
            "name": "Washing Hands",
            "situation": "You call for a basin of water. Before the crowd, you wash your hands, declaring: 'I am innocent of this man's blood.' But the water cannot cleanse what you feel inside.",
            "emphasis": "neutral",
            "memory": "I washed my hands but felt no cleaner",
            "temperature": 0.9
        }
    ]
    
    # Run the simulation
    print(f"\n🎬 Beginning simulation with {len(scenes)} scenes...")
    print(f"📍 Initial state: Confidence={pilate.emotional_state.confidence:.2f}, "
          f"Doubt={pilate.emotional_state.doubt:.2f}\n")
    
    for scene in scenes:
        await simulation.run_scene(pilate, scene)
        await asyncio.sleep(0.5)  # Brief pause between scenes
    
    # Display journey
    simulation.display_emotional_journey()
    
    # Final state
    print(f"\n{'='*70}")
    print("🎭 FINAL CHARACTER STATE")
    print(f"{'='*70}")
    print(f"\n{pilate.name}:")
    print(f"  Anger: {pilate.emotional_state.anger:.2f}")
    print(f"  Doubt: {pilate.emotional_state.doubt:.2f}")
    print(f"  Fear: {pilate.emotional_state.fear:.2f}")
    print(f"  Compassion: {pilate.emotional_state.compassion:.2f}")
    print(f"  Confidence: {pilate.emotional_state.confidence:.2f}")
    
    if hasattr(pilate.memory, 'recent_events') and pilate.memory.recent_events:
        print("\n💭 Final memories:")
        for memory in pilate.memory.recent_events[-3:]:
            print(f"  • {memory}")
    
    print("\n✨ Simulation complete!")


async def test_single_response():
    """Test a single character response"""
    
    print("🧪 SINGLE RESPONSE TEST")
    print("=" * 70)
    
    # Connect to LLM
    llm = LMStudioLLM(endpoint="http://localhost:1234/v1", model="google/gemma-3-12b")
    
    if not await llm.health_check():
        print("❌ LLM not available")
        return
    
    print("✅ Connected to LLM")
    
    # Create simple character
    character = CharacterState(
        id="test",
        name="Test Character",
        backstory={"role": "Roman official"},
        traits=["thoughtful"],
        values=["order"],
        fears=["chaos"],
        desires=["peace"],
        emotional_state=EmotionalState(),
        memory=CharacterMemory(),
        current_goal="Make decision"
    )
    
    # Create simulation
    sim = CharacterSimulation(llm)
    
    # Run single scene
    scene = {
        "name": "Test Scene",
        "situation": "A crowd demands justice. You must decide quickly.",
        "emphasis": "fear",
        "temperature": 0.7
    }
    
    await sim.run_scene(character, scene)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test mode
        asyncio.run(test_single_response())
    else:
        # Full simulation
        asyncio.run(run_pilate_trial())