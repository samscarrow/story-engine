"""
Advanced Character Simulation Experiments
Explore more complex scenarios and real LLM integration
"""

import asyncio
import json
from character_simulation_engine_v2 import (
    CharacterState, EmotionalState, CharacterMemory, 
    SimulationEngine, MockLLM, LMStudioLLM, LLMResponse
)

class EnhancedMockLLM(MockLLM):
    """Enhanced MockLLM with more varied responses based on emphasis and emotion"""
    
    async def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500):
        self.call_count += 1
        await asyncio.sleep(0.05)  # Faster than parent
        
        # Parse character emotional state from prompt
        high_anger = "Anger: 0.9" in prompt or "Anger: 0.8" in prompt
        high_doubt = "Doubt: 0.8" in prompt or "Doubt: 0.9" in prompt  
        high_fear = "Fear: 0.9" in prompt or "Fear: 0.8" in prompt
        high_compassion = "Compassion: 0.9" in prompt or "Compassion: 0.8" in prompt
        
        # Parse emphasis
        emphasis = "neutral"
        if "Emphasis: power" in prompt:
            emphasis = "power"
        elif "Emphasis: doubt" in prompt:
            emphasis = "doubt"
        elif "Emphasis: fear" in prompt:
            emphasis = "fear"
        elif "Emphasis: duty" in prompt:
            emphasis = "duty"
        elif "Emphasis: compassion" in prompt:
            emphasis = "compassion"
        
        # Generate contextual responses
        if emphasis == "power" and high_anger:
            content = json.dumps({
                "dialogue": "Enough! I am the authority here! This man dies if I say he dies!",
                "thought": "They dare question my power? I'll show them what Roman authority means.",
                "action": "Strikes the table violently, rises to dominate the room",
                "emotional_shift": {"anger": 0.1, "confidence": 0.2, "fear": -0.1}
            })
        elif emphasis == "doubt" or high_doubt:
            content = json.dumps({
                "dialogue": "What is truth? You speak as one who knows certainties I cannot grasp...",
                "thought": "This man's words unsettle me. There's something here I don't understand.",
                "action": "Turns away, unable to maintain eye contact, paces uncertainly",
                "emotional_shift": {"doubt": 0.2, "fear": 0.1}
            })
        elif emphasis == "fear" or high_fear:
            content = json.dumps({
                "dialogue": "The crowd grows restless. If I don't act, there will be bloodshed in the streets.",
                "thought": "Caesar's wrath or the mob's fury - both paths lead to my destruction.",
                "action": "Glances nervously toward the sounds of the crowd outside",
                "emotional_shift": {"fear": 0.2, "doubt": 0.1, "confidence": -0.1}
            })
        elif emphasis == "compassion" or high_compassion:
            content = json.dumps({
                "dialogue": "I find no fault in this man. Surely there must be another way...",
                "thought": "He has done nothing deserving death. How can I condemn an innocent?",
                "action": "Looks upon the prisoner with troubled eyes, hesitates",
                "emotional_shift": {"compassion": 0.1, "doubt": 0.1, "fear": 0.05}
            })
        elif emphasis == "duty":
            content = json.dumps({
                "dialogue": "Roman law is clear. I must follow procedure, whatever my personal feelings.",
                "thought": "My duty to Caesar outweighs all other considerations. The law is the law.",
                "action": "Straightens, assumes official bearing, speaks formally",
                "emotional_shift": {"confidence": 0.1, "doubt": -0.05}
            })
        elif "washing" in prompt.lower() or "clean" in prompt.lower():
            content = json.dumps({
                "dialogue": "I wash my hands of this man's blood. See to it yourselves!",
                "thought": "If I cannot save him, at least I can absolve myself before God and history.",
                "action": "Calls for water, washes hands ceremoniously before the crowd",
                "emotional_shift": {"doubt": -0.1, "fear": -0.1, "compassion": 0.1}
            })
        else:
            # Default response
            content = json.dumps({
                "dialogue": "I am trapped between duty and conscience, between Caesar and justice.",
                "thought": "Every choice leads to ruin. Yet choose I must.",
                "action": "Paces the judgment hall, weighing impossible options",
                "emotional_shift": {"doubt": 0.05, "fear": 0.05}
            })
            
        return LLMResponse(content, {"model": "enhanced_mock", "temperature": temperature})

async def character_deep_dive():
    """Explore how a character responds to different scenarios"""
    
    print("🎭 Advanced Character Simulation - Deep Dive Experiments")
    print("=" * 70)
    
    # Use enhanced mock for better variety
    llm = EnhancedMockLLM()
    engine = SimulationEngine(llm_provider=llm, max_concurrent=5)
    
    # Create base Pilate character
    base_pilate = CharacterState(
        id="pontius_pilate",
        name="Pontius Pilate",
        backstory={
            "origin": "Roman equestrian from Samnium",
            "career": "Prefect of Judaea (26-36 CE)",
            "background": "Experienced administrator facing unprecedented situation"
        },
        traits=["pragmatic", "ambitious", "politically_astute", "conflicted"],
        values=["Roman law", "order", "imperial loyalty", "self-preservation"],
        fears=["rebellion", "imperial disfavor", "losing control", "historical judgment"],
        desires=["peace", "career advancement", "moral clarity", "survival"],
        emotional_state=EmotionalState(
            anger=0.3,
            doubt=0.7,
            fear=0.6,
            compassion=0.3,
            confidence=0.4
        ),
        memory=CharacterMemory(),
        current_goal="Navigate this trial without destroying my career or my conscience",
        internal_conflict="Roman duty vs. personal justice"
    )
    
    # EXPERIMENT 1: Scenario Progression
    print("🎬 EXPERIMENT 1: Story Progression - How responses evolve")
    print("-" * 60)
    
    scenarios = [
        ("Initial questioning", "You first encounter Jesus. He's been brought before you accused of claiming to be King of the Jews."),
        ("Private interrogation", "Alone with the prisoner, you question him about his alleged kingdom and his silence before accusers."),
        ("Crowd pressure", "The crowd grows violent, demanding crucifixion. The religious leaders insist he must die."),
        ("Wife's intervention", "Your wife sends word: 'Have nothing to do with this righteous man. I suffered much in a dream because of him.'"),
        ("Final decision", "You must choose: release Jesus and risk rebellion, or condemn him against your conscience."),
        ("Aftermath", "The deed is done. Jesus hangs on the cross. Darkness covers the land.")
    ]
    
    current_pilate = base_pilate
    
    for scene_name, situation in scenarios:
        print(f"\n🎬 Scene: {scene_name}")
        print(f"📝 Situation: {situation}")
        
        result = await engine.run_simulation(
            current_pilate,
            situation,
            emphasis="neutral",
            temperature=0.8
        )
        
        response = result['response']
        print(f"💬 Pilate: \"{response['dialogue']}\"")
        print(f"🤔 Internal: {response['thought']}")
        print(f"🎭 Action: {response['action']}")
        
        # Add this scene to memory for next simulation
        memory_entry = f"{scene_name}: {response['dialogue']}"
        await current_pilate.memory.add_event(memory_entry)
        
        # Show emotional evolution
        emotional_state = current_pilate.emotional_state
        print(f"😰 Current state: Doubt={emotional_state.doubt:.2f}, Fear={emotional_state.fear:.2f}, Compassion={emotional_state.compassion:.2f}")
    
    print("\n" + "=" * 70)
    
    # EXPERIMENT 2: Personality Variants
    print("🎭 EXPERIMENT 2: Character Variants - Different Pilate Personalities")
    print("-" * 60)
    
    variants = {
        "The Ruthless Administrator": {
            "traits": ["ruthless", "ambitious", "cold", "efficient"],
            "emotional_state": EmotionalState(anger=0.6, doubt=0.2, fear=0.3, compassion=0.1, confidence=0.8),
            "internal_conflict": "Efficiency vs. unnecessary complications"
        },
        "The Tormented Idealist": {
            "traits": ["idealistic", "philosophical", "sensitive", "conflicted"],
            "emotional_state": EmotionalState(anger=0.2, doubt=0.9, fear=0.5, compassion=0.8, confidence=0.3),
            "internal_conflict": "Truth vs. political necessity"
        },
        "The Coward": {
            "traits": ["fearful", "weak-willed", "indecisive", "survival-focused"],
            "emotional_state": EmotionalState(anger=0.1, doubt=0.8, fear=0.9, compassion=0.4, confidence=0.2),
            "internal_conflict": "Self-preservation vs. doing what's right"
        }
    }
    
    test_situation = "The crowd demands Jesus' crucifixion. What is your decision?"
    
    for variant_name, variant_config in variants.items():
        print(f"\n🎭 {variant_name}:")
        
        variant_pilate = CharacterState(
            id=f"pilate_{variant_name.lower().replace(' ', '_')}",
            name=f"Pontius Pilate ({variant_name})",
            backstory=base_pilate.backstory,
            traits=variant_config["traits"],
            values=base_pilate.values,
            fears=base_pilate.fears,
            desires=base_pilate.desires,
            emotional_state=variant_config["emotional_state"],
            memory=CharacterMemory(),
            current_goal=base_pilate.current_goal,
            internal_conflict=variant_config["internal_conflict"]
        )
        
        # Test different emphasis modes
        for emphasis in ["power", "fear", "compassion"]:
            result = await engine.run_simulation(
                variant_pilate,
                test_situation,
                emphasis=emphasis,
                temperature=0.7
            )
            
            response = result['response']
            print(f"  🎯 {emphasis.upper()}: \"{response['dialogue']}\"")
    
    print("\n" + "=" * 70)
    
    # EXPERIMENT 3: Temperature Effects
    print("🌡️ EXPERIMENT 3: Temperature Effects - Randomness vs Consistency")
    print("-" * 60)
    
    situation = "You must decide Jesus' fate. The mob screams for blood."
    temperatures = [0.3, 0.7, 1.0, 1.3]
    
    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")
        
        # Run multiple simulations at this temperature
        results = []
        for i in range(3):
            result = await engine.run_simulation(
                base_pilate,
                situation,
                emphasis="doubt",
                temperature=temp
            )
            results.append(result['response']['dialogue'])
        
        print("🎲 Variations:")
        for i, dialogue in enumerate(results, 1):
            print(f"  {i}. \"{dialogue}\"")
        
        # Check for variety
        unique_responses = len(set(results))
        print(f"📊 Uniqueness: {unique_responses}/3 different responses")
    
    print("\n" + "=" * 70)
    print(f"🎉 Advanced experiments completed! Total LLM calls: {llm.call_count}")
    
    return engine, base_pilate

async def real_llm_experiment():
    """Try connecting to real LMStudio (if available)"""
    
    print("\n🔌 BONUS: Real LLM Integration Test")
    print("-" * 50)
    
    # Try to connect to LMStudio
    lmstudio_llm = LMStudioLLM(endpoint="http://localhost:1234/v1")
    
    print("🔍 Testing LMStudio connection...")
    
    try:
        is_available = await lmstudio_llm.health_check()
        if is_available:
            print("✅ LMStudio is available!")
            
            # Create engine with real LLM
            engine = SimulationEngine(llm_provider=lmstudio_llm, max_concurrent=2)
            
            # Simple character for testing
            test_char = CharacterState(
                id="test",
                name="Pontius Pilate", 
                backstory={"origin": "Roman administrator"},
                traits=["conflicted"],
                values=["duty"],
                fears=["chaos"],
                desires=["order"],
                emotional_state=EmotionalState(doubt=0.8),
                memory=CharacterMemory(),
                current_goal="Make a decision"
            )
            
            print("\n🤖 Running simulation with real LLM...")
            
            result = await engine.run_simulation(
                test_char,
                "Jesus stands before you. The crowd demands his death. Decide.",
                emphasis="doubt",
                temperature=0.8
            )
            
            response = result['response']
            print(f"🎭 Real LLM Response:")
            print(f"💬 \"{response['dialogue']}\"")
            print(f"🤔 {response['thought']}")
            print(f"🎬 {response['action']}")
            
        else:
            print("❌ LMStudio not available - using MockLLM instead")
            print("💡 To test with real LLM, start LMStudio server on localhost:1234")
            
    except Exception as e:
        print(f"❌ LMStudio connection failed: {e}")
        print("💡 Make sure LMStudio is running with a loaded model")

if __name__ == "__main__":
    asyncio.run(character_deep_dive())
    # Uncomment to test real LLM:
    # asyncio.run(real_llm_experiment())