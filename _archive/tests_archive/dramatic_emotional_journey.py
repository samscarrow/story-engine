"""
Dramatic Emotional Journey Experiment
Shows dramatic emotional evolution with more intense shifts and better visualization
"""

import asyncio
import json
import yaml
from pathlib import Path
from character_simulation_engine_v2 import (
    CharacterState, EmotionalState, CharacterMemory, 
    SimulationEngine, LMStudioLLM, MockLLM, LLMResponse
)

class DramaticMockLLM(MockLLM):
    """MockLLM with dramatic emotional responses and realistic variety"""
    
    def __init__(self):
        super().__init__()
        self.scene_count = 0
        
    async def generate_response(self, prompt, temperature=0.7, max_tokens=500):
        self.call_count += 1
        self.scene_count += 1
        await asyncio.sleep(0.1)
        
        # Extract emotional context
        emotions = self._extract_current_emotions(prompt)
        emphasis = self._extract_emphasis(prompt)
        situation = self._extract_situation(prompt)
        
        # Generate dramatic responses based on escalating tension
        response_data = self._generate_dramatic_response(
            emotions, emphasis, situation, self.scene_count, temperature
        )
        
        return LLMResponse(json.dumps(response_data), {
            "model": "dramatic_mock",
            "temperature": temperature
        })
    
    def _extract_current_emotions(self, prompt):
        emotions = {}
        import re
        for emotion in ['anger', 'doubt', 'fear', 'compassion', 'confidence']:
            pattern = rf"{emotion.title()}: (\d+\.?\d*)"
            match = re.search(pattern, prompt)
            if match:
                emotions[emotion] = float(match.group(1))
        return emotions
    
    def _extract_emphasis(self, prompt):
        if "Emphasis: " in prompt:
            emphasis_line = [line for line in prompt.split('\n') if 'Emphasis:' in line][0]
            return emphasis_line.split('Emphasis: ')[1].strip()
        return 'neutral'
    
    def _extract_situation(self, prompt):
        if "Situation: " in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if line.startswith('Situation:'):
                    return line.replace('Situation:', '').strip()
        return "facing a difficult decision"
    
    def _generate_dramatic_response(self, emotions, emphasis, situation, scene_num, temperature):
        """Generate increasingly dramatic responses as story progresses"""
        
        # Get current emotional levels
        fear = emotions.get('fear', 0.3)
        doubt = emotions.get('doubt', 0.4)
        anger = emotions.get('anger', 0.2)
        compassion = emotions.get('compassion', 0.5)
        confidence = emotions.get('confidence', 0.7)
        
        # Responses get more intense as scene progresses
        intensity_multiplier = min(2.0, 1.0 + (scene_num * 0.2))
        
        # Scene-specific dramatic responses
        if scene_num == 1:  # Initial encounter
            return {
                "dialogue": "Another prisoner brought before me? What is his crime?",
                "thought": "This seems routine, but something feels... different about this man.",
                "action": "Looks up from administrative duties with mild interest",
                "emotional_shift": {"doubt": 0.1, "curiosity": 0.05}
            }
        
        elif scene_num == 2:  # Private questioning
            return {
                "dialogue": "Your kingdom is not of this world? What manner of king are you?",
                "thought": "His words are strange... there's something unsettling about his certainty.",
                "action": "Leans forward, studying Jesus intently, disturbed by his calm",
                "emotional_shift": {"doubt": 0.2, "fear": 0.1, "curiosity": 0.1}
            }
        
        elif scene_num == 3:  # Crowd pressure
            if fear > 0.5:
                return {
                    "dialogue": "You demand his death? But I have examined him and found no fault deserving death!",
                    "thought": "The crowd grows dangerous. If I don't satisfy them, there will be blood.",
                    "action": "Raises voice over the shouting crowd, gesturing for calm",
                    "emotional_shift": {"fear": 0.25, "anger": 0.15, "confidence": -0.1}
                }
            else:
                return {
                    "dialogue": "The crowd speaks with one voice. Yet justice requires evidence.",
                    "thought": "They want blood, but I see no crime worthy of death.",
                    "action": "Tries to maintain composure before the growing mob",
                    "emotional_shift": {"doubt": 0.2, "fear": 0.2}
                }
        
        elif scene_num == 4:  # Wife's warning
            return {
                "dialogue": "My wife warns me against this? She speaks of dreams and omens...",
                "thought": "First this strange prisoner, now my wife's prophetic dreams. What forces move here?",
                "action": "Pauses, deeply troubled, hand trembling slightly as he reads the message",
                "emotional_shift": {"fear": 0.3, "doubt": 0.25, "compassion": 0.2}
            }
        
        elif scene_num == 5:  # Scourging attempt
            return {
                "dialogue": "Perhaps if I have him scourged, it will satisfy your bloodlust!",
                "thought": "A compromise - pain without death. Surely this will appease them.",
                "action": "Orders the scourging with reluctance, hoping it ends here",
                "emotional_shift": {"desperation": 0.3, "compassion": -0.1, "fear": 0.2}
            }
        
        elif scene_num == 6:  # Caesar threat
            if fear > 0.8:
                return {
                    "dialogue": "You threaten to report me to Caesar?! You know not what you ask!",
                    "thought": "Caesar's wrath means death. But condemning innocence... what have I become?",
                    "action": "Voice cracks with desperation, backing away from the crowd",
                    "emotional_shift": {"fear": 0.3, "anger": 0.2, "confidence": -0.3}
                }
            else:
                return {
                    "dialogue": "Caesar? You would invoke the Emperor's name for this?",
                    "thought": "Now they play their final card. My career, my life - all hangs in balance.",
                    "action": "Face pale, realizes the trap has closed around him",
                    "emotional_shift": {"fear": 0.4, "despair": 0.3, "confidence": -0.2}
                }
        
        elif scene_num == 7:  # Final decision
            return {
                "dialogue": "I find no fault in him... yet you force my hand. The responsibility is yours!",
                "thought": "History will judge this moment. I am lost between duty and conscience.",
                "action": "Speaks with hollow voice, all authority drained from his bearing",
                "emotional_shift": {"despair": 0.4, "doubt": 0.2, "confidence": -0.4}
            }
        
        elif scene_num == 8:  # Washing hands
            return {
                "dialogue": "I wash my hands of this righteous man's blood! His death is upon your heads!",
                "thought": "If I cannot save him, at least let history know I tried to be just.",
                "action": "Washes hands ceremoniously, tears mixing with the water",
                "emotional_shift": {"compassion": 0.3, "guilt": 0.4, "fear": -0.1}
            }
        
        # Fallback for additional scenes
        else:
            return {
                "dialogue": "What have I done? The die is cast, but at what cost to my soul?",
                "thought": "I am forever changed by this day. There is no going back.",
                "action": "Stares at his hands, seeing invisible blood that will never wash clean",
                "emotional_shift": {"regret": 0.5, "despair": 0.3}
            }

def print_dramatic_emotional_visualization(journey_history):
    """Enhanced emotional visualization with drama and color"""
    
    print("\n" + "="*80)
    print("🎭 DRAMATIC EMOTIONAL JOURNEY - PONTIUS PILATE'S TRANSFORMATION")
    print("="*80)
    
    emotions = ['anger', 'doubt', 'fear', 'compassion', 'confidence']
    
    # Print header
    print(f"{'Event':<25} {'Anger':<8} {'Doubt':<8} {'Fear':<8} {'Comp':<8} {'Conf':<8}")
    print("-" * 80)
    
    for i, state in enumerate(journey_history):
        event_name = state['event'][:24]  # Truncate long names
        
        # Print values with visual indicators
        values = []
        for emotion in emotions:
            value = state[emotion]
            
            # Add visual intensity indicators
            if value >= 0.8:
                indicator = "🔥"
            elif value >= 0.6:
                indicator = "🌡️"
            elif value >= 0.4:
                indicator = "💧"
            elif value >= 0.2:
                indicator = "❄️"
            else:
                indicator = "🧊"
            
            values.append(f"{value:.2f}{indicator}")
        
        print(f"{event_name:<25} {values[0]:<8} {values[1]:<8} {values[2]:<8} {values[3]:<8} {values[4]:<8}")
    
    print("\n🎯 EMOTIONAL INTENSITY LEGEND:")
    print("🔥 = Extreme (0.8+)  🌡️ = High (0.6+)  💧 = Medium (0.4+)  ❄️ = Low (0.2+)  🧊 = Minimal (<0.2)")
    
    # Show dramatic changes
    print(f"\n📈 MOST DRAMATIC CHANGES:")
    if len(journey_history) > 1:
        initial = journey_history[0]
        final = journey_history[-1]
        
        changes = []
        for emotion in emotions:
            change = final[emotion] - initial[emotion]
            if abs(change) > 0.1:  # Only show significant changes
                direction = "📈" if change > 0 else "📉"
                changes.append(f"{emotion.upper()} {direction} {abs(change):.2f}")
        
        if changes:
            for change in changes:
                print(f"   {change}")
        else:
            print("   No dramatic changes detected")
    
    # Emotional peaks
    print(f"\n🏔️ EMOTIONAL PEAKS:")
    peak_emotions = {}
    for emotion in emotions:
        max_val = max(state[emotion] for state in journey_history)
        max_scene = next(state for state in journey_history if state[emotion] == max_val)
        if max_val > 0.6:  # Only show significant peaks
            peak_emotions[emotion] = (max_val, max_scene['event'])
    
    for emotion, (value, scene) in peak_emotions.items():
        print(f"   {emotion.upper()}: {value:.2f} during '{scene}'")

async def run_dramatic_experiment():
    """Run the dramatic emotional journey experiment"""
    
    print("🎭 DRAMATIC EMOTIONAL JOURNEY EXPERIMENT")
    print("🔥 Maximum Emotional Impact Version")
    print("=" * 70)
    
    # Check for LMStudio but don't require it
    print("🔌 LLM Connection Check:")
    lmstudio_llm = LMStudioLLM(endpoint="http://localhost:1234/v1")
    
    try:
        is_available = await lmstudio_llm.health_check()
        if is_available:
            # Quick test generation
            test_response = await lmstudio_llm.generate_response(
                "Test response. Respond with JSON containing: dialogue, thought, action, emotional_shift",
                temperature=0.7, max_tokens=200
            )
            
            # Try to parse
            json.loads(test_response.content)
            print("✅ LMStudio available and working! Using real LLM.")
            llm = lmstudio_llm
            using_real_llm = True
        else:
            raise Exception("Health check failed")
            
    except Exception as e:
        print(f"⚡ LMStudio not available ({e})")
        print("🎯 Using Enhanced Dramatic MockLLM for maximum emotional impact!")
        llm = DramaticMockLLM()
        using_real_llm = False
    
    print()
    
    # Create simulation engine
    engine = SimulationEngine(llm_provider=llm, max_concurrent=1)
    
    # Create Pilate with more dramatic starting state
    pilate = CharacterState(
        id="pilate_dramatic",
        name="Pontius Pilate",
        backstory={
            "origin": "Roman equestrian from Samnium", 
            "career": "Prefect of Judaea, ambitious but conflicted",
            "background": "A man of law facing the limits of earthly justice"
        },
        traits=["pragmatic", "politically-astute", "increasingly-desperate", "morally-conflicted"],
        values=["Roman law", "order", "career survival", "hidden justice"],
        fears=["rebellion", "Caesar's wrath", "moral judgment", "loss of control"],
        desires=["clear resolution", "political safety", "personal honor", "peace"],
        emotional_state=EmotionalState(
            anger=0.1,      # Starts calm
            doubt=0.2,      # Slight uncertainty
            fear=0.2,       # Minimal initial fear
            compassion=0.4, # Natural empathy
            confidence=0.8  # High initial confidence
        ),
        memory=CharacterMemory(),
        current_goal="Navigate this trial without political or personal disaster",
        internal_conflict="Roman duty vs. growing certainty of Jesus's innocence"
    )
    
    # More intense story progression
    dramatic_sequence = [
        {
            "event": "Initial Encounter",
            "situation": "Jesus is brought before you accused of claiming to be King of the Jews. The religious leaders watch expectantly.",
            "emphasis": "duty"
        },
        {
            "event": "Private Interrogation",
            "situation": "Alone with Jesus, you ask about his kingdom. He speaks of truth and says his kingdom is not of this world.",
            "emphasis": "doubt"
        },
        {
            "event": "Crowd Demands Blood",
            "situation": "The crowd grows violent, shouting 'Crucify him!' The religious leaders fuel their anger.",
            "emphasis": "fear"
        },
        {
            "event": "Wife's Prophetic Dream",
            "situation": "Your wife sends urgent word: 'Have nothing to do with this righteous man. I have suffered much in a dream because of him.'",
            "emphasis": "compassion"
        },
        {
            "event": "Scourging Compromise",
            "situation": "You try to appease the crowd by having Jesus scourged, hoping they'll accept this punishment instead of death.",
            "emphasis": "fear"
        },
        {
            "event": "Caesar Threat",
            "situation": "The crowd threatens: 'If you release this man, you are no friend of Caesar!' Your political survival is at stake.",
            "emphasis": "power"
        },
        {
            "event": "Final Decision",
            "situation": "Trapped between condemning an innocent man and risking everything, you must choose. There is no escape.",
            "emphasis": "neutral"
        },
        {
            "event": "Washing Hands",
            "situation": "Having delivered the verdict, you call for water and wash your hands before the crowd, declaring your innocence.",
            "emphasis": "compassion"
        }
    ]
    
    journey_history = []
    
    # Record initial state
    initial_state = {
        'event': 'Initial State',
        'anger': pilate.emotional_state.anger,
        'doubt': pilate.emotional_state.doubt,
        'fear': pilate.emotional_state.fear,
        'compassion': pilate.emotional_state.compassion,
        'confidence': pilate.emotional_state.confidence
    }
    journey_history.append(initial_state)
    
    print("🎬 BEGINNING DRAMATIC SEQUENCE")
    print("=" * 50)
    print(f"🎭 Character: {pilate.name}")
    print(f"⚡ Initial state: Confident administrator (Conf: {pilate.emotional_state.confidence:.2f})")
    print()
    
    # Run each scene
    for i, scene in enumerate(dramatic_sequence, 1):
        print(f"🎬 SCENE {i}: {scene['event']}")
        print(f"📍 Situation: {scene['situation']}")
        print()
        
        # Add memory from previous scene
        if i > 1:
            prev_memory = f"Scene {i-1}: {dramatic_sequence[i-2]['event']}"
            await pilate.memory.add_event(prev_memory)
        
        try:
            # Run simulation
            print("⏳ Generating response..." + (" (Real LLM)" if using_real_llm else " (Dramatic Mock)"))
            
            result = await engine.run_simulation(
                pilate,
                scene['situation'],
                emphasis=scene['emphasis'],
                temperature=0.9  # High temperature for variety
            )
            
            response = result['response']
            
            # Display results
            print(f"💬 Pilate: \"{response['dialogue']}\"")
            print(f"💭 Thought: {response['thought']}")
            print(f"🎭 Action: {response['action']}")
            
            # Show emotional changes
            if 'emotional_shift' in response and response['emotional_shift']:
                shifts = []
                for emotion, change in response['emotional_shift'].items():
                    if isinstance(change, (int, float)) and change != 0:
                        direction = "↗️" if change > 0 else "↘️"
                        shifts.append(f"{emotion} {direction} {abs(change):.2f}")
                
                if shifts:
                    print(f"📊 Emotional shifts: {' | '.join(shifts)}")
            
            # Record current state
            current_state = {
                'event': scene['event'],
                'anger': pilate.emotional_state.anger,
                'doubt': pilate.emotional_state.doubt,
                'fear': pilate.emotional_state.fear,
                'compassion': pilate.emotional_state.compassion,
                'confidence': pilate.emotional_state.confidence
            }
            journey_history.append(current_state)
            
            # Show current emotional temperature
            temp = pilate.emotional_state.modulate_temperature()
            print(f"🌡️ Emotional temperature: {temp:.2f}")
            print()
            
        except Exception as e:
            print(f"❌ Error in scene {i}: {e}")
            # Still record state
            error_state = {
                'event': f"{scene['event']} (Error)",
                'anger': pilate.emotional_state.anger,
                'doubt': pilate.emotional_state.doubt,
                'fear': pilate.emotional_state.fear,
                'compassion': pilate.emotional_state.compassion,
                'confidence': pilate.emotional_state.confidence
            }
            journey_history.append(error_state)
        
        # Pause for dramatic effect
        await asyncio.sleep(0.3)
    
    # Show dramatic visualization
    print_dramatic_emotional_visualization(journey_history)
    
    print(f"\n🎉 DRAMATIC JOURNEY COMPLETE")
    print("=" * 60)
    print(f"🎭 Transformation: From confident Roman administrator to tormented judge")
    print(f"📊 Total simulations: {llm.call_count}")
    print(f"🤖 LLM used: {'Real LMStudio' if using_real_llm else 'Enhanced Dramatic Mock'}")
    print(f"🎪 Final emotional state: Complex mix of guilt, compassion, and broken confidence")

if __name__ == "__main__":
    asyncio.run(run_dramatic_experiment())