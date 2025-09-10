"""
Interactive Character Simulation Playground
Build and test characters interactively
"""

import asyncio
import json
from character_simulation_engine_v2 import (
    CharacterState, EmotionalState, CharacterMemory, 
    SimulationEngine, MockLLM, LLMResponse
)

class InteractiveMockLLM(MockLLM):
    """MockLLM with very diverse responses for experimentation"""
    
    def __init__(self):
        super().__init__()
        
        # Different response templates based on context
        self.response_templates = {
            "power_high_anger": {
                "dialogue": "Enough! You dare question my authority? I'll show you the power of Rome!",
                "thought": "They test my limits. Time to remind them who rules here.",
                "action": "Slams fist on table, rises with imperial bearing",
                "emotional_shift": {"anger": 0.1, "confidence": 0.2}
            },
            "doubt_high_doubt": {
                "dialogue": "What is truth? Your words confound me, prisoner...",
                "thought": "Nothing is certain anymore. This man speaks mysteries.",
                "action": "Paces nervously, avoiding eye contact",
                "emotional_shift": {"doubt": 0.2, "fear": 0.1}
            },
            "fear_high_fear": {
                "dialogue": "The mob grows restless! If I don't act, Rome will have my head!",
                "thought": "Every choice leads to my destruction. I'm trapped.",
                "action": "Glances toward the door, wringing hands anxiously",
                "emotional_shift": {"fear": 0.2, "confidence": -0.1}
            },
            "compassion_high_compassion": {
                "dialogue": "I find no fault in this man. Surely innocence deserves mercy?",
                "thought": "How can I condemn one who has done no wrong?",
                "action": "Looks upon the prisoner with troubled compassion",
                "emotional_shift": {"compassion": 0.1, "doubt": 0.1}
            },
            "duty_confident": {
                "dialogue": "Roman law is clear. I must follow procedure above all else.",
                "thought": "My duty to Caesar outweighs personal feelings.",
                "action": "Straightens, speaks with official authority",
                "emotional_shift": {"confidence": 0.1, "doubt": -0.1}
            },
            "washing_hands": {
                "dialogue": "I wash my hands of this man's blood! See to it yourselves!",
                "thought": "If I cannot prevent this, at least I can absolve myself.",
                "action": "Calls for water, ceremoniously washes hands before crowd",
                "emotional_shift": {"compassion": 0.1, "fear": -0.1}
            }
        }
    
    async def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500):
        self.call_count += 1
        await asyncio.sleep(0.02)  # Very fast for interactivity
        
        # Parse emotional state and emphasis from prompt
        emotions = self._parse_emotions(prompt)
        emphasis = self._parse_emphasis(prompt)
        
        # Select best response template
        template_key = self._select_template(emotions, emphasis, prompt)
        template = self.response_templates.get(template_key, self._default_response(emotions, emphasis))
        
        # Add temperature-based variation
        if temperature > 1.0:
            # High temperature - add more dramatic flair
            template = self._add_dramatic_flair(template)
        elif temperature < 0.5:
            # Low temperature - more reserved
            template = self._make_more_reserved(template)
        
        content = json.dumps(template)
        return LLMResponse(content, {"model": "interactive_mock", "temperature": temperature})
    
    def _parse_emotions(self, prompt):
        emotions = {}
        for emotion in ['anger', 'doubt', 'fear', 'compassion', 'confidence']:
            if f"{emotion.title()}: 0.9" in prompt or f"{emotion.title()}: 0.8" in prompt:
                emotions[emotion] = 'high'
            elif f"{emotion.title()}: 0.1" in prompt or f"{emotion.title()}: 0.2" in prompt:
                emotions[emotion] = 'low'
            else:
                emotions[emotion] = 'medium'
        return emotions
    
    def _parse_emphasis(self, prompt):
        emphasis_markers = {
            "power": "power",
            "doubt": "doubt", 
            "fear": "fear",
            "compassion": "compassion",
            "duty": "duty",
            "wash": "washing_hands",
            "clean": "washing_hands"
        }
        
        for marker, emphasis in emphasis_markers.items():
            if marker in prompt.lower():
                return emphasis
        return "neutral"
    
    def _select_template(self, emotions, emphasis, prompt):
        # Combine emphasis and dominant emotion
        if emphasis == "power" and emotions.get('anger') == 'high':
            return "power_high_anger"
        elif emphasis == "doubt" or emotions.get('doubt') == 'high':
            return "doubt_high_doubt"
        elif emphasis == "fear" or emotions.get('fear') == 'high':
            return "fear_high_fear"
        elif emphasis == "compassion" or emotions.get('compassion') == 'high':
            return "compassion_high_compassion"
        elif emphasis == "duty" and emotions.get('confidence') != 'low':
            return "duty_confident"
        elif emphasis == "washing_hands":
            return "washing_hands"
        
        return "doubt_high_doubt"  # Default fallback
    
    def _default_response(self, emotions, emphasis):
        return {
            "dialogue": "I stand at the crossroads of history, and every path leads to darkness.",
            "thought": "The weight of this decision will follow me to my grave.",
            "action": "Stares into the distance, lost in contemplation",
            "emotional_shift": {"doubt": 0.1, "fear": 0.05}
        }
    
    def _add_dramatic_flair(self, template):
        # Add intensity for high temperature
        template = template.copy()
        template["dialogue"] = template["dialogue"].upper() if len(template["dialogue"]) < 50 else template["dialogue"] + "!"
        template["action"] = template["action"].replace(".", " dramatically.")
        return template
    
    def _make_more_reserved(self, template):
        # Make more subdued for low temperature  
        template = template.copy()
        template["dialogue"] = template["dialogue"].replace("!", ".")
        template["action"] = template["action"].replace(" dramatically", "")
        return template

async def character_builder():
    """Interactive character builder and tester"""
    
    print("🏗️  CHARACTER SIMULATION PLAYGROUND")
    print("=" * 50)
    print("Build custom characters and test different scenarios!")
    print()
    
    # Initialize engine
    llm = InteractiveMockLLM()
    engine = SimulationEngine(llm_provider=llm, max_concurrent=3)
    
    # Character templates
    templates = {
        "pilate": {
            "name": "Pontius Pilate",
            "backstory": {"origin": "Roman equestrian", "career": "Prefect of Judaea"},
            "traits": ["pragmatic", "conflicted", "ambitious"],
            "values": ["order", "duty", "survival"],
            "fears": ["rebellion", "disgrace"],
            "desires": ["peace", "advancement"],
            "goal": "Navigate trial without disaster"
        },
        "caesar": {
            "name": "Julius Caesar", 
            "backstory": {"origin": "Patrician family", "career": "Roman general and statesman"},
            "traits": ["ambitious", "charismatic", "ruthless"],
            "values": ["glory", "power", "Rome"],
            "fears": ["weakness", "obscurity"],
            "desires": ["immortality", "absolute power"],
            "goal": "Become the greatest Roman in history"
        },
        "hamlet": {
            "name": "Prince Hamlet",
            "backstory": {"origin": "Danish royalty", "career": "Prince of Denmark"}, 
            "traits": ["melancholic", "philosophical", "indecisive"],
            "values": ["truth", "justice", "honor"],
            "fears": ["inaction", "damnation"],
            "desires": ["revenge", "understanding"],
            "goal": "Avenge father's murder"
        }
    }
    
    print("Choose a character template or build custom:")
    print("1. Pontius Pilate (conflicted administrator)")
    print("2. Julius Caesar (ambitious general)")
    print("3. Prince Hamlet (melancholic prince)")
    print("4. Custom character")
    
    # For automation, let's use Pilate
    choice = "1"
    print(f"Selected: {choice}")
    
    if choice in ["1", "2", "3"]:
        template_key = ["pilate", "caesar", "hamlet"][int(choice) - 1]
        template = templates[template_key]
        
        character = CharacterState(
            id=template_key,
            name=template["name"],
            backstory=template["backstory"],
            traits=template["traits"],
            values=template["values"],
            fears=template["fears"], 
            desires=template["desires"],
            emotional_state=EmotionalState(
                anger=0.4,
                doubt=0.7,
                fear=0.5,
                compassion=0.3,
                confidence=0.4
            ),
            memory=CharacterMemory(),
            current_goal=template["goal"]
        )
    
    print(f"\n🎭 Character: {character.name}")
    print(f"📖 Background: {character.backstory}")
    print(f"🎯 Goal: {character.current_goal}")
    
    # Test scenarios
    scenarios = [
        "You stand at a crossroads. A crowd demands action. What do you choose?",
        "Your advisors urge caution, but your heart calls for bold action. Decide.",
        "An enemy offers you a deal that could save lives but cost you honor. Your response?",
        "The people you serve question your judgment. How do you respond?",
        "You must choose between personal desire and duty to others. What guides you?"
    ]
    
    print(f"\n🎬 Testing scenarios with {character.name}:")
    print("-" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎭 Scenario {i}: {scenario}")
        
        # Test different emotional emphases
        emphases = ["power", "doubt", "fear", "compassion"]
        
        for emphasis in emphases:
            result = await engine.run_simulation(
                character,
                scenario,
                emphasis=emphasis,
                temperature=0.8
            )
            
            response = result['response']
            print(f"  🎯 {emphasis.upper()}: \"{response['dialogue']}\"")
        
        # Show how emotions evolved
        emotions = character.emotional_state
        print(f"  😰 Current emotions: Doubt={emotions.doubt:.2f}, Fear={emotions.fear:.2f}, Anger={emotions.anger:.2f}")
    
    # Extreme emotional states test
    print("\n🔥 EXTREME EMOTIONAL STATES TEST")
    print("-" * 50)
    
    extreme_states = [
        ("Enraged", EmotionalState(anger=0.95, doubt=0.2, fear=0.3, compassion=0.1, confidence=0.8)),
        ("Terrified", EmotionalState(anger=0.2, doubt=0.8, fear=0.95, compassion=0.3, confidence=0.1)),
        ("Utterly Lost", EmotionalState(anger=0.3, doubt=0.95, fear=0.7, compassion=0.2, confidence=0.1)),
        ("Deeply Moved", EmotionalState(anger=0.1, doubt=0.4, fear=0.3, compassion=0.95, confidence=0.6))
    ]
    
    test_situation = "The moment of ultimate decision has arrived. All eyes are upon you."
    
    for state_name, emotional_state in extreme_states:
        print(f"\n😤 {state_name} {character.name}:")
        
        # Temporarily change emotional state
        original_state = character.emotional_state
        character.emotional_state = emotional_state
        
        result = await engine.run_simulation(
            character,
            test_situation,
            emphasis="neutral",
            temperature=0.9  # High temp for variety
        )
        
        response = result['response']
        print(f"💬 \"{response['dialogue']}\"")
        print(f"🎭 {response['action']}")
        print(f"🌡️  Emotional temp: {emotional_state.modulate_temperature():.2f}")
        
        # Restore original state
        character.emotional_state = original_state
    
    print("\n🎉 Playground session complete!")
    print(f"📊 Total simulations: {llm.call_count}")
    print("\n💡 Try editing the emotional states, scenarios, or character traits to explore different behaviors!")

if __name__ == "__main__":
    asyncio.run(character_builder())