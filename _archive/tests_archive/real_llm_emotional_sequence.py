"""
Real LLM Emotional Sequence Experiment
Watch how a character's emotional state evolves through story events using a real LLM
"""

import asyncio
import json
from character_simulation_engine_v2 import (
    CharacterState,
    EmotionalState,
    CharacterMemory,
    SimulationEngine,
    LMStudioLLM,
    MockLLM,
    LLMResponse,
)


class EmotionalJourney:
    """Tracks and visualizes emotional state changes"""

    def __init__(self):
        self.emotional_history = []
        self.event_history = []

    def record_state(self, event_name, character):
        """Record current emotional state"""
        state = {
            "event": event_name,
            "anger": character.emotional_state.anger,
            "doubt": character.emotional_state.doubt,
            "fear": character.emotional_state.fear,
            "compassion": character.emotional_state.compassion,
            "confidence": character.emotional_state.confidence,
        }
        self.emotional_history.append(state)

    def print_emotional_graph(self):
        """Print a simple text-based emotional progression graph"""
        print("\n📊 EMOTIONAL JOURNEY VISUALIZATION")
        print("=" * 80)

        emotions = ["anger", "doubt", "fear", "compassion", "confidence"]

        for i, state in enumerate(self.emotional_history):
            print(f"\n🎬 Event {i+1}: {state['event']}")

            for emotion in emotions:
                value = state[emotion]
                bar_length = int(value * 20)  # Scale to 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {emotion.capitalize():12} |{bar}| {value:.2f}")

            # Show biggest change from previous state
            if i > 0:
                prev_state = self.emotional_history[i - 1]
                biggest_change = ""
                max_change = 0

                for emotion in emotions:
                    change = state[emotion] - prev_state[emotion]
                    if abs(change) > max_change:
                        max_change = abs(change)
                        direction = "↑" if change > 0 else "↓"
                        biggest_change = f"{emotion} {direction} {abs(change):.2f}"

                if max_change > 0.05:  # Only show significant changes
                    print(f"  💫 Biggest change: {biggest_change}")


async def test_llm_connection():
    """Test connection to LMStudio and fallback to MockLLM if needed"""

    print("🔌 Testing LLM Connection...")
    print("-" * 40)

    # Try LMStudio first
    lmstudio_llm = LMStudioLLM(endpoint="http://localhost:1234/v1")

    try:
        print("⏳ Checking LMStudio availability...")
        is_available = await lmstudio_llm.health_check()

        if is_available:
            print("✅ LMStudio is available!")

            # Test a simple generation
            print("🧪 Testing LLM generation...")
            test_response = await lmstudio_llm.generate_response(
                "You are Pontius Pilate. Respond with structured JSON containing dialogue, thought, action, and emotional_shift.",
                temperature=0.7,
                max_tokens=300,
            )

            print(
                f"📝 Test response received (length: {len(test_response.content)} chars)"
            )

            # Try to parse it
            try:
                test_data = json.loads(test_response.content)
                print("✅ JSON parsing successful!")
                print(f"📋 Response keys: {list(test_data.keys())}")
                return lmstudio_llm, True
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parsing failed: {e}")
                print(
                    "📝 Raw response:",
                    (
                        test_response.content[:200] + "..."
                        if len(test_response.content) > 200
                        else test_response.content
                    ),
                )
                print("🔄 Falling back to enhanced MockLLM...")

        else:
            print("❌ LMStudio not available")

    except Exception as e:
        print(f"❌ LMStudio connection failed: {e}")

    print("🔄 Using Enhanced MockLLM with realistic responses...")

    # Enhanced MockLLM for more realistic behavior
    class SequenceMockLLM(MockLLM):
        def __init__(self):
            super().__init__()
            self.context_memory = []

        async def generate_response(self, prompt, temperature=0.7, max_tokens=500):
            self.call_count += 1
            await asyncio.sleep(0.1)  # Simulate network delay

            # Extract context from prompt
            current_emotions = self._extract_emotions(prompt)
            emphasis = self._extract_emphasis(prompt)
            situation = self._extract_situation(prompt)

            # Generate contextual response
            response_data = self._generate_contextual_response(
                current_emotions, emphasis, situation, temperature
            )

            # Store context for next time
            self.context_memory.append(
                {
                    "emotions": current_emotions,
                    "situation": situation,
                    "response": response_data,
                }
            )

            return LLMResponse(
                json.dumps(response_data),
                {"model": "sequence_mock", "temperature": temperature},
            )

        def _extract_emotions(self, prompt):
            emotions = {}
            for emotion in ["anger", "doubt", "fear", "compassion", "confidence"]:
                # Look for patterns like "Anger: 0.7/10"
                import re

                pattern = rf"{emotion.title()}: (\d+\.?\d*)"
                match = re.search(pattern, prompt)
                if match:
                    emotions[emotion] = float(match.group(1))
            return emotions

        def _extract_emphasis(self, prompt):
            emphases = ["power", "doubt", "fear", "compassion", "duty", "neutral"]
            for emphasis in emphases:
                if f"Emphasis: {emphasis}" in prompt:
                    return emphasis
            return "neutral"

        def _extract_situation(self, prompt):
            # Extract situation from prompt
            lines = prompt.split("\n")
            for line in lines:
                if line.startswith("Situation:"):
                    return line.replace("Situation:", "").strip()
            return "unknown situation"

        def _generate_contextual_response(
            self, emotions, emphasis, situation, temperature
        ):
            # More sophisticated response generation based on context

            fear_level = emotions.get("fear", 0.5)
            doubt_level = emotions.get("doubt", 0.5)
            anger_level = emotions.get("anger", 0.5)
            compassion_level = emotions.get("compassion", 0.5)

            # Context-aware responses
            if "trial" in situation.lower() or "judge" in situation.lower():
                if emphasis == "power" and anger_level > 0.7:
                    return {
                        "dialogue": "I have heard enough! The authority of Rome will not be questioned!",
                        "thought": "These people test my patience. Time to assert dominance.",
                        "action": "Stands with imperial bearing, voice echoing through the hall",
                        "emotional_shift": {
                            "anger": 0.1,
                            "confidence": 0.15,
                            "fear": -0.05,
                        },
                    }
                elif doubt_level > 0.8:
                    return {
                        "dialogue": "What is truth? In all my years, I have never encountered such certainty in the face of death...",
                        "thought": "This prisoner disturbs me. His words echo in my mind.",
                        "action": "Paces slowly, avoiding direct eye contact, lost in contemplation",
                        "emotional_shift": {
                            "doubt": 0.1,
                            "fear": 0.05,
                            "compassion": 0.05,
                        },
                    }
                elif fear_level > 0.8:
                    return {
                        "dialogue": "The crowd grows dangerous. If I do not act swiftly, there will be chaos in the streets.",
                        "thought": "Caesar will have my head if there's an uprising. But this man...",
                        "action": "Glances nervously between the prisoner and the door, wringing hands",
                        "emotional_shift": {
                            "fear": 0.15,
                            "doubt": 0.1,
                            "confidence": -0.1,
                        },
                    }
                elif compassion_level > 0.7:
                    return {
                        "dialogue": "I find no fault in this man. Surely there must be another path...",
                        "thought": "How can I condemn innocence? Yet how can I ignore the law?",
                        "action": "Looks upon Jesus with troubled eyes, speaks quietly",
                        "emotional_shift": {
                            "compassion": 0.1,
                            "doubt": 0.1,
                            "anger": -0.05,
                        },
                    }

            # Default responses based on emotional state
            if anger_level > 0.6:
                return {
                    "dialogue": "My patience wears thin! Speak plainly or face the consequences!",
                    "thought": "These games of words and riddles infuriate me.",
                    "action": "Clenches fists, voice rising with frustration",
                    "emotional_shift": {"anger": 0.05, "confidence": 0.1},
                }
            elif fear_level > 0.7:
                return {
                    "dialogue": "Every choice I make leads deeper into darkness...",
                    "thought": "I am trapped between impossible alternatives.",
                    "action": "Stares into the distance, shoulders heavy with burden",
                    "emotional_shift": {"fear": 0.1, "doubt": 0.05},
                }
            else:
                return {
                    "dialogue": "I stand at the crossroads of history, and I know not which path leads to light.",
                    "thought": "The weight of this moment will echo through eternity.",
                    "action": "Pauses in contemplation, feeling the gravity of the decision",
                    "emotional_shift": {"doubt": 0.05, "compassion": 0.05},
                }

    return SequenceMockLLM(), False


async def run_emotional_sequence_experiment():
    """Run the main emotional sequence experiment"""

    print("🎭 REAL LLM EMOTIONAL SEQUENCE EXPERIMENT")
    print("=" * 60)
    print("Watch Pontius Pilate's emotional journey through the trial of Jesus")
    print()

    # Test LLM connection
    llm, is_real_llm = await test_llm_connection()

    if is_real_llm:
        print("🤖 Using Real LLM (LMStudio)")
    else:
        print("🎯 Using Enhanced MockLLM")

    print()

    # Create simulation engine
    engine = SimulationEngine(
        llm_provider=llm, max_concurrent=1  # Sequential for better story flow
    )

    # Create Pontius Pilate with initial emotional state
    pilate = CharacterState(
        id="pilate_sequence",
        name="Pontius Pilate",
        backstory={
            "origin": "Roman equestrian from Samnium",
            "career": "Prefect of Judaea (26-36 CE)",
            "background": "Experienced administrator facing unprecedented religious and political crisis",
        },
        traits=[
            "pragmatic",
            "politically-astute",
            "duty-bound",
            "increasingly-conflicted",
        ],
        values=["Roman law", "order", "imperial loyalty", "self-preservation"],
        fears=["rebellion", "imperial disfavor", "losing control", "moral judgment"],
        desires=["peace in province", "clear guidance", "honorable resolution"],
        emotional_state=EmotionalState(
            anger=0.2,  # Slightly irritated by the situation
            doubt=0.4,  # Some uncertainty about the case
            fear=0.3,  # Concerned about potential unrest
            compassion=0.5,  # Natural human empathy
            confidence=0.7,  # Initially confident in his authority
        ),
        memory=CharacterMemory(),
        current_goal="Navigate this trial without causing political disaster",
        internal_conflict="Roman duty vs. growing sense that this man is innocent",
    )

    # Create emotional journey tracker
    journey = EmotionalJourney()

    # Story sequence - each event builds on the previous
    story_sequence = [
        {
            "event": "Initial Encounter",
            "situation": "Jesus is brought before you, accused of claiming to be King of the Jews. The religious leaders demand judgment.",
            "emphasis": "duty",
            "expected_shift": "Slight increase in doubt as he realizes this isn't ordinary",
        },
        {
            "event": "Private Questioning",
            "situation": "Alone with Jesus, you ask about his kingdom. He speaks of truth and otherworldly realms.",
            "emphasis": "doubt",
            "expected_shift": "Major doubt increase, some fear as Jesus seems otherworldly",
        },
        {
            "event": "Crowd Pressure",
            "situation": "You return to face an increasingly agitated crowd demanding crucifixion. The religious leaders are relentless.",
            "emphasis": "fear",
            "expected_shift": "Fear spikes, anger rises at being pressured",
        },
        {
            "event": "Wife's Warning",
            "situation": "Your wife sends urgent word: 'Have nothing to do with this righteous man. I suffered much in a dream because of him.'",
            "emphasis": "compassion",
            "expected_shift": "Compassion and doubt increase, fear of supernatural",
        },
        {
            "event": "Scourging Attempt",
            "situation": "You try a compromise - have Jesus scourged and released. But the crowd only grows more violent in their demands.",
            "emphasis": "fear",
            "expected_shift": "Fear peaks, doubt about own authority, desperation",
        },
        {
            "event": "Final Confrontation",
            "situation": "The crowd threatens to report you to Caesar if you release Jesus. Your political survival is at stake.",
            "emphasis": "power",
            "expected_shift": "Anger at being cornered, fear of Caesar, confidence shattered",
        },
        {
            "event": "The Decision",
            "situation": "You must choose: condemn an innocent man or risk your career and possibly your life. There is no escape.",
            "emphasis": "neutral",
            "expected_shift": "All emotions in turmoil, final internal collapse",
        },
        {
            "event": "Washing Hands",
            "situation": "Having made your choice, you call for water to wash your hands before the crowd, declaring yourself innocent of this blood.",
            "emphasis": "compassion",
            "expected_shift": "Attempt to regain some compassion/peace, but doubt remains",
        },
    ]

    print("🎬 BEGINNING EMOTIONAL SEQUENCE")
    print("=" * 60)

    # Record initial state
    journey.record_state("Initial State", pilate)
    print("📍 Starting emotional state:")
    print(
        f"   Anger: {pilate.emotional_state.anger:.2f} | Doubt: {pilate.emotional_state.doubt:.2f} | Fear: {pilate.emotional_state.fear:.2f}"
    )
    print(
        f"   Compassion: {pilate.emotional_state.compassion:.2f} | Confidence: {pilate.emotional_state.confidence:.2f}"
    )

    # Run through each story event
    for i, scene in enumerate(story_sequence, 1):
        print("\n" + "─" * 60)
        print(f"🎭 SCENE {i}: {scene['event']}")
        print(f"📝 Situation: {scene['situation']}")
        print(f"🎯 Emphasis: {scene['emphasis']}")
        print(f"💭 Expected: {scene['expected_shift']}")
        print()

        # Add previous scene to memory
        if i > 1:
            prev_scene = story_sequence[i - 2]
            memory_entry = f"Scene {i-1}: {prev_scene['event']} - felt increasingly {prev_scene['emphasis']}"
            await pilate.memory.add_event(memory_entry)

        # Run simulation
        print(
            "⏳ Running simulation with real LLM..."
            if is_real_llm
            else "⏳ Running simulation..."
        )

        try:
            result = await engine.run_simulation(
                pilate,
                scene["situation"],
                emphasis=scene["emphasis"],
                temperature=0.8,  # Higher temperature for more variation
            )

            response = result["response"]

            print(f"💬 Pilate: \"{response['dialogue']}\"")
            print(f"🤔 Thought: {response['thought']}")
            print(f"🎭 Action: {response['action']}")

            # Show emotional changes
            if "emotional_shift" in response and response["emotional_shift"]:
                shifts = response["emotional_shift"]
                shift_text = []
                for emotion, change in shifts.items():
                    if isinstance(change, (int, float)) and change != 0:
                        direction = "↑" if change > 0 else "↓"
                        shift_text.append(f"{emotion} {direction}{abs(change):.2f}")

                if shift_text:
                    print(f"📊 Emotional shifts: {', '.join(shift_text)}")

            # Record state after this event
            journey.record_state(scene["event"], pilate)

            # Show current emotional state
            emotions = pilate.emotional_state
            print(
                f"😰 Current emotions: A:{emotions.anger:.2f} D:{emotions.doubt:.2f} F:{emotions.fear:.2f} C:{emotions.compassion:.2f} Conf:{emotions.confidence:.2f}"
            )

            # Show temperature modulation
            temp = emotions.modulate_temperature()
            print(f"🌡️  Emotional temperature: {temp:.2f}")

        except Exception as e:
            print(f"❌ Error in scene {i}: {e}")
            # Record state anyway
            journey.record_state(f"{scene['event']} (Error)", pilate)

        print()

        # Pause for dramatic effect
        await asyncio.sleep(0.5)

    # Show final emotional journey visualization
    journey.print_emotional_graph()

    print("\n🎉 EMOTIONAL SEQUENCE COMPLETE")
    print("=" * 60)
    print(f"📊 Total LLM calls: {llm.call_count}")
    print("🎭 Character arc: From confident administrator to tormented judge")
    print("📈 Biggest emotional changes:")

    # Calculate biggest changes
    initial = journey.emotional_history[0]
    final = journey.emotional_history[-1]

    for emotion in ["anger", "doubt", "fear", "compassion", "confidence"]:
        change = final[emotion] - initial[emotion]
        direction = "increased" if change > 0 else "decreased"
        print(f"   {emotion.capitalize()}: {direction} by {abs(change):.2f}")


if __name__ == "__main__":
    asyncio.run(run_emotional_sequence_experiment())
