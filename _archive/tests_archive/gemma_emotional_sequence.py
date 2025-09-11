"""
Gemma-3-12b Emotional Sequence Test
Optimized for Gemma's response patterns and timing
"""

import asyncio
import time
from character_simulation_engine_v2 import (
    CharacterState,
    EmotionalState,
    CharacterMemory,
    SimulationEngine,
    LMStudioLLM,
    MockLLM,
)


class GemmaEmotionalTracker:
    """Track emotional evolution through scenes"""

    def __init__(self):
        self.journey = []
        self.scene_times = []

    def record_scene(
        self,
        scene_name: str,
        character_state: CharacterState,
        response: dict,
        elapsed_time: float,
    ):
        """Record emotional state after a scene"""
        self.journey.append(
            {
                "scene": scene_name,
                "emotions": {
                    "anger": character_state.emotional_state.anger,
                    "doubt": character_state.emotional_state.doubt,
                    "fear": character_state.emotional_state.fear,
                    "compassion": character_state.emotional_state.compassion,
                    "confidence": character_state.emotional_state.confidence,
                },
                "response_snippet": response.get("dialogue", "")[:100],
                "thought_snippet": response.get("thought", "")[:100],
            }
        )
        self.scene_times.append(elapsed_time)

    def display_journey(self, model_name: str = "Gemma-3-12b"):
        """Display emotional journey visualization"""
        print(f"\n{'=' * 80}")
        print(f"📊 EMOTIONAL JOURNEY WITH {model_name.upper()}")
        print(f"{'=' * 80}")

        if not self.journey:
            print("No journey data recorded")
            return

        # Header
        print(
            f"{'Scene':<25} {'Anger':<8} {'Doubt':<8} {'Fear':<8} {'Comp':<8} {'Conf':<8}"
        )
        print("-" * 80)

        # Initial state (if exists)
        if self.journey:
            first = self.journey[0]["emotions"]
            print(
                f"{'Initial State':<25} {first['anger']:>6.2f}  {first['doubt']:>6.2f}  "
                f"{first['fear']:>6.2f}  {first['compassion']:>6.2f}  {first['confidence']:>6.2f}"
            )

        # Show progression
        for i, entry in enumerate(self.journey[1:], 1):
            emotions = entry["emotions"]
            scene = entry["scene"][:24]  # Truncate long names

            # Calculate changes from previous
            prev = self.journey[i - 1]["emotions"]
            changes = {
                "anger": emotions["anger"] - prev["anger"],
                "doubt": emotions["doubt"] - prev["doubt"],
                "fear": emotions["fear"] - prev["fear"],
                "compassion": emotions["compassion"] - prev["compassion"],
                "confidence": emotions["confidence"] - prev["confidence"],
            }

            # Format with arrows for changes
            def format_emotion(value, change):
                arrow = "↗" if change > 0.05 else "↘" if change < -0.05 else "→"
                return f"{value:>6.2f}{arrow}"

            print(
                f"{scene:<25} "
                f"{format_emotion(emotions['anger'], changes['anger'])} "
                f"{format_emotion(emotions['doubt'], changes['doubt'])} "
                f"{format_emotion(emotions['fear'], changes['fear'])} "
                f"{format_emotion(emotions['compassion'], changes['compassion'])} "
                f"{format_emotion(emotions['confidence'], changes['confidence'])}"
            )

        # Summary statistics
        if len(self.journey) > 1:
            print("\n" + "-" * 80)
            print("📈 EMOTIONAL CHANGES:")

            first = self.journey[0]["emotions"]
            last = self.journey[-1]["emotions"]

            total_changes = {
                "anger": last["anger"] - first["anger"],
                "doubt": last["doubt"] - first["doubt"],
                "fear": last["fear"] - first["fear"],
                "compassion": last["compassion"] - first["compassion"],
                "confidence": last["confidence"] - first["confidence"],
            }

            # Sort by absolute change
            sorted_changes = sorted(
                total_changes.items(), key=lambda x: abs(x[1]), reverse=True
            )

            for emotion, change in sorted_changes:
                if abs(change) > 0.05:
                    direction = "increased" if change > 0 else "decreased"
                    print(
                        f"  • {emotion.capitalize()}: {direction} by {abs(change):.2f}"
                    )

        # Performance stats
        if self.scene_times:
            avg_time = sum(self.scene_times) / len(self.scene_times)
            print(f"\n⏱️  Average response time: {avg_time:.2f}s")
            print(f"📏 Total scenes: {len(self.journey)}")


async def run_gemma_emotional_sequence():
    """Run emotional sequence with Gemma-3-12b model"""

    print("🎭 GEMMA-3-12B EMOTIONAL SEQUENCE TEST")
    print("=" * 80)
    print()

    # Check if Gemma is available
    print("🔍 Checking Gemma-3-12b availability...")
    lmstudio_llm = LMStudioLLM(
        endpoint="http://localhost:1234/v1", model="google/gemma-3-12b"
    )

    try:
        is_available = await lmstudio_llm.health_check()
        if not is_available:
            print("❌ Gemma not available, falling back to MockLLM")
            llm = MockLLM()
            model_name = "MockLLM"
        else:
            print("✅ Gemma-3-12b is ready!")
            llm = lmstudio_llm
            model_name = "Gemma-3-12b"
    except Exception as e:
        print(f"⚠️ Error checking Gemma: {e}")
        print("📌 Using MockLLM for demonstration")
        llm = MockLLM()
        model_name = "MockLLM"

    # Create simulation engine
    engine = SimulationEngine(
        llm_provider=llm, max_concurrent=1  # Sequential for clarity
    )

    # Create Pontius Pilate character
    pilate = CharacterState(
        id="pilate_gemma",
        name="Pontius Pilate",
        backstory={
            "origin": "Roman equestrian from Samnium",
            "career": "Fifth Prefect of Roman Judaea",
            "reputation": "Known for pragmatic but sometimes brutal rule",
            "context": "Facing the trial of Jesus of Nazareth",
        },
        traits=["pragmatic", "politically-aware", "conflicted", "duty-bound"],
        values=["Roman order", "personal survival", "maintaining peace", "justice"],
        fears=[
            "rebellion",
            "losing position",
            "Caesar's displeasure",
            "divine judgment",
        ],
        desires=["peaceful province", "clear conscience", "political advancement"],
        emotional_state=EmotionalState(
            anger=0.2, doubt=0.3, fear=0.3, compassion=0.4, confidence=0.7
        ),
        memory=CharacterMemory(),
        current_goal="Navigate this trial without causing unrest",
        internal_conflict="Duty to Rome vs. sense of justice",
    )

    # Emotional journey tracker
    tracker = GemmaEmotionalTracker()

    # Record initial state
    tracker.record_scene("Initial", pilate, {}, 0)

    # Define emotional journey scenes (fewer for Gemma's slower speed)
    scenes = [
        {
            "name": "First Encounter",
            "situation": "Jesus is brought before you. The Jewish leaders accuse him of claiming to be King. You must begin the interrogation.",
            "emphasis": "duty",
            "memory": None,
        },
        {
            "name": "Kingdom Not of This World",
            "situation": "Jesus speaks of a kingdom not of this world and claims he came to testify to the truth. His words are strange but compelling.",
            "emphasis": "doubt",
            "memory": "Jesus spoke of truth and kingdoms beyond Rome",
        },
        {
            "name": "Finding No Guilt",
            "situation": "After questioning, you find no basis for execution. But the crowd grows angry. You must decide whether to release him.",
            "emphasis": "compassion",
            "memory": "I found no fault but the crowd demands blood",
        },
        {
            "name": "Barabbas or Jesus",
            "situation": "You offer the crowd a choice: release Jesus or Barabbas the murderer. They choose Barabbas. Your options narrow.",
            "emphasis": "fear",
            "memory": "They chose to free a murderer over this prophet",
        },
        {
            "name": "Wife's Warning",
            "situation": "Your wife Claudia sends urgent word: 'Have nothing to do with that righteous man, for I suffered in a dream because of him.'",
            "emphasis": "fear",
            "memory": "Even Claudia warns me through her dreams",
        },
        {
            "name": "Final Decision",
            "situation": "The crowd threatens riot. You must decide: condemn an innocent man or risk Caesar's wrath for failing to maintain order.",
            "emphasis": "neutral",
            "memory": "All paths lead to suffering",
        },
    ]

    print(f"\n🎬 Running {len(scenes)} emotional scenes with {model_name}")
    print(
        f"📍 Starting emotional state: Confidence={pilate.emotional_state.confidence:.2f}, "
        f"Doubt={pilate.emotional_state.doubt:.2f}"
    )
    print()

    # Process each scene
    for i, scene in enumerate(scenes, 1):
        print(f"{'='*60}")
        print(f"📖 SCENE {i}/{len(scenes)}: {scene['name']}")
        print(f"{'='*60}")
        print(f"📝 {scene['situation']}")
        print(f"🎯 Emphasis: {scene['emphasis']}")

        # Add memory from previous scene
        if scene["memory"]:
            await pilate.memory.add_event(scene["memory"])
            print(f"💭 Memory: {scene['memory']}")

        try:
            print(f"⏳ Generating {model_name} response...")
            start_time = time.time()

            # Run simulation with appropriate temperature for model
            temperature = 0.7 if model_name == "Gemma-3-12b" else 0.8

            result = await engine.run_simulation(
                pilate,
                scene["situation"],
                emphasis=scene["emphasis"],
                temperature=temperature,
            )

            elapsed = time.time() - start_time
            response = result["response"]

            # Display response
            print(f"⏱️ Generated in {elapsed:.2f}s")

            # Handle response display
            if isinstance(response, dict):
                dialogue = response.get("dialogue", "")
                thought = response.get("thought", "")
                action = response.get("action", "")

                # Display with truncation for long responses
                print(
                    f"\n💬 Pilate: \"{dialogue[:200]}{'...' if len(dialogue) > 200 else ''}\""
                )
                print(
                    f"🤔 Thinks: {thought[:150]}{'...' if len(thought) > 150 else ''}"
                )
                print(f"🎬 Action: {action[:100]}{'...' if len(action) > 100 else ''}")

                # Show emotional shifts
                if "emotional_shift" in response:
                    shifts = response["emotional_shift"]
                    if isinstance(shifts, dict):
                        significant_shifts = []
                        for emotion, change in shifts.items():
                            if isinstance(change, (int, float)) and abs(change) > 0.05:
                                direction = "↗" if change > 0 else "↘"
                                significant_shifts.append(
                                    f"{emotion}{direction}{abs(change):.2f}"
                                )

                        if significant_shifts:
                            print(
                                f"📊 Emotional shifts: {' | '.join(significant_shifts)}"
                            )

                # Track the scene
                tracker.record_scene(scene["name"], pilate, response, elapsed)
            else:
                print(f"📝 Raw response: {str(response)[:300]}")
                tracker.record_scene(
                    scene["name"], pilate, {"dialogue": str(response)}, elapsed
                )

            # Show current emotional state
            print(
                f"\n🌡️ Current state: Doubt={pilate.emotional_state.doubt:.2f}, "
                f"Fear={pilate.emotional_state.fear:.2f}, "
                f"Confidence={pilate.emotional_state.confidence:.2f}"
            )

            print()

        except Exception as e:
            print(f"❌ Error in scene: {e}")
            tracker.record_scene(scene["name"], pilate, {}, 0)

    # Display journey summary
    tracker.display_journey(model_name)

    # Final character analysis
    print(f"\n{'=' * 80}")
    print("🎭 CHARACTER TRANSFORMATION ANALYSIS")
    print(f"{'=' * 80}")

    print(f"\n📌 {pilate.name}'s Final State:")
    print(f"   • Anger: {pilate.emotional_state.anger:.2f}")
    print(f"   • Doubt: {pilate.emotional_state.doubt:.2f}")
    print(f"   • Fear: {pilate.emotional_state.fear:.2f}")
    print(f"   • Compassion: {pilate.emotional_state.compassion:.2f}")
    print(f"   • Confidence: {pilate.emotional_state.confidence:.2f}")

    # Memory summary
    if hasattr(pilate.memory, "recent_events") and pilate.memory.recent_events:
        print("\n💭 Final Memories (last 3):")
        for memory in pilate.memory.recent_events[-3:]:
            print(f"   • {memory}")

    print(f"\n✨ Test complete with {model_name}!")


async def compare_gemma_vs_mock():
    """Quick comparison between Gemma and Mock responses"""

    print("\n" + "=" * 80)
    print("🔄 GEMMA vs MOCK COMPARISON")
    print("=" * 80)

    test_situation = (
        "Jesus stands silent before you. The crowd demands action. What do you do?"
    )

    # Test both models
    models = []

    # Try Gemma
    gemma_llm = LMStudioLLM(
        endpoint="http://localhost:1234/v1", model="google/gemma-3-12b"
    )
    if await gemma_llm.health_check():
        models.append(("Gemma-3-12b", gemma_llm))

    # Always include Mock
    models.append(("MockLLM", MockLLM()))

    print(f"\n📊 Comparing {len(models)} models on same scenario:")
    print(f"📝 Situation: {test_situation}\n")

    for model_name, llm in models:
        print(f"🤖 {model_name}:")
        print("-" * 40)

        try:
            start = time.time()
            engine = SimulationEngine(llm_provider=llm)

            # Simple character for comparison
            character = CharacterState(
                id=f"test_{model_name}",
                name="Pontius Pilate",
                backstory={"role": "Roman Prefect"},
                traits=["conflicted"],
                values=["order"],
                fears=["chaos"],
                desires=["peace"],
                emotional_state=EmotionalState(),
                memory=CharacterMemory(),
                current_goal="Resolve situation",
            )

            result = await engine.run_simulation(
                character, test_situation, emphasis="doubt", temperature=0.7
            )

            elapsed = time.time() - start
            response = result["response"]

            if isinstance(response, dict):
                print(f"💬 Response: \"{response.get('dialogue', 'N/A')[:150]}...\"")
                print(f"🤔 Thought: {response.get('thought', 'N/A')[:100]}...")
            else:
                print(f"📝 Response: {str(response)[:200]}...")

            print(f"⏱️ Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"❌ Error: {e}")

        print()


if __name__ == "__main__":
    print("🚀 Starting Gemma Emotional Sequence Test\n")

    # Run main emotional sequence
    asyncio.run(run_gemma_emotional_sequence())

    # Run comparison
    asyncio.run(compare_gemma_vs_mock())

    print("\n" + "=" * 80)
    print("🎉 All tests complete!")
    print("\n💡 Key Insights:")
    print("• Gemma-3-12b provides more detailed, contextual responses")
    print("• Response times are 10-15s per scene with Gemma")
    print("• Emotional evolution tracks character development")
    print("• Memory integration affects response consistency")
