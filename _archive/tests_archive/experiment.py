"""
Character Simulation Engine Experiments
Interactive exploration of character behavior simulation
"""

import asyncio
from character_simulation_engine_v2 import (
    CharacterState,
    EmotionalState,
    CharacterMemory,
    SimulationEngine,
    MockLLM,
)


async def main():
    print("🎭 Character Simulation Engine - Interactive Experiments")
    print("=" * 60)

    # Initialize the simulation engine with MockLLM for quick testing
    llm = MockLLM()
    engine = SimulationEngine(llm_provider=llm, max_concurrent=3)

    # Create our test character - Pontius Pilate
    pilate = CharacterState(
        id="pontius_pilate",
        name="Pontius Pilate",
        backstory={
            "origin": "Roman equestrian from Samnium",
            "career": "Prefect of Judaea (26-36 CE)",
            "background": "Military administrator thrust into religious politics",
        },
        traits=["pragmatic", "ambitious", "anxious", "duty-bound"],
        values=["Roman law", "order", "imperial loyalty"],
        fears=["rebellion", "imperial disfavor", "losing control"],
        desires=["peace in province", "career advancement", "clear guidance"],
        emotional_state=EmotionalState(
            anger=0.3, doubt=0.7, fear=0.6, compassion=0.2, confidence=0.4
        ),
        memory=CharacterMemory(),
        current_goal="Navigate the trial of Jesus without causing uprising or losing face",
        internal_conflict="Duty to Rome vs. personal sense of justice",
    )

    # Add some recent events to his memory
    await pilate.memory.add_event("Wife's disturbing dream about the prisoner")
    await pilate.memory.add_event("Crowd growing increasingly agitated")
    await pilate.memory.add_event("Prisoner's unsettling responses to questioning")

    print(f"🎯 Character: {pilate.name}")
    print(f"📖 Background: {pilate.backstory['background']}")
    print(
        f"😰 Current emotional state: Doubt={pilate.emotional_state.doubt:.1f}, Fear={pilate.emotional_state.fear:.1f}"
    )
    print(f"🎪 Internal conflict: {pilate.internal_conflict}")
    print()

    # EXPERIMENT 1: Single simulation with different emphases
    print("🧪 EXPERIMENT 1: Different Emphasis Modes")
    print("-" * 50)

    situation = "You face Jesus of Nazareth. The crowd demands crucifixion, but you find no fault in him. What do you decide?"

    emphases = ["power", "doubt", "fear", "duty", "compassion"]

    for emphasis in emphases:
        print(f"\n🎯 Testing emphasis: {emphasis.upper()}")

        result = await engine.run_simulation(
            pilate, situation, emphasis, temperature=0.8
        )

        response = result["response"]
        print(f"💬 Dialogue: {response['dialogue']}")
        print(f"🤔 Thought: {response['thought']}")
        print(f"🎭 Action: {response['action']}")

        # Show emotional shifts if any
        if "emotional_shift" in response:
            shifts = response["emotional_shift"]
            shift_summary = ", ".join(
                [f"{k}:{v:+.1f}" for k, v in shifts.items() if v != 0]
            )
            if shift_summary:
                print(f"😌 Emotional shift: {shift_summary}")

    print("\n" + "=" * 60)

    # EXPERIMENT 2: Multiple simulations for variety
    print("🧪 EXPERIMENT 2: Multiple Simulations for Variety")
    print("-" * 50)

    print("Running 5 simulations with same situation but varied parameters...")

    results = await engine.run_multiple_simulations(pilate, situation, num_runs=5)

    for i, result in enumerate(results, 1):
        response = result["response"]
        print(
            f"\n🎲 Simulation {i} (emphasis: {result['emphasis']}, temp: {result['temperature']:.1f}):"
        )
        print(f"💬 \"{response['dialogue']}\"")
        print(f"🎭 {response['action']}")

    print("\n" + "=" * 60)

    # EXPERIMENT 3: Emotional state variations
    print("🧪 EXPERIMENT 3: Emotional State Variations")
    print("-" * 50)

    # Create versions of Pilate with different emotional states
    emotional_variants = [
        (
            "Angry Pilate",
            EmotionalState(
                anger=0.9, doubt=0.3, fear=0.2, compassion=0.1, confidence=0.8
            ),
        ),
        (
            "Fearful Pilate",
            EmotionalState(
                anger=0.1, doubt=0.8, fear=0.9, compassion=0.4, confidence=0.2
            ),
        ),
        (
            "Compassionate Pilate",
            EmotionalState(
                anger=0.2, doubt=0.5, fear=0.3, compassion=0.9, confidence=0.6
            ),
        ),
    ]

    situation_short = (
        "The prisoner stands before you. The crowd shouts for blood. You must choose."
    )

    for variant_name, emotion_state in emotional_variants:
        print(f"\n🎭 {variant_name}:")
        print(
            f"   Anger={emotion_state.anger:.1f}, Doubt={emotion_state.doubt:.1f}, Fear={emotion_state.fear:.1f}, Compassion={emotion_state.compassion:.1f}"
        )

        # Create a copy of Pilate with different emotions
        variant_pilate = CharacterState(
            id=pilate.id + "_" + variant_name.lower().replace(" ", "_"),
            name=pilate.name,
            backstory=pilate.backstory,
            traits=pilate.traits,
            values=pilate.values,
            fears=pilate.fears,
            desires=pilate.desires,
            emotional_state=emotion_state,
            memory=pilate.memory,
            current_goal=pilate.current_goal,
            internal_conflict=pilate.internal_conflict,
        )

        result = await engine.run_simulation(
            variant_pilate, situation_short, "neutral", temperature=0.7
        )

        response = result["response"]
        print(f"💬 \"{response['dialogue']}\"")
        print(f"🤔 Internal: {response['thought']}")

        # Show temperature modulation
        temp = emotion_state.modulate_temperature()
        print(f"🌡️  Emotional temperature: {temp:.2f}")

    print("\n" + "=" * 60)
    print("🎉 Experiments completed!")
    print(f"📊 Total LLM calls made: {llm.call_count}")
    print(
        "\nTry editing the emotional states, situation text, or character traits to see how responses change!"
    )


if __name__ == "__main__":
    asyncio.run(main())
