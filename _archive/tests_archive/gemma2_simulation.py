"""
Gemma-2-27b Character Simulation
Test the larger Gemma-2 model for character simulation
"""

import asyncio
import json
import time
from character_simulation_engine_v2 import (
    CharacterState,
    EmotionalState,
    CharacterMemory,
    SimulationEngine,
    LMStudioLLM,
)


async def test_gemma2_quick():
    """Quick test of Gemma-2-27b model"""

    print("🤖 GEMMA-2-27B QUICK TEST")
    print("=" * 70)

    # Connect to Gemma-2-27b
    llm = LMStudioLLM(endpoint="http://localhost:1234/v1", model="google/gemma-2-27b")

    print("🔌 Connecting to Gemma-2-27b...")

    if not await llm.health_check():
        print("❌ Gemma-2-27b not available")
        return False

    print("✅ Connected to Gemma-2-27b")

    # Test basic response
    test_prompt = """You are Pontius Pilate. Respond with JSON only:
{"dialogue": "your words", "thought": "your thoughts", "action": "your action", "emotional_shift": {"anger": 0, "doubt": 0.1, "fear": 0.1, "compassion": 0}}"""

    print("\n⏳ Testing response generation...")
    start = time.time()

    try:
        response = await llm.generate_response(
            test_prompt, temperature=0.6, max_tokens=250
        )

        elapsed = time.time() - start
        print(f"⏱️  Response in {elapsed:.1f}s")

        # Parse response
        content = response.content.strip()

        # Try to extract JSON
        if content.startswith("{"):
            data = json.loads(content)
        else:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(content[json_start:json_end])
            else:
                print("❌ No JSON found in response")
                print(f"📝 Raw: {content[:200]}")
                return False

        print("\n✅ Parsed response:")
        print(f"💬 {data.get('dialogue', 'N/A')}")
        print(f"🤔 {data.get('thought', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def run_gemma2_emotional_sequence():
    """Run full emotional sequence with Gemma-2-27b"""

    print("\n" + "=" * 70)
    print("🎭 GEMMA-2-27B EMOTIONAL SEQUENCE")
    print("=" * 70)

    # Setup
    llm = LMStudioLLM(endpoint="http://localhost:1234/v1", model="google/gemma-2-27b")

    if not await llm.health_check():
        print("❌ Gemma-2-27b not available")
        return

    print("✅ Using Gemma-2-27b (larger model)")

    engine = SimulationEngine(llm_provider=llm, max_concurrent=1)

    # Create Pilate character
    pilate = CharacterState(
        id="pilate_g2",
        name="Pontius Pilate",
        backstory={
            "role": "Roman Prefect of Judaea",
            "situation": "Judging Jesus of Nazareth",
            "pressure": "Crowd demands crucifixion",
        },
        traits=["pragmatic", "conflicted", "political"],
        values=["Roman order", "survival", "hidden justice"],
        fears=["rebellion", "Caesar's anger", "divine judgment"],
        desires=["peace", "clarity", "escape"],
        emotional_state=EmotionalState(
            anger=0.2, doubt=0.3, fear=0.3, compassion=0.4, confidence=0.7
        ),
        memory=CharacterMemory(),
        current_goal="Resolve this trial without riot",
        internal_conflict="Duty vs conscience",
    )

    # Key scenes for testing
    scenes = [
        {
            "name": "Opening Interrogation",
            "situation": "Jesus stands before you, accused of sedition. The Sanhedrin watches. You must begin questioning.",
            "emphasis": "duty",
        },
        {
            "name": "Truth Discussion",
            "situation": "Jesus says 'I came to testify to the truth. Everyone on the side of truth listens to me.' How do you respond?",
            "emphasis": "doubt",
            "memory": "He speaks of truth I cannot grasp",
        },
        {
            "name": "Crowd Pressure",
            "situation": "The crowd chants 'Crucify him!' They threaten violence if you don't comply. Caesar will not tolerate disorder.",
            "emphasis": "fear",
            "memory": "The mob grows dangerous",
        },
        {
            "name": "Final Choice",
            "situation": "You must decide: condemn an innocent man or face potential rebellion and Caesar's wrath. There is no good option.",
            "emphasis": "neutral",
            "memory": "All paths lead to suffering",
        },
    ]

    print(f"\n📊 Testing {len(scenes)} scenes")
    print(
        f"📍 Initial: Confidence={pilate.emotional_state.confidence:.2f}, Doubt={pilate.emotional_state.doubt:.2f}\n"
    )

    results = []

    for i, scene in enumerate(scenes, 1):
        print(f"{'='*60}")
        print(f"🎬 SCENE {i}: {scene['name']}")
        print(f"📝 {scene['situation']}")
        print(f"🎯 Emphasis: {scene['emphasis']}")

        if "memory" in scene:
            await pilate.memory.add_event(scene["memory"])
            print(f"💭 Memory: {scene['memory']}")

        try:
            print("⏳ Generating...")
            start = time.time()

            result = await engine.run_simulation(
                pilate, scene["situation"], emphasis=scene["emphasis"], temperature=0.7
            )

            elapsed = time.time() - start
            response = result["response"]

            print(f"⏱️  {elapsed:.1f}s")

            if isinstance(response, dict):
                # Show response
                dialogue = response.get("dialogue", "")
                thought = response.get("thought", "")
                action = response.get("action", "")

                print(
                    f"\n💬 \"{dialogue[:150]}{'...' if len(dialogue) > 150 else ''}\""
                )
                print(f"🤔 {thought[:100]}{'...' if len(thought) > 100 else ''}")
                print(f"🎬 {action[:80]}{'...' if len(action) > 80 else ''}")

                # Track emotions
                results.append(
                    {
                        "scene": scene["name"],
                        "time": elapsed,
                        "emotions": {
                            "doubt": pilate.emotional_state.doubt,
                            "fear": pilate.emotional_state.fear,
                            "confidence": pilate.emotional_state.confidence,
                        },
                    }
                )

                # Show emotional state
                print(
                    f"\n📊 State: Doubt={pilate.emotional_state.doubt:.2f}, "
                    f"Fear={pilate.emotional_state.fear:.2f}, "
                    f"Confidence={pilate.emotional_state.confidence:.2f}"
                )
            else:
                print(f"⚠️  Unexpected response type: {type(response)}")

        except Exception as e:
            print(f"❌ Error: {e}")

        print()

    # Summary
    if results:
        print("=" * 70)
        print("📊 GEMMA-2-27B PERFORMANCE SUMMARY")
        print("=" * 70)

        avg_time = sum(r["time"] for r in results) / len(results)
        print(f"\n⏱️  Average response: {avg_time:.1f}s")
        print(f"📝 Scenes completed: {len(results)}/{len(scenes)}")

        if len(results) > 1:
            first = results[0]["emotions"]
            last = results[-1]["emotions"]

            print("\n📈 Emotional changes:")
            print(
                f"  Doubt: {first['doubt']:.2f} → {last['doubt']:.2f} ({last['doubt']-first['doubt']:+.2f})"
            )
            print(
                f"  Fear: {first['fear']:.2f} → {last['fear']:.2f} ({last['fear']-first['fear']:+.2f})"
            )
            print(
                f"  Confidence: {first['confidence']:.2f} → {last['confidence']:.2f} ({last['confidence']-first['confidence']:+.2f})"
            )


async def compare_gemma_models():
    """Compare Gemma-2-27b vs Gemma-3-12b"""

    print("\n" + "=" * 70)
    print("🔄 GEMMA MODEL COMPARISON")
    print("=" * 70)

    test_prompt = """You are Pontius Pilate facing Jesus. The crowd demands crucifixion.
Reply with JSON: {"dialogue": "words", "thought": "thoughts", "action": "action", "emotional_shift": {"anger": 0, "doubt": 0.2, "fear": 0.1, "compassion": 0}}"""

    models = [
        ("google/gemma-2-27b", "Gemma-2-27b (larger)"),
        ("google/gemma-3-12b", "Gemma-3-12b (newer)"),
    ]

    print("\n📊 Testing identical prompt on both models:\n")

    for model_id, model_name in models:
        print(f"🤖 {model_name}:")
        print("-" * 40)

        llm = LMStudioLLM(endpoint="http://localhost:1234/v1", model=model_id)

        try:
            if not await llm.health_check():
                print("❌ Model not available")
                continue

            start = time.time()
            response = await llm.generate_response(
                test_prompt, temperature=0.7, max_tokens=300
            )
            elapsed = time.time() - start

            # Parse response
            content = response.content.strip()

            # Extract JSON
            if content.startswith("{"):
                data = json.loads(content)
            else:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0:
                    data = json.loads(content[json_start:json_end])
                else:
                    print("❌ No JSON in response")
                    continue

            print(f"⏱️  Response time: {elapsed:.1f}s")
            print(f"💬 \"{data.get('dialogue', 'N/A')[:100]}...\"")
            print(f"🤔 {data.get('thought', 'N/A')[:80]}...")
            print(f"📏 Response length: {len(content)} chars")

        except Exception as e:
            print(f"❌ Error: {e}")

        print()

    print("💡 Key differences:")
    print("• Gemma-2-27b: Larger model (27B params), potentially more nuanced")
    print("• Gemma-3-12b: Newer architecture, more efficient")


if __name__ == "__main__":

    print("🚀 GEMMA-2-27B CHARACTER SIMULATION\n")

    # Run all tests
    asyncio.run(test_gemma2_quick())
    asyncio.run(run_gemma2_emotional_sequence())
    asyncio.run(compare_gemma_models())
