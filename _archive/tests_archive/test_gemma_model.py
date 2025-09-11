"""
Gemma-3-12b Model Test for Character Simulation
Test the Character Simulation Engine with Google's Gemma model
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
    MockLLM,
)


async def test_gemma_connection():
    """Test connection to LMStudio with Gemma-3-12b model"""

    print("🤖 GEMMA-3-12B MODEL TEST")
    print("=" * 60)
    print()

    print("📋 CHECKING LMSTUDIO WITH GEMMA-3-12B...")
    print("-" * 40)

    # Create LMStudio client
    lmstudio_llm = LMStudioLLM(
        endpoint="http://localhost:1234/v1",
        model="gemma-3-12b",  # Specify Gemma model if needed
    )

    try:
        # Check server health
        print("🔍 Checking server status...")
        is_available = await lmstudio_llm.health_check()

        if not is_available:
            print("❌ LMStudio server not responding")
            print("\n💡 SETUP INSTRUCTIONS:")
            print("1. Open LMStudio")
            print("2. Go to the Models tab")
            print("3. Search for 'gemma-3-12b' or 'gemma 12b'")
            print("4. Download the model (might take a while, it's ~7-8GB)")
            print("5. Load the model")
            print("6. Start the local server")
            print("7. Run this script again")
            return None

        print("✅ Server is running!")

        # Test model generation
        print("\n🧪 Testing Gemma-3-12b response generation...")
        print("⏳ Sending test prompt (this may take 10-30 seconds)...")

        start_time = time.time()

        test_prompt = """You are simulating a character in a story. 
        Respond ONLY with a valid JSON object containing these exact fields:
        - dialogue: what the character says (string)
        - thought: internal monologue (string)
        - action: physical actions taken (string)
        - emotional_shift: changes in emotions as an object with anger, doubt, fear, compassion as numbers between -0.3 and 0.3
        
        Character: Pontius Pilate facing Jesus
        Situation: You must decide his fate
        
        Respond with JSON only, no other text."""

        try:
            response = await lmstudio_llm.generate_response(
                test_prompt, temperature=0.7, max_tokens=400
            )

            elapsed = time.time() - start_time
            print(f"⏱️  Response generated in {elapsed:.2f} seconds")

            # Try to parse the response
            print("\n📝 Parsing Gemma response...")

            # Gemma might include markdown or extra text, try to extract JSON
            response_text = response.content

            # Try to find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]

                try:
                    parsed_response = json.loads(json_text)
                    print("✅ Successfully parsed JSON from Gemma!")

                    # Check for required fields
                    required_fields = [
                        "dialogue",
                        "thought",
                        "action",
                        "emotional_shift",
                    ]
                    missing = [f for f in required_fields if f not in parsed_response]

                    if missing:
                        print(f"⚠️  Missing fields: {missing}")
                        print("💡 Gemma may need more specific prompting")
                    else:
                        print("✅ All required fields present!")
                        print("\n🎭 Sample Gemma Response:")
                        print(f"💬 Dialogue: {parsed_response['dialogue'][:100]}...")
                        print(f"🤔 Thought: {parsed_response['thought'][:100]}...")
                        print(f"🎬 Action: {parsed_response['action'][:100]}...")

                    return lmstudio_llm

                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON parsing error: {e}")
                    print(f"📄 Raw JSON attempt: {json_text[:200]}...")

            else:
                print("❌ No JSON structure found in response")
                print(f"📄 Raw response: {response_text[:300]}...")

            print("\n💡 GEMMA TIPS:")
            print("• Gemma models can be verbose - may need strict prompting")
            print("• Try temperature 0.5-0.7 for more consistent JSON")
            print("• Consider using system prompts if supported")
            print("• May need to extract JSON from markdown blocks")

            return lmstudio_llm  # Return anyway for testing

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            print("\n💡 TROUBLESHOOTING:")
            print("• Check if Gemma-3-12b is fully loaded (check LMStudio)")
            print("• Ensure you have enough RAM (needs ~8-12GB free)")
            print("• Try closing other applications")
            print("• Consider using a smaller model if resources are limited")
            return None

    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None


async def run_gemma_emotional_test():
    """Run emotional sequence test with Gemma model"""

    print("\n" + "=" * 60)
    print("🎭 GEMMA EMOTIONAL SEQUENCE TEST")
    print("=" * 60)

    # Test Gemma connection
    llm = await test_gemma_connection()

    if not llm:
        print("\n⚠️  Falling back to MockLLM for demonstration...")
        llm = MockLLM()
        model_name = "MockLLM"
    else:
        model_name = "Gemma-3-12b"

    print(f"\n🚀 Running emotional sequence with {model_name}")
    print("-" * 40)

    # Create simulation engine
    engine = SimulationEngine(
        llm_provider=llm, max_concurrent=1  # Run sequentially for Gemma
    )

    # Create Pontius Pilate character
    pilate = CharacterState(
        id="pilate_gemma_test",
        name="Pontius Pilate",
        backstory={
            "origin": "Roman equestrian",
            "career": "Prefect of Judaea",
            "background": "Facing the most difficult decision of his career",
        },
        traits=["pragmatic", "conflicted", "duty-bound", "increasingly-desperate"],
        values=["Roman law", "order", "survival", "hidden justice"],
        fears=["rebellion", "Caesar's wrath", "divine judgment"],
        desires=["peace", "clarity", "escape from this situation"],
        emotional_state=EmotionalState(
            anger=0.2, doubt=0.4, fear=0.3, compassion=0.5, confidence=0.7
        ),
        memory=CharacterMemory(),
        current_goal="Navigate this trial without disaster",
        internal_conflict="Duty to Rome vs. sense of justice",
    )

    # Test scenarios specifically designed for Gemma
    test_scenarios = [
        {
            "name": "Initial Confrontation",
            "situation": "Jesus stands before you, accused by the religious leaders. The crowd watches. You must begin the interrogation.",
            "emphasis": "duty",
            "expected": "Professional, procedural response",
        },
        {
            "name": "Growing Doubt",
            "situation": "Jesus speaks of truth and kingdoms not of this world. His words are strange but compelling. You feel unsettled.",
            "emphasis": "doubt",
            "expected": "Confused, philosophical response",
        },
        {
            "name": "Crowd Pressure",
            "situation": "The crowd shouts for crucifixion. They threaten riots if you don't comply. Your position is precarious.",
            "emphasis": "fear",
            "expected": "Anxious, desperate response",
        },
        {
            "name": "Moment of Compassion",
            "situation": "You see the beaten prisoner, clearly innocent of any real crime. Your conscience speaks louder than politics.",
            "emphasis": "compassion",
            "expected": "Sympathetic, conflicted response",
        },
        {
            "name": "Final Decision",
            "situation": "You must decide: condemn an innocent man or risk everything. There is no good choice, only consequences.",
            "emphasis": "neutral",
            "expected": "Resigned, tragic response",
        },
    ]

    print(f"🎭 Testing {len(test_scenarios)} emotional scenarios with {model_name}")
    print(
        f"📍 Initial emotional state: Confidence={pilate.emotional_state.confidence:.1f}, Doubt={pilate.emotional_state.doubt:.1f}"
    )
    print()

    results_comparison = []

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"📖 SCENARIO {i}: {scenario['name']}")
        print(f"📝 Situation: {scenario['situation']}")
        print(f"🎯 Emphasis: {scenario['emphasis']}")
        print(f"💭 Expected: {scenario['expected']}")

        try:
            print(f"⏳ Generating with {model_name}...")
            start_time = time.time()

            # Add memory from previous scenarios
            if i > 1:
                memory = f"Previous scene: {test_scenarios[i-2]['name']}"
                await pilate.memory.add_event(memory)

            # Run simulation
            result = await engine.run_simulation(
                pilate,
                scenario["situation"],
                emphasis=scenario["emphasis"],
                temperature=0.7 if model_name == "Gemma-3-12b" else 0.8,
            )

            elapsed = time.time() - start_time
            response = result["response"]

            # Display response
            print(f"⏱️  Generated in {elapsed:.2f}s")

            # Handle both dict and string responses
            if isinstance(response, dict):
                dialogue = response.get("dialogue", "No dialogue")
                thought = response.get("thought", "No thought")
                action = response.get("action", "No action")

                print(
                    f'💬 Pilate: "{dialogue[:150]}..."'
                    if len(dialogue) > 150
                    else f'💬 Pilate: "{dialogue}"'
                )
                print(
                    f"🤔 Thought: {thought[:150]}..."
                    if len(thought) > 150
                    else f"🤔 Thought: {thought}"
                )
                print(f"🎬 Action: {action}")

                # Show emotional shifts
                if "emotional_shift" in response:
                    shifts = response["emotional_shift"]
                    if isinstance(shifts, dict):
                        shift_str = ", ".join(
                            [f"{k}:{v:+.2f}" for k, v in shifts.items() if v != 0]
                        )
                        if shift_str:
                            print(f"📊 Emotional shifts: {shift_str}")
            else:
                print(f"📝 Raw response: {str(response)[:300]}...")

            # Show current emotional state
            print(
                f"😰 Current state: Doubt={pilate.emotional_state.doubt:.2f}, Fear={pilate.emotional_state.fear:.2f}, Confidence={pilate.emotional_state.confidence:.2f}"
            )

            # Store for comparison
            results_comparison.append(
                {
                    "scenario": scenario["name"],
                    "model": model_name,
                    "time": elapsed,
                    "response_length": len(str(response)),
                }
            )

        except Exception as e:
            print(f"❌ Error in scenario {i}: {e}")

        print()

    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("-" * 40)

    print(f"🤖 Model tested: {model_name}")
    print(f"📝 Scenarios completed: {len(results_comparison)}/{len(test_scenarios)}")

    if results_comparison:
        avg_time = sum(r["time"] for r in results_comparison) / len(results_comparison)
        avg_length = sum(r["response_length"] for r in results_comparison) / len(
            results_comparison
        )

        print(f"⏱️  Average response time: {avg_time:.2f}s")
        print(f"📏 Average response length: {avg_length:.0f} characters")

    print("\n🎭 Final emotional state:")
    print(f"   Anger: {pilate.emotional_state.anger:.2f}")
    print(f"   Doubt: {pilate.emotional_state.doubt:.2f}")
    print(f"   Fear: {pilate.emotional_state.fear:.2f}")
    print(f"   Compassion: {pilate.emotional_state.compassion:.2f}")
    print(f"   Confidence: {pilate.emotional_state.confidence:.2f}")

    if model_name == "Gemma-3-12b":
        print("\n✨ GEMMA-3-12B NOTES:")
        print("• Gemma tends to be more verbose than other models")
        print("• May need prompt engineering for consistent JSON")
        print("• Good at understanding context and emotion")
        print("• Response times depend on hardware (GPU helps a lot)")

    print("\n🎉 Test complete!")


async def compare_models():
    """Quick comparison between Gemma and MockLLM"""

    print("\n" + "=" * 60)
    print("🔄 MODEL COMPARISON: GEMMA vs MOCK")
    print("=" * 60)

    # Simple test prompt
    test_prompt = "You face a moral dilemma. Respond with JSON containing dialogue, thought, action, emotional_shift."

    models_to_test = []

    # Try Gemma
    gemma_llm = LMStudioLLM(endpoint="http://localhost:1234/v1")
    if await gemma_llm.health_check():
        models_to_test.append(("Gemma-3-12b", gemma_llm))

    # Always test MockLLM for comparison
    models_to_test.append(("MockLLM", MockLLM()))

    print(f"📊 Testing {len(models_to_test)} models...")
    print()

    for model_name, llm in models_to_test:
        print(f"🤖 {model_name}:")

        try:
            start = time.time()
            response = await llm.generate_response(test_prompt, temperature=0.7)
            elapsed = time.time() - start

            print(f"  ⏱️  Response time: {elapsed:.2f}s")
            print(f"  📏 Response length: {len(response.content)} chars")

            # Try to parse
            try:
                if response.content.startswith("{"):
                    json.loads(response.content)
                    print("  ✅ Valid JSON output")
                else:
                    print("  ⚠️  Non-JSON output")
            except:
                print("  ⚠️  JSON parsing issues")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()

    print("💡 Key differences:")
    print("• Gemma: More creative, slower, needs more resources")
    print("• MockLLM: Fast, consistent, but limited variety")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_gemma_emotional_test())
    asyncio.run(compare_models())
