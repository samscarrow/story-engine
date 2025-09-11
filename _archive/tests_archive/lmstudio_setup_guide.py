"""
LMStudio Setup Guide and Real LLM Test
Instructions for connecting to real LLMs and testing emotional sequences
"""

import asyncio
import json
from character_simulation_engine_v2 import (
    CharacterState,
    EmotionalState,
    CharacterMemory,
    SimulationEngine,
    LMStudioLLM,
)


async def test_lmstudio_setup():
    """Test LMStudio setup and provide guidance"""

    print("🤖 LMSTUDIO SETUP GUIDE & REAL LLM TEST")
    print("=" * 60)
    print()

    print("📋 SETUP CHECKLIST:")
    print("1. ✅ Download and install LMStudio from https://lmstudio.ai/")
    print("2. 🧠 Load a model (recommended: Qwen2.5-7B-Instruct or similar)")
    print("3. 🌐 Start the local server (default: localhost:1234)")
    print("4. 🔧 Enable structured output in server settings")
    print()

    # Test connection
    print("🔍 TESTING LMSTUDIO CONNECTION...")
    print("-" * 40)

    lmstudio_llm = LMStudioLLM(endpoint="http://localhost:1234/v1")

    try:
        print("⏳ Checking server availability...")
        is_available = await lmstudio_llm.health_check()

        if not is_available:
            print("❌ LMStudio server not responding")
            print("💡 TROUBLESHOOTING:")
            print("   • Make sure LMStudio is running")
            print(
                "   • Check that the server is started (look for green 'Server Running' indicator)"
            )
            print("   • Verify the endpoint is localhost:1234")
            print("   • Try restarting LMStudio")
            return False

        print("✅ Server is responding!")

        # Test model loading
        print("🧠 Testing model availability...")
        try:
            test_response = await lmstudio_llm.generate_response(
                "You are testing a connection. Respond with JSON containing: dialogue, thought, action, emotional_shift.",
                temperature=0.7,
                max_tokens=200,
            )

            print("✅ Model generated a response!")
            print(f"📏 Response length: {len(test_response.content)} characters")

            # Test JSON parsing
            try:
                response_data = json.loads(test_response.content)
                print("✅ JSON parsing successful!")

                required_keys = ["dialogue", "thought", "action", "emotional_shift"]
                missing_keys = [
                    key for key in required_keys if key not in response_data
                ]

                if missing_keys:
                    print(f"⚠️  Missing required keys: {missing_keys}")
                    print(
                        "💡 Your model might need better prompting or a different model"
                    )
                else:
                    print("✅ All required JSON fields present!")
                    print("🎉 LMStudio setup is PERFECT for character simulation!")
                    return True

            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print("📝 Raw response preview:")
                print(
                    test_response.content[:300] + "..."
                    if len(test_response.content) > 300
                    else test_response.content
                )
                print()
                print("💡 POSSIBLE FIXES:")
                print("   • Try a different model (Qwen2.5, Llama-3.1, or similar)")
                print("   • Enable structured output in LMStudio server settings")
                print("   • Adjust temperature/max_tokens settings")
                return False

        except Exception as e:
            print(f"❌ Model generation failed: {e}")
            print("💡 POSSIBLE ISSUES:")
            print("   • No model is loaded in LMStudio")
            print("   • Model is still loading (check LMStudio interface)")
            print("   • Server configuration issue")
            return False

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 TROUBLESHOOTING:")
        print("   • LMStudio might not be installed or running")
        print("   • Check if another application is using port 1234")
        print("   • Try restarting your computer")
        return False


async def run_real_llm_emotional_test():
    """Run a quick emotional test with real LLM"""

    print("\n" + "=" * 60)
    print("🎭 REAL LLM EMOTIONAL TEST")
    print("=" * 60)

    # Test LMStudio setup
    setup_success = await test_lmstudio_setup()

    if not setup_success:
        print("\n❌ LMStudio setup incomplete. Please fix issues above and try again.")
        print("📖 For now, you can use the mock experiments:")
        print("   • python experiment.py")
        print("   • python advanced_experiment.py")
        print("   • python dramatic_emotional_journey.py")
        return

    print("\n🎉 RUNNING REAL LLM EMOTIONAL TEST...")
    print("=" * 50)

    # Create LLM and engine
    lmstudio_llm = LMStudioLLM(endpoint="http://localhost:1234/v1")
    engine = SimulationEngine(llm_provider=lmstudio_llm, max_concurrent=1)

    # Create test character
    test_character = CharacterState(
        id="pilate_real_test",
        name="Pontius Pilate",
        backstory={"origin": "Roman equestrian", "career": "Prefect of Judaea"},
        traits=["pragmatic", "conflicted", "duty-bound"],
        values=["Roman law", "order", "survival"],
        fears=["rebellion", "Caesar's wrath"],
        desires=["peace", "clear guidance"],
        emotional_state=EmotionalState(
            anger=0.3, doubt=0.6, fear=0.5, compassion=0.4, confidence=0.6
        ),
        memory=CharacterMemory(),
        current_goal="Navigate this trial successfully",
    )

    # Test scenarios
    test_scenarios = [
        {
            "situation": "Jesus stands before you, accused of claiming to be King of the Jews. What is your first response?",
            "emphasis": "doubt",
            "description": "Initial encounter with doubt emphasis",
        },
        {
            "situation": "The crowd grows violent, demanding crucifixion. You must maintain order while seeking justice.",
            "emphasis": "fear",
            "description": "Crowd pressure with fear emphasis",
        },
        {
            "situation": "Your wife warns you in a message about a dream: 'Have nothing to do with this righteous man.'",
            "emphasis": "compassion",
            "description": "Wife's warning with compassion emphasis",
        },
    ]

    print(f"🎭 Testing character: {test_character.name}")
    print(
        f"😰 Initial emotions: Doubt={test_character.emotional_state.doubt:.1f}, Fear={test_character.emotional_state.fear:.1f}"
    )
    print()

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"🎬 TEST {i}: {scenario['description']}")
        print(f"📝 Situation: {scenario['situation']}")
        print(f"🎯 Emphasis: {scenario['emphasis']}")

        try:
            print("⏳ Generating with real LLM...")

            result = await engine.run_simulation(
                test_character,
                scenario["situation"],
                emphasis=scenario["emphasis"],
                temperature=0.8,
            )

            response = result["response"]

            print(f"💬 Pilate: \"{response['dialogue']}\"")
            print(f"🤔 Thought: {response['thought']}")
            print(f"🎭 Action: {response['action']}")

            # Show emotional changes
            if "emotional_shift" in response and response["emotional_shift"]:
                shifts = []
                for emotion, change in response["emotional_shift"].items():
                    if isinstance(change, (int, float)) and abs(change) > 0.01:
                        direction = "↗️" if change > 0 else "↘️"
                        shifts.append(f"{emotion} {direction} {abs(change):.2f}")

                if shifts:
                    print(f"📊 Emotional shifts: {' | '.join(shifts)}")

            print(
                f"🌡️ Emotional temperature: {test_character.emotional_state.modulate_temperature():.2f}"
            )
            print()

        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try adjusting the model settings or using a different model")
            print()

    print("🎉 REAL LLM TEST COMPLETE!")
    print("=" * 50)
    print(
        "✅ If you see varied, contextual responses above, your setup is working perfectly!"
    )
    print("🚀 You can now run any experiment with real LLM by editing the scripts:")
    print("   • Change MockLLM() to LMStudioLLM() in any experiment file")
    print("   • Enjoy realistic, varied character responses!")


async def show_model_recommendations():
    """Show recommended models for character simulation"""

    print("\n" + "=" * 60)
    print("🧠 RECOMMENDED MODELS FOR CHARACTER SIMULATION")
    print("=" * 60)
    print()

    recommendations = [
        {
            "model": "Qwen2.5-7B-Instruct",
            "size": "~4GB",
            "pros": [
                "Excellent instruction following",
                "Good JSON output",
                "Fast inference",
            ],
            "cons": ["Medium context window"],
            "rating": "⭐⭐⭐⭐⭐",
        },
        {
            "model": "Llama-3.2-3B-Instruct",
            "size": "~2GB",
            "pros": ["Very fast", "Small memory footprint", "Decent quality"],
            "cons": ["Less creative", "Shorter responses"],
            "rating": "⭐⭐⭐⭐",
        },
        {
            "model": "Mistral-7B-Instruct-v0.3",
            "size": "~4GB",
            "pros": ["Creative responses", "Good character portrayal", "Reliable"],
            "cons": ["Can be verbose", "Moderate speed"],
            "rating": "⭐⭐⭐⭐",
        },
        {
            "model": "Phi-3.5-Mini-Instruct",
            "size": "~2GB",
            "pros": ["Very fast", "Good reasoning", "Compact"],
            "cons": ["Less creative", "Can be repetitive"],
            "rating": "⭐⭐⭐",
        },
    ]

    for rec in recommendations:
        print(f"🤖 {rec['model']} {rec['rating']}")
        print(f"   💾 Size: {rec['size']}")
        print(f"   ✅ Pros: {', '.join(rec['pros'])}")
        print(f"   ⚠️  Cons: {', '.join(rec['cons'])}")
        print()

    print("💡 TIPS FOR BEST RESULTS:")
    print("   • Use temperature 0.7-0.9 for creative character responses")
    print("   • Set max_tokens to 300-500 for detailed responses")
    print("   • Enable structured output for consistent JSON formatting")
    print("   • Try different models to find your preferred style")


if __name__ == "__main__":
    asyncio.run(run_real_llm_emotional_test())
    asyncio.run(show_model_recommendations())
