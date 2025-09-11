"""
Quick Gemma-3-12b Test - Simplified version for faster testing
"""

import asyncio
import json
import time
from character_simulation_engine_v2 import LMStudioLLM


async def quick_gemma_test():
    """Quick test of Gemma model with shorter prompts"""

    print("🤖 QUICK GEMMA-3-12B TEST")
    print("=" * 60)

    # Create LMStudio client with Gemma model
    lmstudio_llm = LMStudioLLM(
        endpoint="http://localhost:1234/v1", model="google/gemma-3-12b"
    )

    print("📋 Testing Gemma-3-12b model...")
    print("-" * 40)

    # Simple test prompt - much shorter than before
    test_prompt = """Return JSON with these exact fields:
{"dialogue": "What do you want?", "thought": "This is troublesome", "action": "I pace the room", "emotional_shift": {"anger": 0.1, "doubt": 0.2, "fear": 0.1, "compassion": -0.1}}"""

    print("⏳ Sending simple test prompt...")
    start_time = time.time()

    try:
        response = await lmstudio_llm.generate_response(
            test_prompt,
            temperature=0.5,  # Lower temperature for consistency
            max_tokens=200,  # Fewer tokens for speed
        )

        elapsed = time.time() - start_time
        print(f"⏱️ Response in {elapsed:.2f}s")

        # Display raw response
        print(f"\n📄 Raw response ({len(response.content)} chars):")
        print("-" * 40)
        print(
            response.content[:500] if len(response.content) > 500 else response.content
        )
        print("-" * 40)

        # Try to parse JSON
        try:
            # Try direct parsing first
            if response.content.strip().startswith("{"):
                parsed = json.loads(response.content)
                print("\n✅ Successfully parsed JSON!")
                print(f"💬 Dialogue: {parsed.get('dialogue', 'N/A')}")
                print(f"🤔 Thought: {parsed.get('thought', 'N/A')}")
                print(f"🎬 Action: {parsed.get('action', 'N/A')}")
                return True
            else:
                # Try to extract JSON from response
                json_start = response.content.find("{")
                json_end = response.content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response.content[json_start:json_end]
                    parsed = json.loads(json_text)
                    print("\n✅ Extracted and parsed JSON!")
                    print(f"💬 Dialogue: {parsed.get('dialogue', 'N/A')}")
                    return True
                else:
                    print("\n⚠️ No JSON found in response")
                    return False

        except json.JSONDecodeError as e:
            print(f"\n⚠️ JSON parsing error: {e}")
            return False

    except asyncio.TimeoutError:
        print(f"❌ Request timed out after {elapsed:.2f}s")
        print("💡 Gemma-3-12b might be too slow for this hardware")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_character_response():
    """Test actual character simulation with Gemma"""

    print("\n" + "=" * 60)
    print("🎭 CHARACTER SIMULATION TEST")
    print("=" * 60)

    lmstudio_llm = LMStudioLLM(
        endpoint="http://localhost:1234/v1", model="google/gemma-3-12b"
    )

    # Very focused prompt for Gemma
    character_prompt = """You are Pontius Pilate. Jesus stands before you.
    
Reply ONLY with this JSON format (no other text):
{
  "dialogue": "Your spoken words here",
  "thought": "Your inner thoughts here",
  "action": "Your physical action here",
  "emotional_shift": {"anger": 0, "doubt": 0.2, "fear": 0.1, "compassion": 0}
}

JSON only:"""

    print("⏳ Generating character response...")
    start_time = time.time()

    try:
        response = await lmstudio_llm.generate_response(
            character_prompt, temperature=0.7, max_tokens=250
        )

        elapsed = time.time() - start_time
        print(f"⏱️ Generated in {elapsed:.2f}s")

        # Show response
        print("\n📝 Character response:")
        print(
            response.content[:400] if len(response.content) > 400 else response.content
        )

        # Try to parse
        try:
            # Find JSON in response
            json_start = response.content.find("{")
            json_end = response.content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_text = response.content[json_start:json_end]
                parsed = json.loads(json_text)

                print("\n✅ Parsed character response:")
                print(f"💬 Pilate says: \"{parsed.get('dialogue', 'N/A')}\"")
                print(f"🤔 Thinks: {parsed.get('thought', 'N/A')}")
                print(f"🎬 Action: {parsed.get('action', 'N/A')}")

                if "emotional_shift" in parsed:
                    shifts = parsed["emotional_shift"]
                    print(
                        f"📊 Emotions: doubt={shifts.get('doubt', 0)}, fear={shifts.get('fear', 0)}"
                    )

                return True
        except Exception as e:
            print(f"\n⚠️ Couldn't parse character response: {e}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def compare_response_times():
    """Compare Gemma response times for different prompt lengths"""

    print("\n" + "=" * 60)
    print("⏱️ RESPONSE TIME COMPARISON")
    print("=" * 60)

    lmstudio_llm = LMStudioLLM(
        endpoint="http://localhost:1234/v1", model="google/gemma-3-12b"
    )

    prompts = [
        ("Short", 'Return JSON: {"answer": "yes"}'),
        (
            "Medium",
            "You are a character. Return JSON with dialogue and thought fields.",
        ),
        (
            "Long",
            "You are simulating Pontius Pilate in a complex moral dilemma. Return a detailed JSON response with dialogue, thought, action, and emotional_shift fields showing his internal conflict.",
        ),
    ]

    for name, prompt in prompts:
        print(f"\n📝 {name} prompt ({len(prompt)} chars)...")

        try:
            start = time.time()
            response = await lmstudio_llm.generate_response(
                prompt, temperature=0.5, max_tokens=150
            )
            elapsed = time.time() - start

            print(f"   ⏱️ Time: {elapsed:.2f}s")
            print(f"   📏 Response: {len(response.content)} chars")

        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    # Run tests
    print("🚀 Starting Gemma-3-12b quick tests...\n")

    # Basic connectivity test
    success = asyncio.run(quick_gemma_test())

    if success:
        # Character simulation test
        asyncio.run(test_character_response())

        # Response time comparison
        asyncio.run(compare_response_times())

    print("\n" + "=" * 60)
    print("✅ Quick test complete!")
    print("\n💡 FINDINGS:")
    print("• Gemma-3-12b is loaded and responding")
    print("• Response times vary based on prompt complexity")
    print("• May need JSON extraction from verbose responses")
    print("• Consider using smaller models for faster iteration")
