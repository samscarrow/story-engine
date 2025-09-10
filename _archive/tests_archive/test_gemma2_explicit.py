"""
Explicit Gemma-2-27b Test
Ensure we're using the correct model
"""

import asyncio
import json
import aiohttp
import time

async def direct_api_call(prompt: str, model: str = "google/gemma-2-27b"):
    """Make direct API call to ensure correct model"""
    
    url = "http://localhost:1234/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system", 
                "content": "You are Pontius Pilate. Respond only with JSON containing: dialogue, thought, action, emotional_shift"
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 400
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            data = await response.json()
            return data

async def test_gemma2_directly():
    """Test Gemma-2-27b with direct API calls"""
    
    print("🎯 DIRECT GEMMA-2-27B TEST")
    print("=" * 70)
    print()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Initial Meeting",
            "prompt": "Jesus is brought before you. The Jewish leaders accuse him of claiming to be King. What is your response?"
        },
        {
            "name": "Truth Question",
            "prompt": "Jesus says 'I came to bear witness to the truth.' You are puzzled. How do you respond?"
        },
        {
            "name": "Crowd Pressure",
            "prompt": "The crowd shouts 'Crucify him!' They threaten riot. You must maintain order. What do you do?"
        }
    ]
    
    print(f"📊 Testing {len(scenarios)} scenarios with Gemma-2-27b\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{'='*60}")
        print(f"🎬 SCENARIO {i}: {scenario['name']}")
        print(f"📝 {scenario['prompt'][:100]}...")
        
        try:
            print("⏳ Calling Gemma-2-27b...")
            start = time.time()
            
            result = await direct_api_call(scenario['prompt'])
            
            elapsed = time.time() - start
            
            # Verify model
            model_used = result.get('model', 'unknown')
            if model_used != "google/gemma-2-27b":
                print(f"⚠️  WARNING: Expected gemma-2-27b, got {model_used}")
            else:
                print(f"✅ Confirmed using: {model_used}")
            
            print(f"⏱️  Response time: {elapsed:.1f}s")
            
            # Extract content
            content = result['choices'][0]['message']['content']
            
            # Parse JSON from content
            try:
                # Try direct parse
                if content.strip().startswith('{'):
                    data = json.loads(content)
                else:
                    # Extract JSON
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        data = json.loads(content[json_start:json_end])
                    else:
                        print("❌ No JSON in response")
                        print(f"📝 Raw: {content[:200]}")
                        continue
                
                # Display response
                print(f"\n💬 Pilate: \"{data.get('dialogue', 'N/A')[:150]}\"")
                print(f"🤔 Thinks: {data.get('thought', 'N/A')[:100]}")
                print(f"🎬 Action: {data.get('action', 'N/A')[:80]}")
                
                # Show emotional shifts
                if 'emotional_shift' in data and isinstance(data['emotional_shift'], dict):
                    shifts = data['emotional_shift']
                    significant = []
                    for emotion, value in shifts.items():
                        if isinstance(value, (int, float)) and abs(value) > 0.05:
                            significant.append(f"{emotion}:{value:+.1f}")
                    if significant:
                        print(f"📊 Emotions: {', '.join(significant)}")
                
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error: {e}")
                print(f"📝 Content: {content[:200]}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("=" * 70)
    print("✅ Test complete - confirmed using Gemma-2-27b")

async def compare_models_directly():
    """Compare different models with direct API calls"""
    
    print("\n" + "=" * 70)
    print("🔄 DIRECT MODEL COMPARISON")
    print("=" * 70)
    
    prompt = "The crowd demands crucifixion. You must decide. What do you do?"
    
    models = [
        "google/gemma-2-27b",
        "google/gemma-3-12b"
    ]
    
    print(f"\n📝 Test prompt: {prompt}\n")
    
    for model in models:
        print(f"🤖 {model}:")
        print("-" * 40)
        
        try:
            start = time.time()
            result = await direct_api_call(prompt, model)
            elapsed = time.time() - start
            
            # Verify model
            actual_model = result.get('model', 'unknown')
            if actual_model != model:
                print(f"⚠️  WARNING: Requested {model}, got {actual_model}")
            
            content = result['choices'][0]['message']['content']
            
            print(f"✅ Model: {actual_model}")
            print(f"⏱️  Time: {elapsed:.1f}s")
            print(f"📏 Length: {len(content)} chars")
            
            # Try to show dialogue
            if '{' in content:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                try:
                    data = json.loads(content[json_start:json_end])
                    print(f"💬 \"{data.get('dialogue', 'N/A')[:100]}...\"")
                except:
                    print(f"📝 {content[:100]}...")
            else:
                print(f"📝 {content[:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    print("🚀 EXPLICIT GEMMA-2-27B TEST\n")
    
    asyncio.run(test_gemma2_directly())
    asyncio.run(compare_models_directly())