"""
Test KoboldCpp Configuration Options
Explore different parameters to get better structured responses
"""

import asyncio
import aiohttp
import json

async def test_kobold_config(config_name: str, params: dict, prompt: str) -> dict:
    """Test a specific KoboldCpp configuration"""
    
    url = "http://localhost:5001/api/v1/generate"
    
    # Base payload
    payload = {
        "prompt": prompt,
        "max_context_length": 4096,
        "max_length": 200
    }
    
    # Merge with test params
    payload.update(params)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if 'results' in data and len(data['results']) > 0:
                    return {
                        "config": config_name,
                        "params": params,
                        "response": data['results'][0]['text']
                    }
                return {"config": config_name, "error": "No response"}
    except Exception as e:
        return {"config": config_name, "error": str(e)}

async def main():
    """Test different KoboldCpp configurations"""
    
    print("🧪 KOBOLDCPP CONFIGURATION TESTING")
    print("=" * 70)
    
    # Simple test prompt
    test_prompt = "List three improvements for a story engine. Be brief and specific."
    
    # Different configurations to test
    configs = [
        {
            "name": "Low Temperature",
            "params": {
                "temperature": 0.1,
                "top_p": 0.5,
                "top_k": 10,
                "rep_pen": 1.0
            }
        },
        {
            "name": "Focused Sampling",
            "params": {
                "temperature": 0.5,
                "top_p": 0.7,
                "top_k": 20,
                "rep_pen": 1.2,
                "rep_pen_range": 256
            }
        },
        {
            "name": "Strict Stops",
            "params": {
                "temperature": 0.4,
                "top_p": 0.8,
                "stop_sequence": [".", "\n", "!", "?"],
                "trim_stop": False,
                "max_length": 100
            }
        },
        {
            "name": "Instruction Format",
            "params": {
                "temperature": 0.3,
                "prompt": f"### Instruction: {test_prompt}\n### Response:",
                "stop_sequence": ["###", "\n\n"],
                "trim_stop": True
            }
        },
        {
            "name": "Q&A Format",
            "params": {
                "temperature": 0.3,
                "prompt": f"Q: {test_prompt}\nA:",
                "stop_sequence": ["\nQ:", "\n\n"],
                "trim_stop": True
            }
        },
        {
            "name": "List Format",
            "params": {
                "temperature": 0.3,
                "prompt": f"Task: {test_prompt}\n\n1.",
                "stop_sequence": ["\n\n", "\n4."],
                "trim_stop": False
            }
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📝 Testing: {config['name']}")
        print("-" * 40)
        
        # Use custom prompt if specified
        prompt = config['params'].pop('prompt', test_prompt)
        
        result = await test_kobold_config(
            config['name'],
            config['params'],
            prompt
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Response: {result['response'][:200]}")
            if len(result['response']) > 200:
                print("...")
        
        results.append(result)
        await asyncio.sleep(2)
    
    # Save results
    with open('kobold_config_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("📊 CONFIGURATION TEST COMPLETE")
    print("✅ Results saved to 'kobold_config_test_results.json'")
    
    # Find best configuration
    print("\n🏆 BEST CONFIGURATIONS:")
    for result in results:
        if 'error' not in result and len(result.get('response', '')) > 50:
            print(f"- {result['config']}: {len(result['response'])} chars")

if __name__ == "__main__":
    asyncio.run(main())