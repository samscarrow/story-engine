"""
Get Structured Architecture Feedback via KoboldCpp
Uses JSON formatting and specific generation parameters for cleaner responses
"""

import asyncio
import aiohttp
import json

async def ask_kobold_structured(prompt: str, max_tokens: int = 500) -> dict:
    """
    Ask KoboldCpp for structured JSON responses
    
    Args:
        prompt: The structured prompt requesting JSON output
        max_tokens: Maximum tokens to generate
    """
    
    # KoboldCpp API endpoint
    url = "http://localhost:5001/api/v1/generate"
    
    # Enhanced parameters for structured output
    payload = {
        "prompt": prompt,
        "max_context_length": 4096,
        "max_length": max_tokens,
        "temperature": 0.3,  # Lower for more focused responses
        "top_p": 0.8,        # Tighter nucleus sampling
        "top_k": 30,         # Limit vocabulary for consistency
        "rep_pen": 1.15,     # Slightly higher repetition penalty
        "rep_pen_range": 512,
        "sampler_order": [6, 0, 1, 3, 4, 2, 5],
        "stop_sequence": ["```", "\n\n\n", "###", "---"],
        "trim_stop": True,
        "use_default_badwordsids": True,
        "grammar": "json",  # Request JSON formatting if supported
        "dynatemp_range": 0.0,  # Disable dynamic temperature
        "smoothing_factor": 0.0,
        "banned_tokens": []
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=30) as response:
                data = await response.json()
                if 'results' in data and len(data['results']) > 0:
                    text = data['results'][0]['text']
                    # Try to extract JSON from response
                    try:
                        # Look for JSON in the response
                        if '{' in text:
                            json_start = text.find('{')
                            json_end = text.rfind('}') + 1
                            if json_end > json_start:
                                json_text = text[json_start:json_end]
                                return json.loads(json_text)
                    except:
                        pass
                    return {"response": text}
                else:
                    return {"error": "No response generated"}
    except Exception as e:
        return {"error": str(e)}

async def get_structured_feedback():
    """Get structured architectural feedback"""
    
    print("🤖 STRUCTURED ARCHITECTURAL FEEDBACK VIA KOBOLDCPP")
    print("=" * 70)
    print("Using JSON-formatted prompts for cleaner responses")
    print()
    
    # Structured questions with explicit JSON format requests
    questions = [
        {
            "topic": "Critical Weakness Analysis",
            "prompt": """Analyze this story generation architecture and respond in JSON format:

Story Engine Layers:
1. Story Structure - generates plot points
2. Scene Crafting - creates detailed scenes  
3. Character Simulation - generates dialogue/actions
4. Quality Evaluation - measures 8 metrics
5. Enhancement - iterative improvement

Respond with this exact JSON structure:
{
    "weakness": "name of the critical weakness",
    "reason": "why this is the most critical issue",
    "solution": "specific fix to implement",
    "impact": "expected improvement from fix"
}""",
            "max_tokens": 300
        },
        {
            "topic": "Missing Components",
            "prompt": """Identify missing components for a story engine. Return JSON:
{
    "missing_components": [
        {
            "name": "component name",
            "purpose": "what it does",
            "implementation": "how to build it"
        }
    ]
}

List exactly 3 components that would most improve narrative quality.""",
            "max_tokens": 400
        },
        {
            "topic": "Character Psychology",
            "prompt": """Design character psychology dimensions. Return JSON:
{
    "dimensions": [
        {"name": "anger", "range": "0-1", "impact": "affects aggression"},
        {"name": "fear", "range": "0-1", "impact": "affects caution"},
        {"name": "doubt", "range": "0-1", "impact": "affects decisions"},
        {"name": "compassion", "range": "0-1", "impact": "affects helpfulness"},
        {"name": "confidence", "range": "0-1", "impact": "affects boldness"}
    ]
}

Add 3 more critical dimensions to complete an 8-dimension model.""",
            "max_tokens": 350
        },
        {
            "topic": "Agent Architecture",
            "prompt": """Design story writing agents. Return JSON:
{
    "agents": [
        {
            "name": "agent name",
            "role": "primary responsibility",
            "decisions": "what they decide"
        }
    ]
}

Include exactly 5 specialized agents for collaborative story creation.""",
            "max_tokens": 400
        },
        {
            "topic": "Quality Metrics",
            "prompt": """List story quality metrics. Return JSON:
{
    "metrics": [
        {
            "name": "metric name",
            "measures": "what it tracks",
            "calculation": "how to measure it"
        }
    ]
}

List 4 uncommon but crucial metrics for compelling narratives.""",
            "max_tokens": 400
        },
        {
            "topic": "Implementation Priority",
            "prompt": """Rank these improvements by impact. Return JSON:
{
    "rankings": [
        {"rank": 1, "item": "A/B/C/D/E", "reason": "why this ranks here"},
        {"rank": 2, "item": "A/B/C/D/E", "reason": "why this ranks here"},
        {"rank": 3, "item": "A/B/C/D/E", "reason": "why this ranks here"}
    ]
}

Items to rank:
A. Better character psychology
B. Scene-to-scene continuity  
C. Multi-agent collaboration
D. Advanced quality metrics
E. Branching exploration""",
            "max_tokens": 350
        }
    ]
    
    feedback_collection = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n📝 [{i}/{len(questions)}] {q['topic'].upper()}")
        print("-" * 60)
        
        result = await ask_kobold_structured(
            q['prompt'],
            max_tokens=q.get('max_tokens', 400)
        )
        
        # Display result
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(json.dumps(result, indent=2)[:500])
            if len(json.dumps(result)) > 500:
                print("...")
        
        feedback_collection.append({
            "topic": q['topic'],
            "prompt": q['prompt'][:100] + "...",
            "result": result
        })
        
        await asyncio.sleep(2)
    
    # Save structured feedback
    with open('kobold_structured_feedback.json', 'w') as f:
        json.dump(feedback_collection, f, indent=2)
    
    print("\n" + "=" * 70)
    print("📊 STRUCTURED FEEDBACK COMPLETE")
    print("✅ Saved to 'kobold_structured_feedback.json'")
    
    return feedback_collection

async def get_focused_answer(question: str, context: str = "") -> str:
    """Get a single focused answer without chain-of-thought"""
    
    # Use instruction format to suppress reasoning
    prompt = f"""<|im_start|>system
You are a helpful assistant. Provide direct, concise answers without showing your reasoning process.
<|im_end|>
<|im_start|>user
{context}

{question}

Provide only the final answer, no reasoning.
<|im_end|>
<|im_start|>assistant
"""
    
    url = "http://localhost:5001/api/v1/generate"
    
    payload = {
        "prompt": prompt,
        "max_context_length": 4096,
        "max_length": 200,
        "temperature": 0.1,  # Very low for deterministic output
        "top_p": 0.5,
        "top_k": 10,
        "rep_pen": 1.1,
        "stop_sequence": ["<|im_end|>", "<|im_start|>", "\n\n"],
        "trim_stop": True
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if 'results' in data and len(data['results']) > 0:
                    return data['results'][0]['text'].strip()
                return "No response"
    except Exception as e:
        return f"Error: {e}"

async def test_direct_answers():
    """Test getting direct answers without reasoning chains"""
    
    print("🎯 TESTING DIRECT ANSWER MODE")
    print("=" * 70)
    
    tests = [
        "What is 2+2?",
        "Name three colors.",
        "Complete this: The sky is ___",
        "Is water wet? Yes or no.",
        "List 3 story elements in order: beginning, middle, ___"
    ]
    
    for question in tests:
        print(f"\nQ: {question}")
        answer = await get_focused_answer(question)
        print(f"A: {answer}")
        await asyncio.sleep(1)
    
    print("\n" + "=" * 70)
    print("✅ Direct answer test complete")

async def main():
    """Run structured feedback collection"""
    
    print("\n🚀 KOBOLDCPP STRUCTURED FEEDBACK SYSTEM")
    print("=" * 70)
    print("Testing different response modes...")
    print()
    
    # Test direct answers first
    await test_direct_answers()
    
    # Then get structured feedback
    await asyncio.sleep(2)
    await get_structured_feedback()
    
    print("\n✨ All tests complete!")

if __name__ == "__main__":
    asyncio.run(main())