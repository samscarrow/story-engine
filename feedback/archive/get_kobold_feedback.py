"""
Get Architecture Feedback via KoboldCpp
Send architecture review to KoboldCpp API at port 5001
"""

import asyncio
import aiohttp
import json

async def ask_kobold(question: str, max_tokens: int = 600) -> str:
    """Ask a question via KoboldCpp API"""
    
    # KoboldCpp API endpoint
    url = "http://localhost:5001/api/v1/generate"
    
    # KoboldCpp expects different format
    payload = {
        "prompt": question,
        "max_context_length": 4096,
        "max_length": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "rep_pen": 1.1,
        "rep_pen_range": 320,
        "sampler_order": [6, 0, 1, 3, 4, 2, 5],
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                # KoboldCpp returns results in 'results' array
                if 'results' in data and len(data['results']) > 0:
                    return data['results'][0]['text']
                else:
                    return "No response generated"
    except Exception as e:
        return f"Error: {e}"

async def get_kobold_feedback():
    """Get architectural feedback via KoboldCpp"""
    
    print("🤖 ARCHITECTURAL FEEDBACK VIA KOBOLDCPP")
    print("=" * 70)
    print("Connecting to KoboldCpp at localhost:5001...")
    print()
    
    # Test connection first
    test_prompt = "Please respond with 'Connected' if you can read this."
    test_response = await ask_kobold(test_prompt, max_tokens=50)
    print(f"Connection test: {test_response[:100]}")
    print()
    
    questions = [
        {
            "topic": "Architecture Review",
            "question": """I have a story generation system with these layers:
1. Story Structure (generates plot points)
2. Scene Crafting (creates detailed scenes) 
3. Character Simulation (generates dialogue/actions)
4. Quality Evaluation (measures 8 metrics)
5. Enhancement (iterative improvement)

What is the most critical weakness in this architecture and how would you fix it?"""
        },
        {
            "topic": "Missing Components",
            "question": """For a story generation system, what are the 3 most important components missing from:
- Story structure generation
- Scene crafting
- Character simulation
- Quality evaluation
- Enhancement loops

Be specific about what's missing and why it matters."""
        },
        {
            "topic": "Character Model",
            "question": """Current character model tracks 5 emotions: anger, fear, doubt, compassion, confidence.

Design a better psychological model with 8-10 dimensions that would create more believable characters. List each dimension and explain why it's important."""
        },
        {
            "topic": "Agent Design",
            "question": """Design 5 specialized AI agents for collaborative story writing:

For each agent provide:
- Name
- Primary role
- Key decisions they make
- What information they need from other agents

Make them work like a writers room."""
        },
        {
            "topic": "Quality Metrics",
            "question": """What story quality metrics are rarely tracked but crucial? 

List 5 uncommon but important metrics and explain how to measure them programmatically."""
        }
    ]
    
    feedback_collection = []
    
    for q in questions:
        print(f"\n📝 {q['topic'].upper()}")
        print("-" * 60)
        print(f"Asking: {q['question'][:100]}...")
        
        response = await ask_kobold(q['question'])
        
        # Display response
        display_text = response[:600] + "..." if len(response) > 600 else response
        print(f"\nResponse:\n{display_text}")
        
        feedback_collection.append({
            "topic": q['topic'],
            "question": q['question'],
            "response": response
        })
        
        await asyncio.sleep(2)  # Pause between questions
    
    # Save all feedback
    with open('kobold_architecture_feedback.md', 'w') as f:
        f.write("# Architecture Feedback via KoboldCpp\n\n")
        f.write("API Endpoint: localhost:5001\n\n")
        
        for item in feedback_collection:
            f.write(f"## {item['topic']}\n\n")
            f.write(f"**Question:** {item['question']}\n\n")
            f.write(f"**Response:**\n\n{item['response']}\n\n")
            f.write("-" * 60 + "\n\n")
    
    print("\n" + "=" * 70)
    print("✅ Feedback saved to 'kobold_architecture_feedback.md'")
    
    return feedback_collection

async def get_specific_kobold_suggestions():
    """Get specific implementation suggestions"""
    
    print("\n💡 SPECIFIC IMPLEMENTATION SUGGESTIONS")
    print("=" * 70)
    
    prompt = """Give me 3 specific improvements for a story generation system. For each:

1. Name the improvement
2. Why it's important (1 sentence)
3. How to implement (2-3 sentences)
4. Expected impact on story quality

Be practical and specific."""
    
    response = await ask_kobold(prompt, max_tokens=800)
    
    print(response)
    
    # Save suggestions
    with open('kobold_suggestions.md', 'w') as f:
        f.write("# Specific Implementation Suggestions\n\n")
        f.write("From KoboldCpp API\n\n")
        f.write(response)
    
    print("\n✅ Suggestions saved to 'kobold_suggestions.md'")

# Alternative: Try OpenAI-compatible endpoint if available
async def try_openai_compatible():
    """Try OpenAI-compatible endpoint on KoboldCpp"""
    
    url = "http://localhost:5001/v1/completions"
    
    payload = {
        "prompt": "What makes a story compelling? List 3 key elements.",
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                print("\nOpenAI-compatible endpoint response:")
                print(json.dumps(data, indent=2)[:500])
                return data
    except Exception as e:
        print(f"OpenAI endpoint not available: {e}")
        return None

async def main():
    """Run all feedback requests"""
    
    print("\n🚀 STORY ENGINE ARCHITECTURE REVIEW")
    print("=" * 70)
    print("Using KoboldCpp API at localhost:5001")
    print()
    
    # Try main API
    feedback = await get_kobold_feedback()
    
    if feedback and len(feedback) > 0:
        # Get specific suggestions
        await asyncio.sleep(2)
        await get_specific_kobold_suggestions()
    
    # Try OpenAI-compatible endpoint
    await try_openai_compatible()
    
    print("\n" + "=" * 70)
    print("✨ Architecture review complete!")
    print("\nGenerated files:")
    print("  - kobold_architecture_feedback.md")
    print("  - kobold_suggestions.md")

if __name__ == "__main__":
    asyncio.run(main())