"""
Get Architecture Feedback via KoboldCpp with Balanced Limits
Respects context window while allowing adequate responses
"""

import asyncio
import aiohttp
import json

async def ask_kobold(question: str, max_tokens: int = 800, min_tokens: int = 100) -> str:
    """
    Ask a question via KoboldCpp API with context-aware limits
    
    Args:
        question: The prompt to send
        max_tokens: Maximum tokens to generate (default 800 - safe for 4096 context)
        min_tokens: Minimum tokens to generate (default 100)
    """
    
    # Calculate safe limits based on context window
    context_limit = 4096
    prompt_tokens = len(question.split())  # Rough estimate
    safe_max_tokens = min(max_tokens, context_limit - prompt_tokens - 500)  # Leave 500 token buffer
    
    print(f"    [Prompt: ~{prompt_tokens} tokens, Max response: {safe_max_tokens} tokens]")
    
    # KoboldCpp API endpoint
    url = "http://localhost:5001/api/v1/generate"
    
    payload = {
        "prompt": question,
        "max_context_length": context_limit,
        "max_length": safe_max_tokens,  # Adjusted for safety
        "min_length": min_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,  # Limit vocabulary for better coherence
        "rep_pen": 1.1,
        "rep_pen_range": 320,
        "sampler_order": [6, 0, 1, 3, 4, 2, 5],
        "stop_sequence": ["###", "\n\n\n", "Question:", "END"],
        "trim_stop": True,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if 'results' in data and len(data['results']) > 0:
                    response_text = data['results'][0]['text']
                    actual_tokens = len(response_text.split())
                    print(f"    [Generated ~{actual_tokens} tokens]")
                    return response_text
                else:
                    return "No response generated"
    except Exception as e:
        return f"Error: {e}"

async def get_kobold_feedback_balanced():
    """Get architectural feedback with balanced token limits"""
    
    print("🤖 BALANCED ARCHITECTURAL FEEDBACK VIA KOBOLDCPP")
    print("=" * 70)
    print("Connecting to KoboldCpp at localhost:5001...")
    print("Context window: 4096 tokens")
    print("Response limits: Adjusted per question")
    print()
    
    # Test connection
    test_response = await ask_kobold("Respond: OK", max_tokens=10, min_tokens=1)
    print(f"Connection test: {test_response[:50]}")
    print()
    
    # Shorter, focused questions for better responses within context limits
    questions = [
        {
            "topic": "Critical Weakness",
            "question": """Story generation system with 5 layers:
1. Story Structure
2. Scene Crafting
3. Character Simulation
4. Quality Evaluation
5. Enhancement

What is the SINGLE most critical weakness and how to fix it? Be specific and concise.""",
            "max_tokens": 400
        },
        {
            "topic": "Missing Components",
            "question": """What are the TOP 3 missing components in a story engine?
Focus on components that would most improve narrative quality.
For each: name, purpose, implementation approach (brief).""",
            "max_tokens": 500
        },
        {
            "topic": "Character Model",
            "question": """Design a character psychology model with 8 dimensions.
Current: anger, fear, doubt, compassion, confidence.
Add 3 more critical dimensions and explain each briefly.
Format: Dimension name - description - range - impact on behavior.""",
            "max_tokens": 600
        },
        {
            "topic": "Agent Roles",
            "question": """Design 5 specialized story agents:
1. Plot Architect
2. Character Psychologist
3. ?
4. ?
5. ?

For each: name, main responsibility, key decision type.
Keep descriptions brief but specific.""",
            "max_tokens": 500
        },
        {
            "topic": "Quality Metrics",
            "question": """List 4 uncommon but crucial story quality metrics.
For each provide:
- Metric name
- What it measures (1 sentence)
- How to calculate (brief formula/approach)
Focus on: narrative momentum, character voice distinction, subtext, reader engagement.""",
            "max_tokens": 600
        },
        {
            "topic": "Branching Strategy",
            "question": """Besides high-tension moments, identify 3 optimal points for story branching:
1. ?
2. ?
3. ?

For each: when to branch, why it matters, detection method.
Be concise but complete.""",
            "max_tokens": 400
        },
        {
            "topic": "Feedback Loops",
            "question": """Design 3 critical feedback loops between story layers.
Format: Source Layer -> Target Layer -> Information -> Effect
Example: Quality Eval -> Scene Crafting -> Pacing issues -> Adjust scene length
Provide 3 specific, implementable feedback loops.""",
            "max_tokens": 400
        },
        {
            "topic": "Implementation Priority",
            "question": """Given limited resources, rank these improvements by impact:
A. Better character psychology
B. Scene-to-scene continuity
C. Multi-agent collaboration
D. Advanced quality metrics
E. Branching exploration

Provide ranking with brief justification for top 3.""",
            "max_tokens": 350
        }
    ]
    
    feedback_collection = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n📝 [{i}/{len(questions)}] {q['topic'].upper()}")
        print("-" * 60)
        print(f"Question: {q['question'][:100]}...")
        
        response = await ask_kobold(
            q['question'], 
            max_tokens=q.get('max_tokens', 500),
            min_tokens=50
        )
        
        # Display response
        print(f"\nResponse:\n{response[:800]}{'...' if len(response) > 800 else ''}")
        
        feedback_collection.append({
            "topic": q['topic'],
            "question": q['question'],
            "response": response,
            "length": len(response)
        })
        
        await asyncio.sleep(2)
    
    # Save feedback
    with open('kobold_balanced_feedback.md', 'w') as f:
        f.write("# Balanced Architecture Feedback via KoboldCpp\n\n")
        f.write("Context Limit: 4096 tokens\n")
        f.write("Response Strategy: Focused questions with appropriate limits\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        total_chars = sum(item['length'] for item in feedback_collection)
        f.write(f"- Total questions: {len(feedback_collection)}\n")
        f.write(f"- Total feedback: {total_chars:,} characters\n")
        f.write(f"- Average response: {total_chars//len(feedback_collection):,} characters\n\n")
        
        # Table of contents
        f.write("## Contents\n\n")
        for item in feedback_collection:
            f.write(f"- [{item['topic']}](#{item['topic'].lower().replace(' ', '-')})\n")
        f.write("\n---\n\n")
        
        # Full responses
        for item in feedback_collection:
            f.write(f"## {item['topic']}\n\n")
            f.write(f"**Length:** {item['length']} characters\n\n")
            f.write(f"**Question:**\n```\n{item['question']}\n```\n\n")
            f.write(f"**Response:**\n\n{item['response']}\n\n")
            f.write("-" * 60 + "\n\n")
    
    print("\n" + "=" * 70)
    print("📊 FEEDBACK COMPLETE")
    print(f"✅ Saved to 'kobold_balanced_feedback.md'")
    
    return feedback_collection

async def get_quick_improvements():
    """Get quick, actionable improvements"""
    
    print("\n💡 QUICK IMPROVEMENTS")
    print("=" * 70)
    
    prompt = """List 5 quick improvements for a story engine.
Format each as:
[Name]: [What to do] -> [Expected impact]

Be extremely concise. One line each. Focus on high-impact changes."""
    
    response = await ask_kobold(prompt, max_tokens=300, min_tokens=100)
    
    print(response)
    
    with open('quick_improvements.md', 'w') as f:
        f.write("# Quick Improvements\n\n")
        f.write(response)
    
    print("\n✅ Saved to 'quick_improvements.md'")
    
    return response

async def get_agent_personalities():
    """Get specific agent personality designs"""
    
    print("\n🎭 AGENT PERSONALITIES")
    print("=" * 70)
    
    prompt = """Design personalities for 5 story agents:
1. The Architect (structure)
2. The Director (scenes)
3. The Actor (characters)
4. The Critic (quality)
5. The Editor (revision)

For each, provide:
- Core personality trait (1-2 words)
- Working style (1 sentence)
- Favorite phrase or motto

Keep it brief but distinctive."""
    
    response = await ask_kobold(prompt, max_tokens=400, min_tokens=150)
    
    print(response)
    
    with open('agent_personalities.md', 'w') as f:
        f.write("# Story Agent Personalities\n\n")
        f.write(response)
    
    print("\n✅ Saved to 'agent_personalities.md'")
    
    return response

async def main():
    """Run balanced feedback session"""
    
    print("\n🚀 STORY ENGINE ARCHITECTURE REVIEW (BALANCED)")
    print("=" * 70)
    print("KoboldCpp API: localhost:5001")
    print("Strategy: Focused questions within context limits")
    print()
    
    # Get main feedback
    feedback = await get_kobold_feedback_balanced()
    
    if feedback and len(feedback) > 0:
        # Get quick improvements
        await asyncio.sleep(2)
        await get_quick_improvements()
        
        # Get agent personalities
        await asyncio.sleep(2)
        await get_agent_personalities()
    
    print("\n" + "=" * 70)
    print("✨ Architecture review complete!")
    print("\nGenerated files:")
    print("  📄 kobold_balanced_feedback.md - Main feedback")
    print("  💡 quick_improvements.md - Actionable improvements")
    print("  🎭 agent_personalities.md - Agent designs")
    
    if feedback:
        total = sum(item['length'] for item in feedback)
        print(f"\n📊 Total feedback: {total:,} characters across {len(feedback)} topics")

if __name__ == "__main__":
    asyncio.run(main())