"""
Get Architecture Feedback via KoboldCpp with Extended Token Limits
Allows longer responses without forcing them
"""

import asyncio
import aiohttp
import json

async def ask_kobold(question: str, max_tokens: int = 2000, min_tokens: int = 100) -> str:
    """
    Ask a question via KoboldCpp API with flexible token limits
    
    Args:
        question: The prompt to send
        max_tokens: Maximum allowed tokens (default 2000)
        min_tokens: Minimum tokens to generate (default 100)
    """
    
    # KoboldCpp API endpoint
    url = "http://localhost:5001/api/v1/generate"
    
    # KoboldCpp parameters with flexible limits
    payload = {
        "prompt": question,
        "max_context_length": 4096,
        "max_length": max_tokens,  # Allow up to this many
        "min_length": min_tokens,   # But at least this many
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 0,  # 0 = disabled, allows all tokens
        "rep_pen": 1.1,
        "rep_pen_range": 320,
        "sampler_order": [6, 0, 1, 3, 4, 2, 5],
        "stop_sequence": ["###", "\n\n\n"],  # Natural stopping points
        "trim_stop": True,  # Remove stop sequence from output
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                # KoboldCpp returns results in 'results' array
                if 'results' in data and len(data['results']) > 0:
                    response_text = data['results'][0]['text']
                    # Report actual length used
                    actual_tokens = len(response_text.split())  # Rough estimate
                    print(f"    [Generated ~{actual_tokens} tokens]")
                    return response_text
                else:
                    return "No response generated"
    except Exception as e:
        return f"Error: {e}"

async def get_kobold_feedback_extended():
    """Get architectural feedback with extended responses"""
    
    print("🤖 EXTENDED ARCHITECTURAL FEEDBACK VIA KOBOLDCPP")
    print("=" * 70)
    print("Connecting to KoboldCpp at localhost:5001...")
    print("Token limits: 100-2000 (flexible)")
    print()
    
    # Test connection with small limit
    test_prompt = "Respond with 'Connected successfully' if you receive this."
    test_response = await ask_kobold(test_prompt, max_tokens=50, min_tokens=2)
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

Please provide a detailed analysis:
- What is the most critical weakness in this architecture?
- How would you fix it?
- What components are missing?
- How should the layers communicate?
- What feedback loops are needed?

Be thorough in your response.""",
            "max_tokens": 2500,  # Allow longer response for architecture review
            "min_tokens": 500
        },
        {
            "topic": "Missing Components",
            "question": """For a story generation system, identify missing components from these layers:
- Story structure generation
- Scene crafting
- Character simulation
- Quality evaluation
- Enhancement loops

List at least 5 missing components. For each:
1. Name the component
2. Why it's critical
3. Where it fits in the architecture
4. How it integrates with existing layers
5. Implementation approach

Provide complete details for each component.""",
            "max_tokens": 2000,
            "min_tokens": 400
        },
        {
            "topic": "Character Psychology Model",
            "question": """Current character model tracks only 5 emotions: anger, fear, doubt, compassion, confidence.

Design a comprehensive psychological model with 10-12 dimensions for believable characters. Include:

1. Core emotions (beyond the current 5)
2. Personality traits (Big Five or similar)
3. Motivational drives
4. Cognitive states
5. Social dynamics
6. Physical/energy states

For each dimension:
- Name and description
- Range and scale (e.g., 0-1, -1 to +1)
- How it affects behavior
- How it changes over time
- Interactions with other dimensions

Provide the complete model specification.""",
            "max_tokens": 2500,
            "min_tokens": 600
        },
        {
            "topic": "Multi-Agent Writers Room",
            "question": """Design a complete multi-agent system for collaborative story writing that simulates a writers room.

Create 7-8 specialized agents:

For each agent provide:
- Name and title
- Personality and working style
- Primary responsibilities
- Expertise areas
- Key decisions they make
- Information they need from others
- Information they provide to others
- How they interact in meetings
- Conflict points with other agents
- Unique contributions to the story

Also describe:
- How agents collaborate on a scene
- Conflict resolution mechanisms
- Quality control processes
- The creative review pipeline

Make this a complete, implementable design.""",
            "max_tokens": 3000,  # Allow very long response for complete design
            "min_tokens": 800
        },
        {
            "topic": "Advanced Quality Metrics",
            "question": """List 8-10 advanced story quality metrics that are rarely tracked but crucial for compelling narratives.

For each metric provide:
1. Metric name
2. What it measures (be specific)
3. Why it's important for story quality
4. How to calculate it programmatically (algorithm/formula)
5. Acceptable ranges and thresholds
6. How to improve it if it's low
7. Example of good vs bad scores

Focus on subtle but important aspects like:
- Subtext and implications
- Character voice distinctiveness
- Thematic resonance
- Narrative momentum
- Emotional authenticity
- Reader engagement predictors

Provide complete implementation details.""",
            "max_tokens": 2500,
            "min_tokens": 600
        },
        {
            "topic": "Branching and Alternative Narratives",
            "question": """Design a sophisticated branching system for narrative exploration.

Address:
1. When to branch (beyond just high-tension moments):
   - List 6-8 optimal branching points
   - Why each is valuable
   - How to detect them programmatically

2. How to evaluate branches:
   - Scoring different narrative paths
   - Predicting long-term consequences
   - Balancing exploration vs exploitation

3. Branch management:
   - Limiting explosion of possibilities
   - Merging similar branches
   - Pruning unpromising paths
   
4. Learning from branches:
   - What patterns indicate good branches
   - How to apply lessons to future stories
   
Provide specific algorithms and implementation strategies.""",
            "max_tokens": 2000,
            "min_tokens": 500
        }
    ]
    
    feedback_collection = []
    
    for q in questions:
        print(f"\n📝 {q['topic'].upper()}")
        print("-" * 60)
        print(f"Question preview: {q['question'][:150]}...")
        
        # Use question-specific token limits
        response = await ask_kobold(
            q['question'], 
            max_tokens=q.get('max_tokens', 2000),
            min_tokens=q.get('min_tokens', 100)
        )
        
        # Display response (show more since we have more tokens)
        display_limit = min(1200, len(response))
        display_text = response[:display_limit] + "..." if len(response) > display_limit else response
        print(f"\nResponse:\n{display_text}")
        
        feedback_collection.append({
            "topic": q['topic'],
            "question": q['question'],
            "response": response,
            "response_length": len(response)
        })
        
        await asyncio.sleep(3)  # Slightly longer pause for longer responses
    
    # Save all feedback with metadata
    with open('kobold_extended_feedback.md', 'w') as f:
        f.write("# Extended Architecture Feedback via KoboldCpp\n\n")
        f.write("API Endpoint: localhost:5001\n")
        f.write("Token Limits: Flexible (100-3000 per response)\n\n")
        
        # Add table of contents
        f.write("## Table of Contents\n\n")
        for item in feedback_collection:
            f.write(f"- [{item['topic']}](#{item['topic'].lower().replace(' ', '-')})\n")
        f.write("\n---\n\n")
        
        for item in feedback_collection:
            f.write(f"## {item['topic']}\n\n")
            f.write(f"**Response Length:** {item['response_length']} characters\n\n")
            f.write(f"**Question:** {item['question']}\n\n")
            f.write(f"**Response:**\n\n{item['response']}\n\n")
            f.write("-" * 60 + "\n\n")
    
    print("\n" + "=" * 70)
    print("📊 FEEDBACK SUMMARY")
    print("-" * 70)
    for item in feedback_collection:
        print(f"  {item['topic']}: {item['response_length']} chars")
    
    print("\n✅ Complete feedback saved to 'kobold_extended_feedback.md'")
    
    return feedback_collection

async def get_implementation_roadmap():
    """Get a specific implementation roadmap based on feedback"""
    
    print("\n🗺️ IMPLEMENTATION ROADMAP")
    print("=" * 70)
    
    prompt = """Based on the story generation architecture we discussed, create a prioritized implementation roadmap.

Provide 10 specific improvements in priority order. For each:

1. **Improvement Name**
2. **Priority Level** (Critical/High/Medium)
3. **Estimated Impact** (1-10 scale)
4. **Implementation Complexity** (Simple/Moderate/Complex)
5. **Dependencies** (what must be done first)
6. **Specific Steps** (3-5 bullet points)
7. **Success Metrics** (how to measure improvement)
8. **Code Structure** (brief pseudocode or class outline)

Focus on practical improvements that can be implemented incrementally.
Start with high-impact, low-complexity items.
Include both architectural and algorithmic improvements.

Provide a complete, actionable roadmap."""
    
    response = await ask_kobold(prompt, max_tokens=3000, min_tokens=1000)
    
    print(response[:1500] + "..." if len(response) > 1500 else response)
    
    # Save roadmap
    with open('implementation_roadmap.md', 'w') as f:
        f.write("# Story Engine Implementation Roadmap\n\n")
        f.write("Generated via KoboldCpp\n\n")
        f.write(response)
    
    print("\n✅ Roadmap saved to 'implementation_roadmap.md'")
    
    return response

async def main():
    """Run all feedback requests with extended token limits"""
    
    print("\n🚀 EXTENDED STORY ENGINE ARCHITECTURE REVIEW")
    print("=" * 70)
    print("Using KoboldCpp API at localhost:5001")
    print("Flexible token limits: 100-3000 tokens per response")
    print()
    
    # Get extended feedback
    feedback = await get_kobold_feedback_extended()
    
    if feedback and len(feedback) > 0:
        # Get implementation roadmap
        await asyncio.sleep(3)
        await get_implementation_roadmap()
    
    print("\n" + "=" * 70)
    print("✨ Extended architecture review complete!")
    print("\nGenerated files:")
    print("  - kobold_extended_feedback.md (full responses)")
    print("  - implementation_roadmap.md (prioritized improvements)")
    
    # Calculate total feedback received
    if feedback:
        total_chars = sum(item['response_length'] for item in feedback)
        print(f"\nTotal feedback received: {total_chars:,} characters")
        print(f"Average response length: {total_chars//len(feedback):,} characters")

if __name__ == "__main__":
    asyncio.run(main())