"""
Get LMStudio Feedback on Architecture
Send architecture review request to LMStudio for analysis
"""

import asyncio
import aiohttp
import json

async def get_architecture_feedback():
    """Request feedback on the story engine architecture from LMStudio"""
    
    # Read the architecture review request
    with open('architecture_review_request.md', 'r') as f:
        architecture_description = f.read()
    
    # Prepare the prompt
    prompt = f"""As an AI systems architect with expertise in narrative generation and complex software systems, please review this story engine architecture and provide detailed feedback.

{architecture_description}

Please provide:
1. Specific answers to each of the 10 questions
2. Detailed recommendations for improvements
3. Identification of any architectural anti-patterns
4. Suggestions for novel components or approaches
5. Priority ranking of suggested changes

Focus on practical, implementable improvements that would significantly enhance the system's capability to generate compelling narratives."""

    # Call LMStudio
    url = "http://localhost:1234/v1/chat/completions"
    
    payload = {
        "model": "google/gemma-2-27b",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert AI systems architect specializing in narrative generation, story engines, and complex creative AI systems. Provide detailed, technical feedback with specific implementation suggestions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    print("🤖 REQUESTING ARCHITECTURE FEEDBACK FROM LMSTUDIO")
    print("=" * 60)
    print("\nSending architecture review request...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                feedback = data['choices'][0]['message']['content']
                
                print("\n" + "=" * 60)
                print("📝 LMSTUDIO ARCHITECTURE FEEDBACK")
                print("=" * 60)
                print(feedback)
                
                # Save feedback
                with open('architecture_feedback_from_lmstudio.md', 'w') as f:
                    f.write(f"# Architecture Feedback from LMStudio\n\n")
                    f.write(f"Model: {data.get('model', 'unknown')}\n\n")
                    f.write(f"## Feedback\n\n")
                    f.write(feedback)
                
                print("\n" + "=" * 60)
                print("✅ Feedback saved to 'architecture_feedback_from_lmstudio.md'")
                
                return feedback
                
    except Exception as e:
        print(f"\n❌ Error getting feedback: {e}")
        return None

async def get_specific_improvements():
    """Ask for specific implementation improvements"""
    
    prompt = """Based on the story engine architecture I described, please provide 5 specific, concrete improvements I could implement immediately. For each improvement:

1. Name the improvement
2. Explain why it's important
3. Describe how to implement it
4. Show a brief code structure or pseudocode example
5. Estimate the impact on story quality

Focus on improvements that would have the highest impact on narrative quality and character believability."""

    url = "http://localhost:1234/v1/chat/completions"
    
    payload = {
        "model": "google/gemma-2-27b",
        "messages": [
            {
                "role": "system",
                "content": "You are a narrative AI systems expert. Provide specific, actionable improvements with code examples."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.6,
        "max_tokens": 1500
    }
    
    print("\n🔧 REQUESTING SPECIFIC IMPROVEMENTS")
    print("=" * 60)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                improvements = data['choices'][0]['message']['content']
                
                print("\n" + "=" * 60)
                print("💡 SPECIFIC IMPROVEMENTS")
                print("=" * 60)
                print(improvements)
                
                # Save improvements
                with open('specific_improvements.md', 'w') as f:
                    f.write(f"# Specific Implementation Improvements\n\n")
                    f.write(improvements)
                
                print("\n✅ Improvements saved to 'specific_improvements.md'")
                
                return improvements
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

async def get_agent_architecture():
    """Ask about agent-based architecture design"""
    
    prompt = """For the story engine system, design a multi-agent architecture where different specialized agents handle different aspects of narrative generation. Please provide:

1. **Agent Roster**: List 5-7 specialized agents with clear roles
2. **Agent Personas**: Define the personality and expertise of each agent
3. **Communication Protocol**: How agents share information
4. **Collaboration Patterns**: How agents work together
5. **Conflict Resolution**: How disagreements between agents are resolved
6. **Quality Control**: How the agent collective ensures story quality

For each agent, specify:
- Name and role
- Expertise domain
- Input requirements
- Output products
- Interaction style with other agents

This should create a "writers room" simulation where specialized AI agents collaborate to create stories."""

    url = "http://localhost:1234/v1/chat/completions"
    
    payload = {
        "model": "google/gemma-2-27b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.8,
        "max_tokens": 1500
    }
    
    print("\n👥 REQUESTING AGENT ARCHITECTURE DESIGN")
    print("=" * 60)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                agent_design = data['choices'][0]['message']['content']
                
                print("\n" + "=" * 60)
                print("🎭 AGENT ARCHITECTURE DESIGN")
                print("=" * 60)
                print(agent_design)
                
                # Save design
                with open('agent_architecture_design.md', 'w') as f:
                    f.write(f"# Multi-Agent Architecture Design\n\n")
                    f.write(agent_design)
                
                print("\n✅ Agent design saved to 'agent_architecture_design.md'")
                
                return agent_design
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

async def main():
    """Run all feedback requests"""
    
    print("\n🚀 STORY ENGINE ARCHITECTURE REVIEW")
    print("=" * 60)
    print("Requesting comprehensive feedback from LMStudio...")
    print()
    
    # Get general architecture feedback
    feedback = await get_architecture_feedback()
    
    if feedback:
        # Get specific improvements
        await asyncio.sleep(2)  # Brief pause
        await get_specific_improvements()
        
        # Get agent architecture design
        await asyncio.sleep(2)
        await get_agent_architecture()
    
    print("\n" + "=" * 60)
    print("✨ Architecture review complete!")
    print("\nGenerated files:")
    print("  - architecture_feedback_from_lmstudio.md")
    print("  - specific_improvements.md")
    print("  - agent_architecture_design.md")

if __name__ == "__main__":
    asyncio.run(main())