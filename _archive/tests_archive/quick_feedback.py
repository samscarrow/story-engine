"""
Quick Architecture Feedback from LMStudio
Get focused feedback on specific architectural questions
"""

import asyncio
import aiohttp
import json

async def ask_lmstudio(question: str, max_tokens: int = 600) -> str:
    """Ask a specific question to LMStudio"""
    
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": "google/gemma-2-27b",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                return data['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

async def get_feedback():
    """Get architectural feedback"""
    
    print("🤖 ARCHITECTURAL FEEDBACK FROM GEMMA-2-27B")
    print("=" * 70)
    
    questions = [
        {
            "topic": "Missing Components",
            "question": "What are the 3 most important missing components in a story engine that has: story structure generation, scene crafting, character simulation, quality evaluation, and enhancement loops? Be specific and practical."
        },
        {
            "topic": "Character Psychology",
            "question": "For character simulation in stories, what psychological model would be better than just tracking 5 emotions (anger, fear, doubt, compassion, confidence)? Suggest a specific model with 8-10 dimensions."
        },
        {
            "topic": "Agent Architecture", 
            "question": "Design 5 specialized AI agents for collaborative story writing. For each agent, provide: name, primary responsibility, key decisions they make, and what they contribute to other agents."
        },
        {
            "topic": "Quality Metrics",
            "question": "What are 5 story quality metrics NOT commonly tracked but crucial for compelling narratives? Explain how to measure each one programmatically."
        },
        {
            "topic": "Branching Strategy",
            "question": "Besides high-tension moments, what are 3 other optimal points to branch narratives? Explain why each is valuable for story exploration."
        }
    ]
    
    feedback_collection = []
    
    for q in questions:
        print(f"\n📝 {q['topic'].upper()}")
        print("-" * 60)
        
        response = await ask_lmstudio(q['question'])
        print(response[:800] + "..." if len(response) > 800 else response)
        
        feedback_collection.append({
            "topic": q['topic'],
            "question": q['question'],
            "response": response
        })
        
        await asyncio.sleep(1)  # Pause between questions
    
    # Save all feedback
    with open('architecture_feedback_compiled.md', 'w') as f:
        f.write("# Architecture Feedback from Gemma-2-27b\n\n")
        
        for item in feedback_collection:
            f.write(f"## {item['topic']}\n\n")
            f.write(f"**Question:** {item['question']}\n\n")
            f.write(f"**Response:**\n\n{item['response']}\n\n")
            f.write("-" * 60 + "\n\n")
    
    print("\n" + "=" * 70)
    print("✅ Feedback saved to 'architecture_feedback_compiled.md'")

if __name__ == "__main__":
    asyncio.run(get_feedback())