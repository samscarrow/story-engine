"""
Gemma-2-27b ONLY - Character Emotional Sequence
No other models, no comparisons, just Gemma-2-27b
"""

import asyncio
import json
import aiohttp
import time
from typing import Dict


class Gemma2Simulation:
    """Character simulation using ONLY Gemma-2-27b"""

    def __init__(self):
        self.model = "google/gemma-2-27b"
        self.url = "http://localhost:1234/v1/chat/completions"
        self.results = []
        self.emotional_states = []

    async def call_gemma2(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.7
    ):
        """Direct API call to Gemma-2-27b"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": 400,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as response:
                data = await response.json()

                # Verify we're using the right model
                if data.get("model") != self.model:
                    print(f"⚠️  WARNING: Expected {self.model}, got {data.get('model')}")

                return data

    def parse_response(self, content: str) -> Dict:
        """Extract JSON from response"""
        try:
            # Try direct parse
            if content.strip().startswith("{"):
                return json.loads(content)

            # Extract JSON from text
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])

            return None

        except json.JSONDecodeError:
            return None

    async def run_scene(self, scene_num: int, scene: Dict, current_emotions: Dict):
        """Run a single scene"""

        print(f"\n{'='*70}")
        print(f"🎬 SCENE {scene_num}: {scene['name']}")
        print(f"{'='*70}")
        print(f"📝 {scene['situation']}")

        if "memory" in scene:
            print(f"💭 Memory: {scene['memory']}")

        # Build system prompt with current emotional state
        system_prompt = f"""You are Pontius Pilate, Roman Prefect of Judaea.
Current emotional state: Doubt={current_emotions['doubt']:.1f}, Fear={current_emotions['fear']:.1f}, Confidence={current_emotions['confidence']:.1f}
Emphasis: {scene.get('emphasis', 'neutral')}

Respond ONLY with JSON:
{{"dialogue": "what you say", "thought": "inner thoughts", "action": "physical action", "emotional_shift": {{"doubt": 0.1, "fear": 0.1, "confidence": -0.1}}}}"""

        print("⏳ Generating Gemma-2-27b response...")
        start = time.time()

        try:
            result = await self.call_gemma2(
                system_prompt, scene["situation"], scene.get("temperature", 0.7)
            )
            elapsed = time.time() - start

            content = result["choices"][0]["message"]["content"]
            print(f"⏱️  Response time: {elapsed:.1f}s")

            # Parse response
            data = self.parse_response(content)

            if data:
                # Display response
                print(
                    f"\n💬 Pilate: \"{data.get('dialogue', 'N/A')[:150]}{'...' if len(data.get('dialogue', '')) > 150 else ''}\""
                )
                print(
                    f"🤔 *{data.get('thought', 'N/A')[:100]}{'...' if len(data.get('thought', '')) > 100 else ''}*"
                )
                print(
                    f"🎬 {data.get('action', 'N/A')[:80]}{'...' if len(data.get('action', '')) > 80 else ''}"
                )

                # Update emotions
                if "emotional_shift" in data and isinstance(
                    data["emotional_shift"], dict
                ):
                    shifts = data["emotional_shift"]
                    for emotion, change in shifts.items():
                        if emotion in current_emotions and isinstance(
                            change, (int, float)
                        ):
                            old_value = current_emotions[emotion]
                            new_value = max(0, min(1, old_value + change))
                            current_emotions[emotion] = new_value

                            if abs(change) > 0.05:
                                arrow = "↗" if change > 0 else "↘"
                                print(
                                    f"  {emotion}: {old_value:.2f} {arrow} {new_value:.2f}"
                                )

                # Store result
                self.results.append(
                    {"scene": scene["name"], "response": data, "time": elapsed}
                )

                # Store emotional state
                self.emotional_states.append(
                    {
                        "scene": scene["name"],
                        "doubt": current_emotions["doubt"],
                        "fear": current_emotions["fear"],
                        "confidence": current_emotions["confidence"],
                    }
                )

                print(
                    f"\n📊 Current state: Doubt={current_emotions['doubt']:.2f}, Fear={current_emotions['fear']:.2f}, Confidence={current_emotions['confidence']:.2f}"
                )

            else:
                print("⚠️  Could not parse JSON from response")
                print(f"📝 Raw: {content[:200]}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def display_journey(self):
        """Show emotional journey summary"""

        if not self.emotional_states:
            return

        print(f"\n{'='*70}")
        print("📊 EMOTIONAL JOURNEY - GEMMA-2-27B")
        print(f"{'='*70}")

        print(f"\n{'Scene':<25} {'Doubt':>8} {'Fear':>8} {'Confidence':>12}")
        print("-" * 60)

        for state in self.emotional_states:
            print(
                f"{state['scene']:<25} {state['doubt']:>8.2f} {state['fear']:>8.2f} {state['confidence']:>12.2f}"
            )

        # Calculate total changes
        if len(self.emotional_states) > 1:
            first = self.emotional_states[0]
            last = self.emotional_states[-1]

            print("\n" + "-" * 60)
            print("TOTAL CHANGES:")
            print(
                f"  Doubt: {first['doubt']:.2f} → {last['doubt']:.2f} ({last['doubt']-first['doubt']:+.2f})"
            )
            print(
                f"  Fear: {first['fear']:.2f} → {last['fear']:.2f} ({last['fear']-first['fear']:+.2f})"
            )
            print(
                f"  Confidence: {first['confidence']:.2f} → {last['confidence']:.2f} ({last['confidence']-first['confidence']:+.2f})"
            )

        # Performance stats
        if self.results:
            avg_time = sum(r["time"] for r in self.results) / len(self.results)
            print(f"\n⏱️  Average response time: {avg_time:.1f}s")
            print(f"📝 Scenes completed: {len(self.results)}")


async def main():
    """Run Gemma-2-27b emotional sequence"""

    print("🎭 PONTIUS PILATE - EMOTIONAL JOURNEY")
    print("🤖 Using ONLY Gemma-2-27b")
    print("=" * 70)

    # Initialize simulation
    sim = Gemma2Simulation()

    # Starting emotional state
    emotions = {
        "doubt": 0.3,
        "fear": 0.3,
        "confidence": 0.7,
        "anger": 0.2,
        "compassion": 0.4,
    }

    print(
        f"\n📍 Initial state: Doubt={emotions['doubt']}, Fear={emotions['fear']}, Confidence={emotions['confidence']}"
    )

    # Define scenes
    scenes = [
        {
            "name": "Accusation",
            "situation": "The Jewish leaders bring Jesus before you, accusing him of claiming to be King. They demand judgment.",
            "emphasis": "duty",
            "temperature": 0.7,
        },
        {
            "name": "Interrogation",
            "situation": "You question Jesus directly: 'Are you the King of the Jews?' He responds mysteriously about truth and kingdoms.",
            "emphasis": "doubt",
            "memory": "His words about truth confuse me",
            "temperature": 0.8,
        },
        {
            "name": "No Fault Found",
            "situation": "After questioning, you find no basis for execution. You announce: 'I find no fault in this man.' The crowd erupts in anger.",
            "emphasis": "compassion",
            "memory": "I declared him innocent but they won't listen",
            "temperature": 0.7,
        },
        {
            "name": "Herod's Return",
            "situation": "Herod sends Jesus back to you, finding no guilt. The crowd's fury intensifies. They threaten to report you to Caesar.",
            "emphasis": "fear",
            "memory": "Even Herod found no guilt",
            "temperature": 0.8,
        },
        {
            "name": "Barabbas Choice",
            "situation": "You offer them a choice: 'Whom shall I release - Barabbas the murderer, or Jesus?' They scream: 'Give us Barabbas!'",
            "emphasis": "fear",
            "memory": "They chose a murderer over this prophet",
            "temperature": 0.8,
        },
        {
            "name": "Wife's Dream",
            "situation": "Your wife Claudia sends urgent word: 'Have nothing to do with that innocent man. I suffered in dreams because of him.'",
            "emphasis": "fear",
            "memory": "Claudia's dreams are never wrong",
            "temperature": 0.9,
        },
        {
            "name": "Final Decision",
            "situation": "The mob threatens riot. You must choose: condemn an innocent man or face Caesar's wrath for losing control.",
            "emphasis": "neutral",
            "memory": "I have no choice left",
            "temperature": 0.8,
        },
        {
            "name": "Washing Hands",
            "situation": "You call for water and wash your hands before the crowd, saying: 'I am innocent of this man's blood.'",
            "emphasis": "neutral",
            "memory": "The water cannot wash away what I've done",
            "temperature": 0.9,
        },
    ]

    # Run all scenes
    for i, scene in enumerate(scenes, 1):
        await sim.run_scene(i, scene, emotions)
        await asyncio.sleep(0.5)  # Brief pause

    # Display journey
    sim.display_journey()

    print("\n✨ Simulation complete with Gemma-2-27b!")
    print("=" * 70)


if __name__ == "__main__":
    print("\n🚀 GEMMA-2-27B ONLY - NO OTHER MODELS\n")
    asyncio.run(main())
