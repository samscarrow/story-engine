"""
Multi-Character Interaction System
Complex character interactions and group dynamics simulation
"""

import logging
import asyncio
import json
import aiohttp
import time
from typing import Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of character relationships"""

    ALLY = "ally"
    ENEMY = "enemy"
    NEUTRAL = "neutral"
    SUPERIOR = "superior"
    SUBORDINATE = "subordinate"
    RIVAL = "rival"
    FRIEND = "friend"
    SPOUSE = "spouse"


@dataclass
class Character:
    """Enhanced character with relationship tracking"""

    id: str
    name: str
    role: str
    personality: List[str]
    goals: List[str]
    fears: List[str]
    relationships: Dict[str, RelationshipType] = field(default_factory=dict)
    emotional_state: Dict[str, float] = field(
        default_factory=lambda: {
            "anger": 0.2,
            "fear": 0.3,
            "confidence": 0.7,
            "doubt": 0.3,
            "compassion": 0.5,
        }
    )
    memories: List[str] = field(default_factory=list)

    def add_memory(self, memory: str):
        """Add a memory, keep last 5"""
        self.memories.append(memory)
        if len(self.memories) > 5:
            self.memories.pop(0)

    def get_relationship(self, other_id: str) -> RelationshipType:
        """Get relationship with another character"""
        return self.relationships.get(other_id, RelationshipType.NEUTRAL)

    def update_emotion(self, emotion: str, change: float):
        """Update an emotion, clamping between 0 and 1"""
        if emotion in self.emotional_state:
            self.emotional_state[emotion] = max(
                0, min(1, self.emotional_state[emotion] + change)
            )


class MultiCharacterEngine:
    """Engine for multi-character interactions"""

    def __init__(self, model: str = "google/gemma-2-27b"):
        self.model = model
        self.url = "http://localhost:1234/v1/chat/completions"
        self.characters: Dict[str, Character] = {}
        self.interaction_history = []

    def add_character(self, character: Character):
        """Add a character to the simulation"""
        self.characters[character.id] = character

    def set_relationship(
        self, char1_id: str, char2_id: str, relationship: RelationshipType
    ):
        """Set bidirectional relationship between characters"""
        if char1_id in self.characters:
            self.characters[char1_id].relationships[char2_id] = relationship
        if char2_id in self.characters:
            # Inverse relationships
            inverse_map = {
                RelationshipType.SUPERIOR: RelationshipType.SUBORDINATE,
                RelationshipType.SUBORDINATE: RelationshipType.SUPERIOR,
                RelationshipType.ENEMY: RelationshipType.ENEMY,
                RelationshipType.ALLY: RelationshipType.ALLY,
                RelationshipType.RIVAL: RelationshipType.RIVAL,
                RelationshipType.FRIEND: RelationshipType.FRIEND,
                RelationshipType.SPOUSE: RelationshipType.SPOUSE,
                RelationshipType.NEUTRAL: RelationshipType.NEUTRAL,
            }
            self.characters[char2_id].relationships[char1_id] = inverse_map.get(
                relationship, relationship
            )

    async def generate_response(
        self,
        character: Character,
        situation: str,
        other_characters: List[str],
        emphasis: str = "neutral",
    ):
        """Generate response for a character in a multi-character situation"""

        # Build context about other characters
        others_context = []
        for other_id in other_characters:
            if other_id in self.characters:
                other = self.characters[other_id]
                rel = character.get_relationship(other_id)
                others_context.append(f"{other.name} ({other.role}, {rel.value})")

        # Build system prompt
        system_prompt = f"""You are {character.name}, {character.role}.
Personality: {', '.join(character.personality)}
Goals: {', '.join(character.goals)}
Fears: {', '.join(character.fears)}
Current emotions: Anger={character.emotional_state['anger']:.1f}, Fear={character.emotional_state['fear']:.1f}, Confidence={character.emotional_state['confidence']:.1f}
Others present: {', '.join(others_context)}
Recent memories: {'; '.join(character.memories[-3:])}
Emphasis: {emphasis}

Respond with JSON only:
{{"dialogue": "what you say", "thought": "inner thoughts", "action": "physical action", "target": "who you're addressing/acting toward", "emotional_shift": {{"anger": 0, "fear": 0, "confidence": 0}}}}"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": situation},
            ],
            "temperature": 0.8,
            "max_tokens": 400,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as response:
                data = await response.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON
                try:
                    if content.strip().startswith("{"):
                        return json.loads(content)
                    else:
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        if json_start >= 0:
                            return json.loads(content[json_start:json_end])
                except (KeyError, ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Data extraction error: {e}")
                    return None

                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise

                return None

    async def run_interaction(self, scene: Dict):
        """Run a multi-character interaction scene"""

        print(f"\n{'='*80}")
        print(f"🎬 {scene['name'].upper()}")
        print(f"{'='*80}")
        print(f"📍 Location: {scene.get('location', 'Judgment Hall')}")
        print(f"📝 {scene['situation']}")
        print(
            f"👥 Characters: {', '.join([self.characters[cid].name for cid in scene['characters']])}"
        )

        if "setup" in scene:
            print(f"🎯 Focus: {scene['setup']}")

        print()

        results = []

        # Generate responses for each character
        for char_id in scene["characters"]:
            character = self.characters[char_id]
            other_chars = [c for c in scene["characters"] if c != char_id]

            print(f"\n💭 {character.name} ({character.role}):")
            print("-" * 40)

            try:
                start = time.time()
                response = await self.generate_response(
                    character,
                    scene["situation"],
                    other_chars,
                    scene.get("emphasis", {}).get(char_id, "neutral"),
                )
                elapsed = time.time() - start

                if response:
                    # Display response
                    print(
                        f"💬 \"{response.get('dialogue', 'N/A')[:150]}{'...' if len(response.get('dialogue', '')) > 150 else ''}\""
                    )
                    print(
                        f"🤔 *{response.get('thought', 'N/A')[:100]}{'...' if len(response.get('thought', '')) > 100 else ''}*"
                    )
                    print(
                        f"🎬 {response.get('action', 'N/A')[:80]}{'...' if len(response.get('action', '')) > 80 else ''}"
                    )

                    if "target" in response:
                        print(f"🎯 Directed at: {response['target']}")

                    # Update emotions
                    if "emotional_shift" in response and isinstance(
                        response["emotional_shift"], dict
                    ):
                        changes = []
                        for emotion, shift in response["emotional_shift"].items():
                            if isinstance(shift, (int, float)) and abs(shift) > 0.05:
                                character.update_emotion(emotion, shift)
                                changes.append(f"{emotion}:{shift:+.1f}")

                        if changes:
                            print(f"📊 Emotional shifts: {', '.join(changes)}")

                    # Add to character's memory
                    memory = f"{scene['name']}: {response.get('dialogue', '')[:50]}"
                    character.add_memory(memory)

                    # Store result
                    results.append(
                        {
                            "character": character.name,
                            "response": response,
                            "time": elapsed,
                        }
                    )

                    print(f"⏱️  Response time: {elapsed:.1f}s")
                else:
                    print("⚠️  Could not generate response")

            except Exception as e:
                print(f"❌ Error: {e}")

        # Store interaction
        self.interaction_history.append({"scene": scene["name"], "results": results})

        return results

    def display_character_states(self):
        """Display current state of all characters"""

        print(f"\n{'='*80}")
        print("📊 CHARACTER STATES")
        print(f"{'='*80}")

        for char_id, character in self.characters.items():
            print(f"\n👤 {character.name} ({character.role}):")
            print(f"   Anger: {character.emotional_state['anger']:.2f}")
            print(f"   Fear: {character.emotional_state['fear']:.2f}")
            print(f"   Confidence: {character.emotional_state['confidence']:.2f}")
            print(f"   Doubt: {character.emotional_state['doubt']:.2f}")

            if character.memories:
                print(f"   Recent memory: {character.memories[-1]}")


# Create characters for the trial
def create_trial_characters() -> Dict[str, Character]:
    """Create characters for the trial of Jesus"""

    characters = {}

    # Pontius Pilate
    pilate = Character(
        id="pilate",
        name="Pontius Pilate",
        role="Roman Prefect",
        personality=["pragmatic", "political", "conflicted", "authoritative"],
        goals=["maintain order", "avoid Caesar's wrath", "find truth"],
        fears=["rebellion", "losing position", "divine judgment"],
    )
    characters["pilate"] = pilate

    # Caiaphas - High Priest
    caiaphas = Character(
        id="caiaphas",
        name="Caiaphas",
        role="High Priest",
        personality=["cunning", "religious", "political", "determined"],
        goals=[
            "eliminate threat to authority",
            "preserve religious order",
            "maintain power",
        ],
        fears=["losing influence", "Roman intervention", "false prophets"],
        emotional_state={
            "anger": 0.6,
            "fear": 0.4,
            "confidence": 0.8,
            "doubt": 0.1,
            "compassion": 0.1,
        },
    )
    characters["caiaphas"] = caiaphas

    # Claudia Procula - Pilate's wife
    claudia = Character(
        id="claudia",
        name="Claudia Procula",
        role="Pilate's Wife",
        personality=["intuitive", "caring", "spiritual", "worried"],
        goals=["protect husband", "prevent injustice", "heed divine warnings"],
        fears=["divine retribution", "husband's downfall", "nightmares coming true"],
        emotional_state={
            "anger": 0.1,
            "fear": 0.8,
            "confidence": 0.3,
            "doubt": 0.2,
            "compassion": 0.9,
        },
    )
    characters["claudia"] = claudia

    # Barabbas - The Criminal
    barabbas = Character(
        id="barabbas",
        name="Barabbas",
        role="Zealot Prisoner",
        personality=["violent", "rebellious", "desperate", "cunning"],
        goals=["gain freedom", "fight Romans", "survive"],
        fears=["execution", "torture", "betrayal"],
        emotional_state={
            "anger": 0.9,
            "fear": 0.6,
            "confidence": 0.4,
            "doubt": 0.3,
            "compassion": 0.1,
        },
    )
    characters["barabbas"] = barabbas

    # Marcus - Centurion
    marcus = Character(
        id="marcus",
        name="Marcus",
        role="Roman Centurion",
        personality=["disciplined", "loyal", "observant", "uncomfortable"],
        goals=["follow orders", "maintain discipline", "protect Pilate"],
        fears=["disorder", "mutiny", "supernatural events"],
        emotional_state={
            "anger": 0.3,
            "fear": 0.4,
            "confidence": 0.6,
            "doubt": 0.5,
            "compassion": 0.3,
        },
    )
    characters["marcus"] = marcus

    return characters


async def run_trial_simulation():
    """Run multi-character trial simulation"""

    print("🏛️ THE TRIAL - MULTI-CHARACTER SIMULATION")
    print("=" * 80)

    # Create engine and characters
    engine = MultiCharacterEngine(model="google/gemma-2-27b")
    characters = create_trial_characters()

    # Add characters to engine
    for char in characters.values():
        engine.add_character(char)

    # Set relationships
    engine.set_relationship("pilate", "caiaphas", RelationshipType.RIVAL)
    engine.set_relationship("pilate", "claudia", RelationshipType.SPOUSE)
    engine.set_relationship("pilate", "marcus", RelationshipType.SUPERIOR)
    engine.set_relationship("caiaphas", "barabbas", RelationshipType.ENEMY)
    engine.set_relationship("marcus", "barabbas", RelationshipType.ENEMY)

    print("\n👥 CAST OF CHARACTERS:")
    for char in characters.values():
        print(f"  • {char.name}: {char.role}")

    # Define multi-character scenes
    scenes = [
        {
            "name": "The Accusation",
            "location": "Pilate's Judgment Hall",
            "situation": "Caiaphas brings Jesus before Pilate, accusing him of claiming to be King. Marcus stands guard. The morning sun illuminates the tense scene.",
            "characters": ["pilate", "caiaphas", "marcus"],
            "emphasis": {"pilate": "duty", "caiaphas": "power", "marcus": "neutral"},
            "setup": "Initial confrontation between religious and civil authority",
        },
        {
            "name": "Private Warning",
            "location": "Pilate's Private Chamber",
            "situation": "Claudia bursts in, interrupting Pilate's thoughts. She's visibly shaken from her dreams. Marcus guards the door, concerned about the breach of protocol.",
            "characters": ["pilate", "claudia", "marcus"],
            "emphasis": {"pilate": "doubt", "claudia": "fear", "marcus": "duty"},
            "setup": "Personal intervention in political matter",
        },
        {
            "name": "The Choice",
            "location": "Public Platform",
            "situation": "Pilate presents both Jesus and Barabbas to the crowd. Caiaphas stands ready to influence. The crowd must choose who to release.",
            "characters": ["pilate", "caiaphas", "barabbas"],
            "emphasis": {"pilate": "fear", "caiaphas": "power", "barabbas": "fear"},
            "setup": "The pivotal Passover pardoning decision",
        },
        {
            "name": "After the Decision",
            "location": "Private Chamber",
            "situation": "The decision is made. Pilate retreats with Claudia and Marcus. The weight of what happened hangs heavy. Outside, the crowd's noise continues.",
            "characters": ["pilate", "claudia", "marcus"],
            "emphasis": {
                "pilate": "neutral",
                "claudia": "compassion",
                "marcus": "doubt",
            },
            "setup": "Immediate aftermath and reflection",
        },
    ]

    # Run scenes
    print("\n🎬 Beginning multi-character simulation...")
    print(f"📍 {len(scenes)} scenes with multiple perspectives\n")

    for scene in scenes:
        await engine.run_interaction(scene)
        await asyncio.sleep(1)  # Pause between scenes

    # Display final states
    engine.display_character_states()

    # Relationship dynamics summary
    print(f"\n{'='*80}")
    print("🔄 RELATIONSHIP DYNAMICS")
    print(f"{'='*80}")

    print("\nKey Interactions:")
    print("  • Pilate ↔ Caiaphas: Political rivals, power struggle")
    print("  • Pilate ↔ Claudia: Marital tension over moral choice")
    print("  • Pilate → Marcus: Authority tested by events")
    print("  • Caiaphas vs Barabbas: Religious law vs rebellion")

    print("\n✨ Multi-character simulation complete!")


if __name__ == "__main__":
    print("\n🚀 MULTI-CHARACTER INTERACTION SYSTEM\n")
    asyncio.run(run_trial_simulation())
