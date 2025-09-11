"""
Complex Group Dynamics Simulation
Advanced multi-character scenarios with faction dynamics, alliances, and betrayals
"""

import asyncio
import json
import aiohttp
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Faction(Enum):
    """Character factions"""

    ROMAN = "Roman Authority"
    JEWISH_ELITE = "Jewish Religious Elite"
    ZEALOTS = "Revolutionary Zealots"
    DISCIPLES = "Jesus' Followers"
    NEUTRAL = "Unaffiliated"
    CROWD = "Common People"


class ActionType(Enum):
    """Types of character actions"""

    ACCUSE = "accuse"
    DEFEND = "defend"
    THREATEN = "threaten"
    PLEAD = "plead"
    CONSPIRE = "conspire"
    OBSERVE = "observe"
    INTERVENE = "intervene"
    RETREAT = "retreat"


@dataclass
class GroupCharacter:
    """Character with faction allegiance and action preferences"""

    id: str
    name: str
    role: str
    faction: Faction
    personality: List[str]
    motivations: Dict[str, float]  # motivation: strength (0-1)
    preferred_actions: List[ActionType]
    allies: Set[str] = field(default_factory=set)
    enemies: Set[str] = field(default_factory=set)
    trust_levels: Dict[str, float] = field(
        default_factory=dict
    )  # character_id: trust (0-1)
    emotional_state: Dict[str, float] = field(
        default_factory=lambda: {
            "anger": 0.3,
            "fear": 0.3,
            "confidence": 0.5,
            "tension": 0.5,
            "desperation": 0.2,
        }
    )
    influence_power: float = 0.5  # 0-1, ability to sway others

    def calculate_action_weight(
        self, action: ActionType, situation_tension: float
    ) -> float:
        """Calculate likelihood of taking an action based on personality and situation"""
        base_weight = 1.0 if action in self.preferred_actions else 0.3

        # Modify based on emotional state
        if action == ActionType.THREATEN:
            base_weight *= (
                self.emotional_state["anger"] + self.emotional_state["confidence"]
            ) / 2
        elif action == ActionType.RETREAT:
            base_weight *= self.emotional_state["fear"]
        elif action == ActionType.CONSPIRE:
            base_weight *= self.emotional_state["tension"]

        # Situation tension affects action choice
        if situation_tension > 0.7:
            if action in [ActionType.THREATEN, ActionType.ACCUSE]:
                base_weight *= 1.5
            elif action == ActionType.OBSERVE:
                base_weight *= 0.5

        return min(1.0, base_weight)


class GroupDynamicsEngine:
    """Advanced engine for complex group interactions"""

    def __init__(self, model: str = "google/gemma-2-27b"):
        self.model = model
        self.url = "http://localhost:1234/v1/chat/completions"
        self.characters: Dict[str, GroupCharacter] = {}
        self.faction_tensions: Dict[Tuple[Faction, Faction], float] = {}
        self.scene_history = []
        self.alliance_shifts = []

    def add_character(self, character: GroupCharacter):
        """Add character to simulation"""
        self.characters[character.id] = character

    def set_faction_tension(self, faction1: Faction, faction2: Faction, tension: float):
        """Set tension between factions (0-1)"""
        self.faction_tensions[(faction1, faction2)] = tension
        self.faction_tensions[(faction2, faction1)] = tension

    def calculate_group_dynamics(self, present_characters: List[str]) -> Dict:
        """Calculate current group dynamics"""
        factions_present = {}
        total_influence = 0
        tensions = []

        for char_id in present_characters:
            char = self.characters[char_id]
            faction = char.faction

            if faction not in factions_present:
                factions_present[faction] = []
            factions_present[faction].append(char_id)
            total_influence += char.influence_power

        # Calculate average tension
        for f1 in factions_present:
            for f2 in factions_present:
                if f1 != f2:
                    tension = self.faction_tensions.get((f1, f2), 0.5)
                    tensions.append(tension)

        avg_tension = sum(tensions) / len(tensions) if tensions else 0.5

        return {
            "factions": factions_present,
            "tension_level": avg_tension,
            "dominant_faction": (
                max(
                    factions_present.keys(),
                    key=lambda f: sum(
                        self.characters[c].influence_power for c in factions_present[f]
                    ),
                )
                if factions_present
                else None
            ),
        }

    async def generate_character_action(
        self,
        character: GroupCharacter,
        situation: str,
        dynamics: Dict,
        previous_actions: List[Dict],
    ) -> Dict:
        """Generate character's action in group context"""

        # Build context from previous actions
        action_context = []
        for action in previous_actions[-3:]:  # Last 3 actions
            action_context.append(
                f"{action['character']}: {action.get('action_type', 'spoke')}"
            )

        # Determine likely action type
        action_weights = {}
        for action_type in ActionType:
            weight = character.calculate_action_weight(
                action_type, dynamics["tension_level"]
            )
            action_weights[action_type.value] = weight

        # Build prompt
        system_prompt = f"""You are {character.name}, {character.role} of the {character.faction.value}.
Personality: {', '.join(character.personality)}
Motivations: {', '.join([f"{m}({v:.1f})" for m, v in character.motivations.items()])}
Current tension level: {dynamics['tension_level']:.1f}
Dominant faction present: {dynamics['dominant_faction'].value if dynamics['dominant_faction'] else 'None'}
Your emotional state: Anger={character.emotional_state['anger']:.1f}, Fear={character.emotional_state['fear']:.1f}, Desperation={character.emotional_state['desperation']:.1f}
Recent actions: {'; '.join(action_context)}

Choose your action type based on weights: {action_weights}

Respond with JSON:
{{"action_type": "one of: accuse/defend/threaten/plead/conspire/observe/intervene/retreat", "dialogue": "what you say", "thought": "inner thoughts", "gesture": "physical action", "target": "who you're addressing", "intended_effect": "what you hope to achieve", "emotional_shift": {{"anger": 0, "fear": 0, "desperation": 0}}}}"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": situation},
            ],
            "temperature": 0.8,
            "max_tokens": 500,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as response:
                data = await response.json()
                content = data["choices"][0]["message"]["content"]

                try:
                    if content.strip().startswith("{"):
                        return json.loads(content)
                    else:
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        if json_start >= 0:
                            return json.loads(content[json_start:json_end])
                except:
                    pass

                return None

    async def run_complex_scene(self, scene: Dict):
        """Run a complex multi-character scene with faction dynamics"""

        print(f"\n{'='*80}")
        print(f"🎬 {scene['name'].upper()}")
        print(f"{'='*80}")
        print(f"📍 {scene.get('location', 'Judgment Hall')}")
        print(f"📝 {scene['situation']}")

        # Calculate dynamics
        dynamics = self.calculate_group_dynamics(scene["characters"])

        print("\n📊 GROUP DYNAMICS:")
        print(f"  ⚡ Tension Level: {dynamics['tension_level']:.1%}")
        print(
            f"  👑 Dominant Faction: {dynamics['dominant_faction'].value if dynamics['dominant_faction'] else 'Balanced'}"
        )

        faction_counts = {}
        for char_id in scene["characters"]:
            faction = self.characters[char_id].faction
            faction_counts[faction] = faction_counts.get(faction, 0) + 1

        print(
            f"  🏛️ Factions Present: {', '.join([f'{f.value}({c})' for f, c in faction_counts.items()])}"
        )

        print("\n🎭 ACTIONS UNFOLD:")
        print("-" * 60)

        actions = []

        # Characters act in influence order (most influential first)
        ordered_chars = sorted(
            scene["characters"],
            key=lambda c: self.characters[c].influence_power,
            reverse=True,
        )

        for char_id in ordered_chars:
            character = self.characters[char_id]

            print(f"\n💭 {character.name} ({character.faction.value}):")

            try:
                start = asyncio.get_event_loop().time()
                response = await self.generate_character_action(
                    character, scene["situation"], dynamics, actions
                )
                elapsed = asyncio.get_event_loop().time() - start

                if response:
                    action_type = response.get("action_type", "speak").upper()

                    print(f"🎯 [{action_type}] → {response.get('target', 'all')}")
                    print(f"💬 \"{response.get('dialogue', 'N/A')[:120]}...\"")
                    print(f"🤔 *{response.get('thought', 'N/A')[:80]}...*")
                    print(f"🎬 {response.get('gesture', 'N/A')[:60]}")

                    if "intended_effect" in response:
                        print(f"📌 Intent: {response['intended_effect'][:50]}")

                    # Update emotional state
                    if "emotional_shift" in response:
                        for emotion, shift in response["emotional_shift"].items():
                            if emotion in character.emotional_state and isinstance(
                                shift, (int, float)
                            ):
                                old_val = character.emotional_state[emotion]
                                new_val = max(0, min(1, old_val + shift))
                                character.emotional_state[emotion] = new_val
                                if abs(shift) > 0.1:
                                    print(f"📊 {emotion}: {old_val:.1f}→{new_val:.1f}")

                    actions.append(
                        {
                            "character": character.name,
                            "action_type": response.get("action_type"),
                            "target": response.get("target"),
                            "dialogue": response.get("dialogue"),
                        }
                    )

                    print(f"⏱️  {elapsed:.1f}s")

            except Exception as e:
                print(f"❌ Error: {e}")

        # Store scene results
        self.scene_history.append(
            {"scene": scene["name"], "dynamics": dynamics, "actions": actions}
        )

        return actions

    def analyze_power_shifts(self):
        """Analyze how power dynamics shifted during scenes"""

        print(f"\n{'='*80}")
        print("⚖️ POWER DYNAMICS ANALYSIS")
        print(f"{'='*80}")

        # Track influence changes
        for char_id, character in self.characters.items():
            print(f"\n{character.name} ({character.faction.value}):")
            print(f"  Influence: {character.influence_power:.2f}")
            print(f"  Desperation: {character.emotional_state['desperation']:.2f}")
            print(
                f"  Allies: {', '.join(character.allies) if character.allies else 'None'}"
            )
            print(
                f"  Enemies: {', '.join(character.enemies) if character.enemies else 'None'}"
            )


# Create expanded cast of characters
def create_expanded_cast() -> Dict[str, GroupCharacter]:
    """Create a larger, more complex cast"""

    characters = {}

    # ROMAN FACTION
    characters["pilate"] = GroupCharacter(
        id="pilate",
        name="Pontius Pilate",
        role="Prefect",
        faction=Faction.ROMAN,
        personality=["pragmatic", "conflicted", "authoritative"],
        motivations={
            "maintain_order": 0.9,
            "survive_politically": 0.8,
            "find_truth": 0.4,
        },
        preferred_actions=[
            ActionType.OBSERVE,
            ActionType.INTERVENE,
            ActionType.THREATEN,
        ],
        influence_power=0.9,
    )

    characters["marcus"] = GroupCharacter(
        id="marcus",
        name="Marcus",
        role="Centurion",
        faction=Faction.ROMAN,
        personality=["loyal", "disciplined", "uncomfortable"],
        motivations={"follow_orders": 0.9, "protect_prefect": 0.8},
        preferred_actions=[ActionType.OBSERVE, ActionType.INTERVENE, ActionType.DEFEND],
        influence_power=0.5,
    )

    # JEWISH ELITE FACTION
    characters["caiaphas"] = GroupCharacter(
        id="caiaphas",
        name="Caiaphas",
        role="High Priest",
        faction=Faction.JEWISH_ELITE,
        personality=["cunning", "political", "ruthless"],
        motivations={"eliminate_threat": 0.9, "maintain_power": 0.8},
        preferred_actions=[ActionType.ACCUSE, ActionType.CONSPIRE, ActionType.THREATEN],
        influence_power=0.8,
    )

    characters["annas"] = GroupCharacter(
        id="annas",
        name="Annas",
        role="Former High Priest",
        faction=Faction.JEWISH_ELITE,
        personality=["manipulative", "experienced", "vengeful"],
        motivations={"restore_authority": 0.7, "destroy_blasphemer": 0.9},
        preferred_actions=[ActionType.CONSPIRE, ActionType.ACCUSE],
        influence_power=0.7,
    )

    # ZEALOT FACTION
    characters["barabbas"] = GroupCharacter(
        id="barabbas",
        name="Barabbas",
        role="Revolutionary",
        faction=Faction.ZEALOTS,
        personality=["violent", "desperate", "defiant"],
        motivations={"gain_freedom": 1.0, "fight_rome": 0.8},
        preferred_actions=[ActionType.THREATEN, ActionType.PLEAD, ActionType.RETREAT],
        influence_power=0.3,
    )

    characters["simon_zealot"] = GroupCharacter(
        id="simon_zealot",
        name="Simon",
        role="Zealot Leader",
        faction=Faction.ZEALOTS,
        personality=["passionate", "strategic", "uncompromising"],
        motivations={"free_judea": 0.9, "inspire_revolt": 0.8},
        preferred_actions=[ActionType.ACCUSE, ActionType.THREATEN, ActionType.CONSPIRE],
        influence_power=0.4,
    )

    # DISCIPLES FACTION
    characters["peter"] = GroupCharacter(
        id="peter",
        name="Simon Peter",
        role="Chief Disciple",
        faction=Faction.DISCIPLES,
        personality=["impulsive", "loyal", "frightened"],
        motivations={"protect_jesus": 0.9, "survive": 0.7},
        preferred_actions=[ActionType.DEFEND, ActionType.PLEAD, ActionType.RETREAT],
        influence_power=0.2,
        emotional_state={
            "anger": 0.4,
            "fear": 0.8,
            "confidence": 0.2,
            "tension": 0.9,
            "desperation": 0.7,
        },
    )

    characters["judas"] = GroupCharacter(
        id="judas",
        name="Judas Iscariot",
        role="Betrayer",
        faction=Faction.NEUTRAL,  # Changed allegiance
        personality=["guilt-ridden", "desperate", "conflicted"],
        motivations={"undo_betrayal": 0.8, "escape_guilt": 0.9},
        preferred_actions=[ActionType.RETREAT, ActionType.PLEAD, ActionType.OBSERVE],
        influence_power=0.1,
        emotional_state={
            "anger": 0.2,
            "fear": 0.9,
            "confidence": 0.1,
            "tension": 1.0,
            "desperation": 1.0,
        },
    )

    # CROWD REPRESENTATIVES
    characters["crowd_leader"] = GroupCharacter(
        id="crowd_leader",
        name="Josiah",
        role="Mob Leader",
        faction=Faction.CROWD,
        personality=["volatile", "passionate", "easily_swayed"],
        motivations={"see_justice": 0.6, "follow_majority": 0.8},
        preferred_actions=[ActionType.ACCUSE, ActionType.THREATEN],
        influence_power=0.5,
    )

    return characters


async def run_complex_trial():
    """Run complex group dynamics simulation"""

    print("⚔️ THE TRIAL - COMPLEX GROUP DYNAMICS")
    print("=" * 80)

    engine = GroupDynamicsEngine()
    characters = create_expanded_cast()

    # Add all characters
    for char in characters.values():
        engine.add_character(char)

    # Set faction tensions
    engine.set_faction_tension(Faction.ROMAN, Faction.JEWISH_ELITE, 0.6)
    engine.set_faction_tension(Faction.ROMAN, Faction.ZEALOTS, 0.9)
    engine.set_faction_tension(Faction.JEWISH_ELITE, Faction.ZEALOTS, 0.8)
    engine.set_faction_tension(Faction.JEWISH_ELITE, Faction.DISCIPLES, 1.0)
    engine.set_faction_tension(Faction.ROMAN, Faction.DISCIPLES, 0.5)
    engine.set_faction_tension(Faction.CROWD, Faction.DISCIPLES, 0.7)

    # Set initial relationships
    characters["pilate"].allies.add("marcus")
    characters["caiaphas"].allies.add("annas")
    characters["caiaphas"].enemies.add("peter")
    characters["peter"].enemies.add("judas")
    characters["barabbas"].enemies.add("marcus")

    print("\n🎭 DRAMATIS PERSONAE:")
    for faction in Faction:
        faction_chars = [c for c in characters.values() if c.faction == faction]
        if faction_chars:
            print(f"\n{faction.value}:")
            for char in faction_chars:
                print(
                    f"  • {char.name} - {char.role} (Influence: {char.influence_power:.1f})"
                )

    # Complex scenes
    scenes = [
        {
            "name": "The Confrontation",
            "location": "Praetorium Courtyard",
            "situation": "All factions converge. Caiaphas and Annas demand Jesus' death. Peter hides in the crowd. Judas watches from shadows. The mob grows restless.",
            "characters": [
                "pilate",
                "caiaphas",
                "annas",
                "peter",
                "crowd_leader",
                "marcus",
            ],
        },
        {
            "name": "The Bargain",
            "location": "Public Platform",
            "situation": "Pilate offers the Passover pardon. Barabbas and Jesus stand before the crowd. Each faction tries to sway the decision.",
            "characters": [
                "pilate",
                "caiaphas",
                "barabbas",
                "simon_zealot",
                "crowd_leader",
                "peter",
            ],
        },
        {
            "name": "The Betrayal's Echo",
            "location": "Temple Steps",
            "situation": "Judas attempts to return the silver. Caiaphas refuses. Peter confronts Judas. The crowd watches this secondary drama unfold.",
            "characters": ["judas", "caiaphas", "annas", "peter", "crowd_leader"],
        },
    ]

    print(f"\n🎬 Beginning complex simulation with {len(scenes)} scenes...")

    for scene in scenes:
        await engine.run_complex_scene(scene)
        await asyncio.sleep(1)

    # Analyze outcomes
    engine.analyze_power_shifts()

    print("\n✨ Complex group dynamics simulation complete!")


if __name__ == "__main__":
    print("\n🚀 COMPLEX GROUP DYNAMICS SYSTEM\n")
    asyncio.run(run_complex_trial())
