"""
Story Arc Engine
Generates narrative arcs, crafts scenes, and feeds them to character simulation
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import aiohttp

# Import our character simulation components
from character_simulation_engine_v2 import (
    CharacterState,
    EmotionalState,
    CharacterMemory,
    SimulationEngine,
    LMStudioLLM,
)


class StoryStructure(Enum):
    """Classic story structures"""

    THREE_ACT = "three_act"
    FIVE_ACT = "five_act"  # Freytag's pyramid
    HEROS_JOURNEY = "heros_journey"
    KISHOTENKETSU = "kishotenketsu"  # Japanese 4-act
    SAVE_THE_CAT = "save_the_cat"  # Blake Snyder's structure


class ConflictType(Enum):
    """Types of dramatic conflict"""

    PERSON_VS_PERSON = "character_conflict"
    PERSON_VS_SELF = "internal_conflict"
    PERSON_VS_SOCIETY = "social_conflict"
    PERSON_VS_NATURE = "natural_conflict"
    PERSON_VS_FATE = "fate_conflict"
    PERSON_VS_TECHNOLOGY = "tech_conflict"
    PERSON_VS_SUPERNATURAL = "supernatural_conflict"


class TensionCurve(Enum):
    """Tension progression patterns"""

    LINEAR_RISE = "linear_rise"
    EXPONENTIAL = "exponential"
    SAWTOOTH = "sawtooth"  # Rise and fall repeatedly
    PLATEAU = "plateau"  # Quick rise, maintain, resolution
    WAVE = "wave"  # Sinusoidal ups and downs


class SceneType(Enum):
    """Types of scenes"""

    EXPOSITION = "exposition"
    INCITING = "inciting_incident"
    RISING = "rising_action"
    CLIMAX = "climax"
    FALLING = "falling_action"
    RESOLUTION = "resolution"
    SETUP = "setup"
    PAYOFF = "payoff"
    REVERSAL = "reversal"
    REVELATION = "revelation"


@dataclass
class StoryTheme:
    """Thematic elements of the story"""

    primary_theme: str
    secondary_themes: List[str]
    moral_question: str
    symbolic_elements: Dict[str, str]  # symbol: meaning
    motifs: List[str]  # Recurring elements


@dataclass
class PlotPoint:
    """A key plot point in the story"""

    id: str
    name: str
    description: str
    scene_type: SceneType
    tension_level: float  # 0-1
    required_characters: List[str]
    optional_characters: List[str]
    location: str
    time_context: str  # morning, night, "three days later", etc.
    prerequisites: List[str]  # IDs of plot points that must happen first
    consequences: List[str]  # What changes after this point
    emotional_target: Dict[str, float]  # Target emotional states for key characters


@dataclass
class StoryArc:
    """Complete story arc definition"""

    id: str
    title: str
    structure: StoryStructure
    theme: StoryTheme
    conflict_type: ConflictType
    tension_curve: TensionCurve
    plot_points: List[PlotPoint]
    character_arcs: Dict[str, List[str]]  # character_id: [arc_stages]
    setting: Dict[str, Any]  # World-building details
    target_length: int  # Number of scenes
    tone: List[str]  # "dark", "hopeful", "tragic", etc.


@dataclass
class Scene:
    """A specific scene ready for simulation"""

    id: str
    arc_id: str
    sequence_number: int
    plot_point_id: str
    name: str
    location: str
    time_context: str
    situation: str  # The actual scene description
    characters_present: List[str]
    character_goals: Dict[str, str]  # character_id: goal_in_scene
    tension_level: float
    emphasis_per_character: Dict[str, str]  # character_id: emphasis
    expected_outcomes: List[str]
    dramatic_question: str  # What question does this scene answer?
    scene_type: SceneType
    pacing: str  # "slow", "moderate", "fast"
    sensory_details: Dict[str, str]  # sense: detail


class StoryArcEngine:
    """Engine for generating story arcs and scenes"""

    def __init__(self, llm_model: str = "google/gemma-2-27b"):
        self.llm_model = llm_model
        self.llm_url = "http://localhost:1234/v1/chat/completions"
        self.current_arc: Optional[StoryArc] = None
        self.scenes: List[Scene] = []
        self.story_state: Dict[str, Any] = {}

    async def call_llm(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.8
    ) -> Optional[Dict]:
        """Call LLM for story generation"""

        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": 800,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.llm_url, json=payload) as response:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]

                    # Parse JSON response
                    if "{" in content and "}" in content:
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        return json.loads(content[json_start:json_end])
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None

    def create_story_arc(
        self,
        title: str,
        structure: StoryStructure,
        theme_desc: str,
        conflict: ConflictType,
        characters: List[Dict],
        setting: Dict,
        target_scenes: int = 12,
    ) -> StoryArc:
        """Create a story arc programmatically"""

        # Define theme
        theme = StoryTheme(
            primary_theme=theme_desc,
            secondary_themes=[],
            moral_question="",
            symbolic_elements={},
            motifs=[],
        )

        # Generate plot points based on structure
        plot_points = self._generate_plot_points(
            structure, conflict, characters, target_scenes
        )

        # Define character arcs
        character_arcs = {}
        for char in characters:
            character_arcs[char["id"]] = self._generate_character_arc(structure)

        arc = StoryArc(
            id=f"arc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=title,
            structure=structure,
            theme=theme,
            conflict_type=conflict,
            tension_curve=TensionCurve.EXPONENTIAL,
            plot_points=plot_points,
            character_arcs=character_arcs,
            setting=setting,
            target_length=target_scenes,
            tone=["dramatic", "tense"],
        )

        self.current_arc = arc
        return arc

    def _generate_plot_points(
        self,
        structure: StoryStructure,
        conflict: ConflictType,
        characters: List[Dict],
        target_scenes: int,
    ) -> List[PlotPoint]:
        """Generate plot points based on story structure"""

        plot_points = []

        if structure == StoryStructure.THREE_ACT:
            # Act 1 - Setup (25%)
            plot_points.append(
                PlotPoint(
                    id="pp_opening",
                    name="Opening Image",
                    description="Establish the world and protagonist's normal",
                    scene_type=SceneType.EXPOSITION,
                    tension_level=0.2,
                    required_characters=[characters[0]["id"]],
                    optional_characters=[],
                    location="",
                    time_context="Beginning",
                    prerequisites=[],
                    consequences=["World established"],
                    emotional_target={characters[0]["id"]: {"confidence": 0.7}},
                )
            )

            plot_points.append(
                PlotPoint(
                    id="pp_inciting",
                    name="Inciting Incident",
                    description="The event that disrupts the normal",
                    scene_type=SceneType.INCITING,
                    tension_level=0.4,
                    required_characters=[c["id"] for c in characters[:2]],
                    optional_characters=[],
                    location="",
                    time_context="Shortly after",
                    prerequisites=["pp_opening"],
                    consequences=["Conflict introduced"],
                    emotional_target={characters[0]["id"]: {"doubt": 0.5, "fear": 0.3}},
                )
            )

            # Act 2 - Confrontation (50%)
            plot_points.append(
                PlotPoint(
                    id="pp_first_obstacle",
                    name="First Obstacle",
                    description="Initial attempt to resolve conflict fails",
                    scene_type=SceneType.RISING,
                    tension_level=0.5,
                    required_characters=[c["id"] for c in characters],
                    optional_characters=[],
                    location="",
                    time_context="Next day",
                    prerequisites=["pp_inciting"],
                    consequences=["Stakes raised"],
                    emotional_target={characters[0]["id"]: {"fear": 0.5}},
                )
            )

            plot_points.append(
                PlotPoint(
                    id="pp_midpoint",
                    name="Midpoint Reversal",
                    description="Major shift in understanding or stakes",
                    scene_type=SceneType.REVERSAL,
                    tension_level=0.6,
                    required_characters=[c["id"] for c in characters],
                    optional_characters=[],
                    location="",
                    time_context="Later",
                    prerequisites=["pp_first_obstacle"],
                    consequences=["New understanding", "Higher stakes"],
                    emotional_target={characters[0]["id"]: {"doubt": 0.7}},
                )
            )

            plot_points.append(
                PlotPoint(
                    id="pp_crisis",
                    name="Dark Night of Soul",
                    description="Lowest point, all seems lost",
                    scene_type=SceneType.RISING,
                    tension_level=0.8,
                    required_characters=[c["id"] for c in characters],
                    optional_characters=[],
                    location="",
                    time_context="That night",
                    prerequisites=["pp_midpoint"],
                    consequences=["Desperation peaks"],
                    emotional_target={
                        characters[0]["id"]: {"fear": 0.9, "confidence": 0.1}
                    },
                )
            )

            # Act 3 - Resolution (25%)
            plot_points.append(
                PlotPoint(
                    id="pp_climax",
                    name="Climax",
                    description="Final confrontation with the conflict",
                    scene_type=SceneType.CLIMAX,
                    tension_level=1.0,
                    required_characters=[c["id"] for c in characters],
                    optional_characters=[],
                    location="",
                    time_context="The moment of truth",
                    prerequisites=["pp_crisis"],
                    consequences=["Conflict resolved"],
                    emotional_target={},
                )
            )

            plot_points.append(
                PlotPoint(
                    id="pp_resolution",
                    name="Resolution",
                    description="New normal established",
                    scene_type=SceneType.RESOLUTION,
                    tension_level=0.3,
                    required_characters=[characters[0]["id"]],
                    optional_characters=[c["id"] for c in characters[1:]],
                    location="",
                    time_context="Afterward",
                    prerequisites=["pp_climax"],
                    consequences=["Story complete"],
                    emotional_target={characters[0]["id"]: {"confidence": 0.5}},
                )
            )

        elif structure == StoryStructure.FIVE_ACT:
            # Implement Freytag's pyramid
            # Exposition, Rising Action, Climax, Falling Action, Resolution
            pass

        elif structure == StoryStructure.HEROS_JOURNEY:
            # Implement Campbell's monomyth stages
            pass

        return plot_points

    def _generate_character_arc(self, structure: StoryStructure) -> List[str]:
        """Generate character arc stages"""

        if structure == StoryStructure.THREE_ACT:
            return [
                "Established in comfort",
                "Challenged and resistant",
                "Forced to adapt",
                "Moment of truth",
                "Transformed or broken",
            ]

        return ["Beginning", "Middle", "End"]

    async def generate_scene_details(
        self,
        plot_point: PlotPoint,
        characters: Dict[str, Any],
        previous_scenes: List[Scene],
    ) -> Scene:
        """Generate detailed scene from plot point using LLM"""

        # Build context from previous scenes
        context = (
            "Previous events: "
            + "; ".join(
                [
                    f"{s.name}: {s.expected_outcomes[0] if s.expected_outcomes else 'occurred'}"
                    for s in previous_scenes[-3:]
                ]
            )
            if previous_scenes
            else "This is the beginning."
        )

        system_prompt = f"""You are a narrative scene designer. Create a detailed scene.
Plot point: {plot_point.name} - {plot_point.description}
Scene type: {plot_point.scene_type.value}
Tension level: {plot_point.tension_level}
Required characters: {', '.join(plot_point.required_characters)}
{context}

Generate a scene with rich sensory details and clear dramatic purpose.

Respond with JSON:
{{
  "situation": "Detailed scene description with action and dialogue opportunities",
  "dramatic_question": "What key question does this scene answer?",
  "sensory_details": {{"sight": "", "sound": "", "smell": "", "touch": "", "atmosphere": ""}},
  "character_goals": {{"character_id": "specific goal in this scene"}},
  "expected_outcomes": ["outcome1", "outcome2"],
  "pacing": "slow/moderate/fast"
}}"""

        user_prompt = f"Create scene: {plot_point.name} at {plot_point.location or 'appropriate location'}"

        details = await self.call_llm(system_prompt, user_prompt)

        if not details:
            # Fallback generation
            details = {
                "situation": plot_point.description,
                "dramatic_question": "What happens next?",
                "sensory_details": {"atmosphere": "tense"},
                "character_goals": {
                    char_id: "navigate situation"
                    for char_id in plot_point.required_characters
                },
                "expected_outcomes": ["scene completes"],
                "pacing": "moderate",
            }

        scene = Scene(
            id=f"scene_{len(self.scenes):03d}",
            arc_id=self.current_arc.id if self.current_arc else "unknown",
            sequence_number=len(self.scenes),
            plot_point_id=plot_point.id,
            name=plot_point.name,
            location=plot_point.location or "unspecified",
            time_context=plot_point.time_context,
            situation=details.get("situation", plot_point.description),
            characters_present=plot_point.required_characters
            + plot_point.optional_characters,
            character_goals=details.get("character_goals", {}),
            tension_level=plot_point.tension_level,
            emphasis_per_character=self._calculate_emphasis(plot_point, characters),
            expected_outcomes=details.get("expected_outcomes", []),
            dramatic_question=details.get("dramatic_question", ""),
            scene_type=plot_point.scene_type,
            pacing=details.get("pacing", "moderate"),
            sensory_details=details.get("sensory_details", {}),
        )

        self.scenes.append(scene)
        return scene

    def _calculate_emphasis(
        self, plot_point: PlotPoint, characters: Dict[str, Any]
    ) -> Dict[str, str]:
        """Calculate emphasis for each character based on plot point"""

        emphasis_map = {
            SceneType.EXPOSITION: "neutral",
            SceneType.INCITING: "doubt",
            SceneType.RISING: "fear",
            SceneType.CLIMAX: "power",
            SceneType.FALLING: "compassion",
            SceneType.RESOLUTION: "neutral",
            SceneType.REVERSAL: "doubt",
            SceneType.REVELATION: "fear",
        }

        base_emphasis = emphasis_map.get(plot_point.scene_type, "neutral")

        # Customize per character based on their emotional targets
        result = {}
        for char_id in plot_point.required_characters:
            if char_id in plot_point.emotional_target:
                # Pick emphasis based on strongest emotion change
                emotions = plot_point.emotional_target[char_id]
                if "fear" in emotions and emotions["fear"] > 0.5:
                    result[char_id] = "fear"
                elif "doubt" in emotions and emotions["doubt"] > 0.5:
                    result[char_id] = "doubt"
                elif "confidence" in emotions and emotions["confidence"] < 0.3:
                    result[char_id] = "fear"
                else:
                    result[char_id] = base_emphasis
            else:
                result[char_id] = base_emphasis

        return result

    async def generate_full_story(
        self, arc: StoryArc, characters: Dict[str, Any]
    ) -> List[Scene]:
        """Generate all scenes for a story arc"""

        self.current_arc = arc
        self.scenes = []

        print(f"\n📚 GENERATING STORY: {arc.title}")
        print(f"Structure: {arc.structure.value}")
        print(f"Conflict: {arc.conflict_type.value}")
        print(f"Target scenes: {arc.target_length}")
        print()

        # Process plot points in order
        for i, plot_point in enumerate(arc.plot_points):
            print(
                f"🎬 Generating scene {i+1}/{len(arc.plot_points)}: {plot_point.name}"
            )

            # Check prerequisites
            if plot_point.prerequisites:
                completed = all(
                    any(s.plot_point_id == prereq for s in self.scenes)
                    for prereq in plot_point.prerequisites
                )
                if not completed:
                    print("  ⚠️ Prerequisites not met, skipping")
                    continue

            # Generate scene
            scene = await self.generate_scene_details(
                plot_point, characters, self.scenes
            )

            print(f"  ✅ Scene created: {scene.dramatic_question[:50]}...")
            print(f"  📊 Tension: {scene.tension_level:.1%}")
            print(f"  👥 Characters: {', '.join(scene.characters_present)}")

        return self.scenes


class IntegratedStoryEngine:
    """Integrates story arc generation with character simulation"""

    def __init__(self, llm_model: str = "google/gemma-2-27b"):
        self.story_engine = StoryArcEngine(llm_model)
        self.llm = LMStudioLLM(endpoint="http://localhost:1234/v1", model=llm_model)
        self.simulation_engine = SimulationEngine(
            llm_provider=self.llm, max_concurrent=1
        )
        self.characters: Dict[str, CharacterState] = {}
        self.simulation_results = []

    def create_character_for_story(self, char_def: Dict) -> CharacterState:
        """Create a CharacterState from story definition"""

        return CharacterState(
            id=char_def["id"],
            name=char_def["name"],
            backstory=char_def.get("backstory", {}),
            traits=char_def.get("traits", []),
            values=char_def.get("values", []),
            fears=char_def.get("fears", []),
            desires=char_def.get("desires", []),
            emotional_state=EmotionalState(
                anger=char_def.get("initial_emotions", {}).get("anger", 0.3),
                doubt=char_def.get("initial_emotions", {}).get("doubt", 0.3),
                fear=char_def.get("initial_emotions", {}).get("fear", 0.3),
                compassion=char_def.get("initial_emotions", {}).get("compassion", 0.5),
                confidence=char_def.get("initial_emotions", {}).get("confidence", 0.6),
            ),
            memory=CharacterMemory(),
            current_goal=char_def.get("goal", "Navigate the story"),
            internal_conflict=char_def.get("internal_conflict", ""),
        )

    async def run_story(
        self,
        title: str,
        structure: StoryStructure,
        theme: str,
        conflict: ConflictType,
        character_defs: List[Dict],
        setting: Dict,
    ) -> Dict:
        """Run complete story generation and simulation"""

        print(f"\n{'='*80}")
        print("🎭 INTEGRATED STORY GENERATION & SIMULATION")
        print(f"{'='*80}")

        # Create story arc
        arc = self.story_engine.create_story_arc(
            title=title,
            structure=structure,
            theme_desc=theme,
            conflict=conflict,
            characters=character_defs,
            setting=setting,
        )

        # Create characters for simulation
        for char_def in character_defs:
            self.characters[char_def["id"]] = self.create_character_for_story(char_def)

        # Generate all scenes
        scenes = await self.story_engine.generate_full_story(arc, self.characters)

        print(f"\n{'='*60}")
        print("🎬 RUNNING CHARACTER SIMULATIONS")
        print(f"{'='*60}")

        # Simulate each scene
        for scene in scenes:
            print(f"\n📖 SCENE {scene.sequence_number}: {scene.name}")
            print(f"📍 {scene.location} - {scene.time_context}")
            print(f"❓ {scene.dramatic_question}")
            print(f"📝 {scene.situation[:150]}...")

            scene_results = {}

            # Run simulation for each character in scene
            for char_id in scene.characters_present:
                if char_id in self.characters:
                    character = self.characters[char_id]

                    print(f"\n  💭 {character.name}:")

                    try:
                        # Add scene context to character memory
                        if scene.sequence_number > 0:
                            character.memory.add_event(
                                f"Scene {scene.sequence_number-1} completed"
                            )

                        # Run simulation
                        result = await self.simulation_engine.run_simulation(
                            character,
                            scene.situation,
                            emphasis=scene.emphasis_per_character.get(
                                char_id, "neutral"
                            ),
                            temperature=0.8,
                        )

                        response = result["response"]

                        if isinstance(response, dict):
                            print(f"    💬 \"{response.get('dialogue', '')[:100]}...\"")
                            print(f"    🤔 {response.get('thought', '')[:80]}...")

                            scene_results[char_id] = response

                            # Update character state based on emotional targets
                            if char_id in scene.plot_point_id:
                                # Apply plot point emotional targets
                                pass

                    except Exception as e:
                        print(f"    ❌ Simulation error: {e}")

            self.simulation_results.append({"scene": scene, "results": scene_results})

        # Generate summary
        return self._generate_story_summary()

    def _generate_story_summary(self) -> Dict:
        """Generate summary of the story simulation"""

        summary = {
            "title": (
                self.story_engine.current_arc.title
                if self.story_engine.current_arc
                else "Untitled"
            ),
            "total_scenes": len(self.simulation_results),
            "characters": list(self.characters.keys()),
            "emotional_journeys": {},
            "key_moments": [],
        }

        # Track emotional journeys
        for char_id, character in self.characters.items():
            summary["emotional_journeys"][char_id] = {
                "name": character.name,
                "final_state": {
                    "anger": character.emotional_state.anger,
                    "doubt": character.emotional_state.doubt,
                    "fear": character.emotional_state.fear,
                    "compassion": character.emotional_state.compassion,
                    "confidence": character.emotional_state.confidence,
                },
            }

        # Identify key moments (high tension scenes)
        for result in self.simulation_results:
            if result["scene"].tension_level > 0.7:
                summary["key_moments"].append(
                    {
                        "scene": result["scene"].name,
                        "tension": result["scene"].tension_level,
                        "type": result["scene"].scene_type.value,
                    }
                )

        return summary


async def demo_story_generation():
    """Demo the integrated story system"""

    # Define characters
    characters = [
        {
            "id": "protagonist",
            "name": "Sarah Chen",
            "backstory": {
                "role": "Whistleblower",
                "background": "Former corporate executive",
            },
            "traits": ["courageous", "conflicted", "intelligent"],
            "values": ["truth", "justice"],
            "fears": ["retaliation", "being silenced"],
            "desires": ["expose corruption", "protect family"],
            "initial_emotions": {"confidence": 0.5, "fear": 0.4},
        },
        {
            "id": "antagonist",
            "name": "Marcus Vale",
            "backstory": {"role": "CEO", "background": "Built empire on secrets"},
            "traits": ["ruthless", "charismatic", "paranoid"],
            "values": ["power", "control"],
            "fears": ["exposure", "losing empire"],
            "desires": ["silence threats", "maintain status"],
            "initial_emotions": {"confidence": 0.9, "anger": 0.3},
        },
        {
            "id": "ally",
            "name": "James Rivera",
            "backstory": {"role": "Journalist", "background": "Seeking truth"},
            "traits": ["persistent", "ethical", "resourceful"],
            "values": ["truth", "public good"],
            "fears": ["missing the story", "endangering sources"],
            "desires": ["break story", "protect Sarah"],
            "initial_emotions": {"confidence": 0.6, "doubt": 0.3},
        },
    ]

    # Define setting
    setting = {
        "location": "Corporate headquarters and city",
        "time_period": "Present day",
        "atmosphere": "Tense, modern thriller",
    }

    # Create and run story
    engine = IntegratedStoryEngine()

    summary = await engine.run_story(
        title="The Whistleblower's Gambit",
        structure=StoryStructure.THREE_ACT,
        theme="Truth vs Power",
        conflict=ConflictType.PERSON_VS_SOCIETY,
        character_defs=characters,
        setting=setting,
    )

    # Display summary
    print(f"\n{'='*80}")
    print("📊 STORY SUMMARY")
    print(f"{'='*80}")
    print(f"Title: {summary['title']}")
    print(f"Total scenes: {summary['total_scenes']}")
    print(f"Key moments: {len(summary['key_moments'])}")

    for moment in summary["key_moments"]:
        print(f"  • {moment['scene']} (Tension: {moment['tension']:.1%})")

    print("\n🎭 Character Emotional Journeys:")
    for char_id, journey in summary["emotional_journeys"].items():
        print(f"\n{journey['name']}:")
        for emotion, value in journey["final_state"].items():
            print(f"  {emotion}: {value:.2f}")

    print("\n✨ Story generation and simulation complete!")


if __name__ == "__main__":
    print("\n🚀 STORY ARC ENGINE - NARRATIVE GENERATION SYSTEM\n")
    asyncio.run(demo_story_generation())
