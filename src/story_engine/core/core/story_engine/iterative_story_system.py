"""
Iterative Story Development System
Enhances story generation through feedback loops, evaluation, and refinement
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback for story iteration"""

    PACING = "pacing"
    CONSISTENCY = "consistency"
    EMOTIONAL_IMPACT = "emotional_impact"
    DIALOGUE_QUALITY = "dialogue_quality"
    PLOT_COHERENCE = "plot_coherence"
    CHARACTER_DEVELOPMENT = "character_development"
    TENSION_CURVE = "tension_curve"
    THEMATIC_RESONANCE = "thematic_resonance"


class RevisionStrategy(Enum):
    """Strategies for revising story elements"""

    AMPLIFY = "amplify"  # Make stronger
    SOFTEN = "soften"  # Make gentler
    ACCELERATE = "accelerate"  # Speed up pacing
    DECELERATE = "decelerate"  # Slow down
    DEEPEN = "deepen"  # Add complexity
    SIMPLIFY = "simplify"  # Remove complexity
    REDIRECT = "redirect"  # Change direction
    MAINTAIN = "maintain"  # Keep as is


@dataclass
class StoryMetrics:
    """Metrics for evaluating story quality"""

    pacing_score: float = 0.0  # 0-1
    consistency_score: float = 0.0
    emotional_range: float = 0.0
    dialogue_naturalism: float = 0.0
    plot_coherence: float = 0.0
    character_arc_completion: float = 0.0
    tension_effectiveness: float = 0.0
    thematic_clarity: float = 0.0

    def overall_score(self) -> float:
        """Calculate weighted overall score"""
        weights = {
            "pacing": 0.15,
            "consistency": 0.20,
            "emotional": 0.15,
            "dialogue": 0.10,
            "plot": 0.15,
            "character": 0.10,
            "tension": 0.10,
            "theme": 0.05,
        }

        return (
            self.pacing_score * weights["pacing"]
            + self.consistency_score * weights["consistency"]
            + self.emotional_range * weights["emotional"]
            + self.dialogue_naturalism * weights["dialogue"]
            + self.plot_coherence * weights["plot"]
            + self.character_arc_completion * weights["character"]
            + self.tension_effectiveness * weights["tension"]
            + self.thematic_clarity * weights["theme"]
        )


@dataclass
class Feedback:
    """Feedback on a story element"""

    element_id: str
    feedback_type: FeedbackType
    severity: float  # 0-1, how important
    description: str
    suggested_revision: RevisionStrategy
    specific_notes: str


@dataclass
class StoryVersion:
    """A version of the story with its evaluation"""

    version_id: str
    timestamp: datetime
    scenes: List[Any]
    metrics: StoryMetrics
    feedback: List[Feedback]
    parent_version: Optional[str] = None
    revision_notes: str = ""


@dataclass
class BranchPoint:
    """A point where story can branch"""

    scene_id: str
    decision_point: str
    options: List[Dict[str, Any]]
    selection_criteria: Dict[str, float]  # criteria: weight
    consequences: Dict[str, List[str]]  # option: [consequences]


class IterativeStoryEngine:
    """Engine for iterative story development with feedback loops"""

    def __init__(self, model: str = "google/gemma-2-27b"):
        self.model = model
        self.url = "http://localhost:1234/v1/chat/completions"
        self.story_versions: List[StoryVersion] = []
        self.current_version: Optional[StoryVersion] = None
        self.branch_points: List[BranchPoint] = []
        self.iteration_limit = 5
        self.quality_threshold = 0.75
        self.feedback_history: Dict[str, List[Feedback]] = defaultdict(list)
        # Optional POML integration
        self.use_poml = False
        self.poml_adapter = None
        try:
            from story_engine.core.common.config import load_config

            cfg = load_config("config.yaml")
            self.use_poml = bool(cfg.get("simulation", {}).get("use_poml", False))
            from story_engine.poml.lib.poml_integration import StoryEnginePOMLAdapter

            self.poml_adapter = StoryEnginePOMLAdapter()
        except Exception:
            self.poml_adapter = None

    async def call_llm(
        self, prompt: str, context: str = "", temperature: float = 0.8
    ) -> str:
        """LLM call for various story operations"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 600,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload) as response:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM error: {e}")
            return ""

    def evaluate_story(
        self, scenes: List[Any], character_responses: Dict[str, Any]
    ) -> StoryMetrics:
        """Evaluate story quality across multiple dimensions"""

        metrics = StoryMetrics()

        # Pacing evaluation
        scene_lengths = [len(s.get("situation", "")) for s in scenes]
        pacing_variance = (
            np.std(scene_lengths) / np.mean(scene_lengths) if scene_lengths else 0
        )
        metrics.pacing_score = max(0, 1 - pacing_variance)

        # Consistency check
        character_names = set()
        for scene in scenes:
            if "characters_present" in scene:
                character_names.update(scene["characters_present"])

        consistency_issues = 0
        for scene in scenes:
            # Check if characters appear/disappear randomly
            if "characters_present" in scene:
                if len(set(scene["characters_present"]) - character_names) > 0:
                    consistency_issues += 1

        metrics.consistency_score = (
            max(0, 1 - (consistency_issues / len(scenes))) if scenes else 0
        )

        # Emotional range
        emotions_expressed = set()
        for responses in character_responses.values():
            if isinstance(responses, dict):
                for response in responses.values():
                    if "emotional_shift" in response:
                        emotions_expressed.update(response["emotional_shift"].keys())

        metrics.emotional_range = min(
            1.0, len(emotions_expressed) / 5
        )  # Expect ~5 emotions

        # Dialogue naturalism (simple heuristic)
        dialogue_scores = []
        for responses in character_responses.values():
            if isinstance(responses, dict):
                for response in responses.values():
                    dialogue = response.get("dialogue", "")
                    # Check for varied sentence length, natural flow
                    if dialogue:
                        sentences = dialogue.split(".")
                        if len(sentences) > 1:
                            dialogue_scores.append(0.8)
                        elif "?" in dialogue or "!" in dialogue:
                            dialogue_scores.append(0.7)
                        else:
                            dialogue_scores.append(0.5)

        metrics.dialogue_naturalism = (
            np.mean(dialogue_scores) if dialogue_scores else 0.5
        )

        # Plot coherence
        tension_levels = [s.get("tension", 0.5) for s in scenes]
        if tension_levels:
            # Check for appropriate tension curve
            expected_curve = [0.2, 0.4, 0.6, 0.8, 0.3]  # Classic structure
            if len(tension_levels) >= len(expected_curve):
                actual = tension_levels[: len(expected_curve)]
                difference = sum(abs(a - e) for a, e in zip(actual, expected_curve))
                metrics.plot_coherence = max(0, 1 - (difference / len(expected_curve)))
            else:
                metrics.plot_coherence = 0.6  # Partial credit

        # Character arc completion
        character_growth = 0
        for char_id, responses in character_responses.items():
            if isinstance(responses, list) and len(responses) > 1:
                # Check if character changed over time
                first_response = responses[0] if responses else {}
                last_response = responses[-1] if responses else {}
                if first_response.get("thought") != last_response.get("thought"):
                    character_growth += 1

        total_chars = len(character_responses)
        metrics.character_arc_completion = (
            character_growth / total_chars if total_chars > 0 else 0
        )

        # Tension effectiveness
        if tension_levels:
            # Should build to climax then resolve
            max_tension_idx = tension_levels.index(max(tension_levels))
            ideal_position = len(tension_levels) * 0.75  # Climax at 75% point
            position_score = 1 - abs(max_tension_idx - ideal_position) / len(
                tension_levels
            )
            metrics.tension_effectiveness = position_score

        # Thematic clarity (simplified)
        metrics.thematic_clarity = 0.7  # Default, would need theme analysis

        return metrics

    def generate_feedback(
        self, scenes: List[Any], metrics: StoryMetrics
    ) -> List[Feedback]:
        """Generate specific feedback based on metrics"""

        feedback_items = []

        # Pacing feedback
        if metrics.pacing_score < 0.6:
            feedback_items.append(
                Feedback(
                    element_id="pacing",
                    feedback_type=FeedbackType.PACING,
                    severity=0.8,
                    description="Pacing is uneven across scenes",
                    suggested_revision=(
                        RevisionStrategy.ACCELERATE
                        if metrics.pacing_score < 0.3
                        else RevisionStrategy.MAINTAIN
                    ),
                    specific_notes="Consider balancing scene lengths and intensity",
                )
            )

        # Consistency feedback
        if metrics.consistency_score < 0.7:
            feedback_items.append(
                Feedback(
                    element_id="consistency",
                    feedback_type=FeedbackType.CONSISTENCY,
                    severity=0.9,
                    description="Character or plot inconsistencies detected",
                    suggested_revision=RevisionStrategy.SIMPLIFY,
                    specific_notes="Ensure characters maintain consistent presence and motivation",
                )
            )

        # Emotional impact feedback
        if metrics.emotional_range < 0.5:
            feedback_items.append(
                Feedback(
                    element_id="emotions",
                    feedback_type=FeedbackType.EMOTIONAL_IMPACT,
                    severity=0.6,
                    description="Limited emotional range in character responses",
                    suggested_revision=RevisionStrategy.AMPLIFY,
                    specific_notes="Add more varied emotional responses and reactions",
                )
            )

        # Dialogue feedback
        if metrics.dialogue_naturalism < 0.6:
            feedback_items.append(
                Feedback(
                    element_id="dialogue",
                    feedback_type=FeedbackType.DIALOGUE_QUALITY,
                    severity=0.5,
                    description="Dialogue feels unnatural or stilted",
                    suggested_revision=RevisionStrategy.DEEPEN,
                    specific_notes="Vary sentence structure and add character voice",
                )
            )

        # Plot coherence feedback
        if metrics.plot_coherence < 0.6:
            feedback_items.append(
                Feedback(
                    element_id="plot",
                    feedback_type=FeedbackType.PLOT_COHERENCE,
                    severity=0.8,
                    description="Plot structure needs strengthening",
                    suggested_revision=RevisionStrategy.REDIRECT,
                    specific_notes="Ensure clear cause-and-effect between scenes",
                )
            )

        # Tension feedback
        if metrics.tension_effectiveness < 0.5:
            feedback_items.append(
                Feedback(
                    element_id="tension",
                    feedback_type=FeedbackType.TENSION_CURVE,
                    severity=0.7,
                    description="Tension curve is not effective",
                    suggested_revision=RevisionStrategy.AMPLIFY,
                    specific_notes="Build tension gradually to climax, then resolve",
                )
            )

        return feedback_items

    async def revise_scene(self, scene: Dict, feedback: List[Feedback]) -> Dict:
        """Revise a scene based on feedback"""
        # Build revision instructions
        revision_notes = []
        for fb in feedback:
            if fb.suggested_revision == RevisionStrategy.AMPLIFY:
                revision_notes.append(f"Make {fb.element_id} stronger and more intense")
            elif fb.suggested_revision == RevisionStrategy.SOFTEN:
                revision_notes.append(f"Make {fb.element_id} gentler and subtler")
            elif fb.suggested_revision == RevisionStrategy.ACCELERATE:
                revision_notes.append("Speed up the pacing")
            elif fb.suggested_revision == RevisionStrategy.DEEPEN:
                revision_notes.append(f"Add more complexity to {fb.element_id}")
            elif fb.suggested_revision == RevisionStrategy.REDIRECT:
                revision_notes.append(f"Change the direction of {fb.element_id}")

        # Summarize evaluation and focus
        evaluation_text = ("; ".join(rn for rn in revision_notes)) or "No evaluation"
        focus = (
            ", ".join(
                sorted(set(fb.feedback_type.value.replace("_", " ") for fb in feedback))
            )
            or "general"
        )

        if self.use_poml and self.poml_adapter:
            # Include simple structured metrics approximation from feedback counts
            metrics = {}
            for fb in feedback:
                k = fb.feedback_type.value
                metrics[k] = max(metrics.get(k, 0.0), fb.severity)
            prompt = self.poml_adapter.get_enhancement_prompt(
                content=scene.get("situation", "No situation"),
                evaluation_text=evaluation_text,
                focus=focus,
                metrics=metrics,
            )
        else:
            prompt = f"""Revise this scene based on feedback:

Original scene: {scene.get('situation', 'No situation')}

Feedback to address:
{chr(10).join(revision_notes)}

Provide an improved version that addresses these issues while maintaining story continuity.

Revised scene:"""

        revised_situation = await self.call_llm(prompt, temperature=0.7)

        # Update scene
        revised_scene = scene.copy()
        revised_scene["situation"] = (
            revised_situation if revised_situation else scene.get("situation")
        )
        revised_scene["revision_notes"] = revision_notes

        return revised_scene

    def identify_branch_points(self, scenes: List[Dict]) -> List[BranchPoint]:
        """Identify potential branching points in the narrative"""

        branch_points = []

        for i, scene in enumerate(scenes):
            tension = scene.get("tension", 0)

            # High tension scenes are good branch points
            if tension > 0.6:
                options = [
                    {"choice": "escalate", "description": "Increase conflict"},
                    {"choice": "de-escalate", "description": "Seek resolution"},
                    {"choice": "redirect", "description": "Introduce new element"},
                ]

                branch = BranchPoint(
                    scene_id=f"scene_{i}",
                    decision_point=f"High tension moment in {scene.get('name', 'scene')}",
                    options=options,
                    selection_criteria={
                        "tension": 0.5,
                        "coherence": 0.3,
                        "surprise": 0.2,
                    },
                    consequences={
                        "escalate": ["Higher conflict", "Character crisis"],
                        "de-escalate": ["Temporary peace", "Character reflection"],
                        "redirect": ["New subplot", "Unexpected ally/enemy"],
                    },
                )
                branch_points.append(branch)

        return branch_points

    async def explore_branch(
        self, branch: BranchPoint, choice: str, current_scenes: List[Dict]
    ) -> List[Dict]:
        """Explore a narrative branch"""

        consequences = branch.consequences.get(choice, [])

        prompt = f"""Given this story branch:
Scene: {branch.decision_point}
Choice taken: {choice}
Consequences: {', '.join(consequences)}

Generate the next 2 scenes that follow from this choice.

Provide JSON with two scene descriptions:
{{"scene1": {{"situation": "...", "tension": 0.X}}, "scene2": {{"situation": "...", "tension": 0.Y}}}}"""

        response = await self.call_llm(prompt, temperature=0.8)

        # Parse response
        try:
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                data = json.loads(response[json_start:json_end])

                new_scenes = []
                for key in ["scene1", "scene2"]:
                    if key in data:
                        new_scenes.append(data[key])

                return current_scenes + new_scenes
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error: {e}")
            pass

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

        return current_scenes

    async def iterate_story(
        self,
        initial_scenes: List[Dict],
        character_responses: Dict[str, Any],
        max_iterations: int = 3,
    ) -> StoryVersion:
        """Iteratively improve story based on feedback"""

        print("\n🔄 ITERATIVE STORY DEVELOPMENT")
        print("=" * 60)

        current_scenes = initial_scenes
        best_version = None
        best_score = 0

        for iteration in range(max_iterations):
            print(f"\n📝 Iteration {iteration + 1}/{max_iterations}")

            # Evaluate current version
            metrics = self.evaluate_story(current_scenes, character_responses)
            overall_score = metrics.overall_score()

            print(f"  📊 Overall Score: {overall_score:.2%}")
            print(f"     Pacing: {metrics.pacing_score:.2f}")
            print(f"     Consistency: {metrics.consistency_score:.2f}")
            print(f"     Emotional Range: {metrics.emotional_range:.2f}")
            print(f"     Plot Coherence: {metrics.plot_coherence:.2f}")

            # Generate feedback
            feedback = self.generate_feedback(current_scenes, metrics)

            if feedback:
                print(f"\n  📋 Feedback ({len(feedback)} items):")
                for fb in feedback[:3]:  # Show top 3
                    print(f"     • {fb.description} (Severity: {fb.severity:.1f})")

            # Create version record
            version = StoryVersion(
                version_id=f"v{iteration}_{datetime.now().strftime('%H%M%S')}",
                timestamp=datetime.now(),
                scenes=current_scenes.copy(),
                metrics=metrics,
                feedback=feedback,
                parent_version=(
                    self.current_version.version_id if self.current_version else None
                ),
                revision_notes=f"Iteration {iteration + 1}",
            )

            self.story_versions.append(version)

            # Check if good enough
            if overall_score >= self.quality_threshold:
                print(
                    f"\n  ✅ Quality threshold reached ({overall_score:.2%} >= {self.quality_threshold:.2%})"
                )
                best_version = version
                break

            # Keep best version
            if overall_score > best_score:
                best_score = overall_score
                best_version = version

            # Stop if no significant feedback
            if not feedback or all(fb.severity < 0.3 for fb in feedback):
                print("\n  ✅ Minor feedback only, stopping iterations")
                break

            # Revise based on feedback
            if iteration < max_iterations - 1:
                print("\n  🔧 Applying revisions...")

                # Focus on most severe feedback
                critical_feedback = sorted(
                    feedback, key=lambda f: f.severity, reverse=True
                )[:2]

                revised_scenes = []
                for i, scene in enumerate(current_scenes):
                    # Check if this scene needs revision
                    scene_feedback = [
                        fb
                        for fb in critical_feedback
                        if fb.element_id in ["pacing", "consistency", "plot"]
                    ]

                    if (
                        scene_feedback and i < len(current_scenes) // 2
                    ):  # Focus on first half
                        revised_scene = await self.revise_scene(scene, scene_feedback)
                        revised_scenes.append(revised_scene)
                        print(f"     Revised scene {i + 1}")
                    else:
                        revised_scenes.append(scene)

                current_scenes = revised_scenes
                self.current_version = version

        return best_version or version

    def generate_alternatives(
        self, scene: Dict, num_alternatives: int = 3
    ) -> List[Dict]:
        """Generate alternative versions of a scene"""

        alternatives = []

        variations = [
            {"tone": "darker", "modifier": "Add more tension and danger"},
            {"tone": "lighter", "modifier": "Add hope and humor"},
            {"tone": "mysterious", "modifier": "Add ambiguity and questions"},
            {"tone": "action-packed", "modifier": "Add physical conflict"},
            {"tone": "emotional", "modifier": "Focus on internal struggles"},
        ]

        for i in range(min(num_alternatives, len(variations))):
            alt = scene.copy()
            alt["variation"] = variations[i]
            alt["id"] = f"{scene.get('id', 'scene')}_alt{i}"
            alternatives.append(alt)

        return alternatives

    async def parallel_exploration(
        self, scenes: List[Dict], branch_points: List[BranchPoint]
    ) -> Dict[str, List[Dict]]:
        """Explore multiple narrative paths in parallel"""

        print("\n🌳 PARALLEL NARRATIVE EXPLORATION")
        print("=" * 60)

        narrative_paths = {}

        for i, branch in enumerate(branch_points[:2]):  # Limit to 2 branches
            print(f"\n🔀 Branch Point {i + 1}: {branch.decision_point}")

            for option in branch.options:
                choice = option["choice"]
                print(f"  → Exploring: {choice} - {option['description']}")

                # Explore this branch
                branched_scenes = await self.explore_branch(branch, choice, scenes)

                path_id = f"branch_{i}_{choice}"
                narrative_paths[path_id] = branched_scenes

                print(f"     Generated {len(branched_scenes) - len(scenes)} new scenes")

        return narrative_paths

    def select_best_path(
        self,
        narrative_paths: Dict[str, List[Dict]],
        character_responses: Dict[str, Any],
    ) -> Tuple[str, List[Dict]]:
        """Select the best narrative path based on metrics"""

        best_path_id = None
        best_score = 0
        best_scenes = []

        print("\n🎯 EVALUATING NARRATIVE PATHS")
        print("-" * 40)

        for path_id, scenes in narrative_paths.items():
            metrics = self.evaluate_story(scenes, character_responses)
            score = metrics.overall_score()

            print(f"  {path_id}: {score:.2%}")

            if score > best_score:
                best_score = score
                best_path_id = path_id
                best_scenes = scenes

        print(f"\n  ✅ Best path: {best_path_id} ({best_score:.2%})")

        return best_path_id, best_scenes


class AdaptiveStorySystem:
    """Complete adaptive story generation system"""

    def __init__(self, model: str = "google/gemma-2-27b"):
        self.iterative_engine = IterativeStoryEngine(model)
        self.model = model
        self.learning_history: List[Dict] = []

    async def generate_and_refine(
        self,
        title: str,
        premise: str,
        characters: List[Dict],
        target_quality: float = 0.75,
        max_iterations: int = 3,
    ) -> Dict:
        """Generate and iteratively refine a story"""

        print(f"\n{'='*80}")
        print(f"🎨 ADAPTIVE STORY GENERATION: {title}")
        print(f"{'='*80}")
        print(f"Premise: {premise}")
        print(f"Target Quality: {target_quality:.0%}")
        print(f"Max Iterations: {max_iterations}")

        # Initial generation (simplified for demo)
        initial_scenes = [
            {
                "name": "Opening",
                "situation": f"Opening scene of {title}",
                "tension": 0.2,
                "characters_present": [c["id"] for c in characters],
            },
            {
                "name": "Development",
                "situation": "Story develops with conflict",
                "tension": 0.5,
                "characters_present": [c["id"] for c in characters],
            },
            {
                "name": "Climax",
                "situation": "Climactic confrontation",
                "tension": 0.8,
                "characters_present": [c["id"] for c in characters],
            },
            {
                "name": "Resolution",
                "situation": "Story concludes",
                "tension": 0.3,
                "characters_present": [c["id"] for c in characters[:1]],
            },
        ]

        # Mock character responses for evaluation
        character_responses = {}
        for char in characters:
            character_responses[char["id"]] = [
                {
                    "dialogue": "Opening dialogue",
                    "thought": "Initial thought",
                    "emotional_shift": {"fear": 0.1},
                },
                {
                    "dialogue": "Development dialogue",
                    "thought": "Growing concern",
                    "emotional_shift": {"fear": 0.3, "anger": 0.2},
                },
                {
                    "dialogue": "Climax dialogue",
                    "thought": "Critical moment",
                    "emotional_shift": {"fear": 0.5, "determination": 0.4},
                },
                {
                    "dialogue": "Resolution dialogue",
                    "thought": "Final reflection",
                    "emotional_shift": {"relief": 0.3},
                },
            ]

        # Iterative refinement
        best_version = await self.iterative_engine.iterate_story(
            initial_scenes, character_responses, max_iterations
        )

        # Identify branch points
        branch_points = self.iterative_engine.identify_branch_points(
            best_version.scenes
        )

        if branch_points:
            print(f"\n🔍 Found {len(branch_points)} potential branch points")

            # Explore alternatives
            narrative_paths = await self.iterative_engine.parallel_exploration(
                best_version.scenes, branch_points
            )

            # Select best path
            if narrative_paths:
                best_path_id, best_scenes = self.iterative_engine.select_best_path(
                    narrative_paths, character_responses
                )

                # Update best version with best path
                best_version.scenes = best_scenes

        # Generate summary
        summary = {
            "title": title,
            "final_version": best_version.version_id,
            "total_iterations": len(self.iterative_engine.story_versions),
            "final_score": best_version.metrics.overall_score(),
            "metrics": {
                "pacing": best_version.metrics.pacing_score,
                "consistency": best_version.metrics.consistency_score,
                "emotional_range": best_version.metrics.emotional_range,
                "plot_coherence": best_version.metrics.plot_coherence,
            },
            "feedback_addressed": len(best_version.feedback),
            "branch_points_explored": len(branch_points) if branch_points else 0,
        }

        # Store learning
        self.learning_history.append(
            {
                "title": title,
                "iterations": len(self.iterative_engine.story_versions),
                "final_score": best_version.metrics.overall_score(),
                "successful_revisions": [
                    f.suggested_revision.value
                    for f in best_version.feedback
                    if f.severity < 0.5
                ],
            }
        )

        return summary

    def analyze_learning(self) -> Dict:
        """Analyze what the system has learned"""

        if not self.learning_history:
            return {}

        analysis = {
            "stories_generated": len(self.learning_history),
            "average_iterations": np.mean(
                [h["iterations"] for h in self.learning_history]
            ),
            "average_final_score": np.mean(
                [h["final_score"] for h in self.learning_history]
            ),
            "common_successful_revisions": defaultdict(int),
        }

        for history in self.learning_history:
            for revision in history.get("successful_revisions", []):
                analysis["common_successful_revisions"][revision] += 1

        return analysis


async def demo_iterative_system():
    """Demonstrate the iterative story development system"""

    characters = [
        {
            "id": "hero",
            "name": "Elena Vasquez",
            "role": "protagonist",
            "traits": ["determined", "haunted", "brilliant"],
        },
        {
            "id": "mentor",
            "name": "Professor Chen",
            "role": "guide",
            "traits": ["wise", "secretive", "protective"],
        },
        {
            "id": "rival",
            "name": "Marcus Stone",
            "role": "antagonist",
            "traits": ["ambitious", "cunning", "desperate"],
        },
    ]

    system = AdaptiveStorySystem()

    # Generate first story
    summary1 = await system.generate_and_refine(
        title="The Quantum Paradox",
        premise="A scientist discovers time travel but each use creates dangerous alternate realities",
        characters=characters,
        target_quality=0.75,
        max_iterations=3,
    )

    print(f"\n{'='*60}")
    print("📊 STORY GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Final Version: {summary1['final_version']}")
    print(f"Total Iterations: {summary1['total_iterations']}")
    print(f"Final Score: {summary1['final_score']:.2%}")
    print("\nMetrics:")
    for metric, value in summary1["metrics"].items():
        print(f"  {metric}: {value:.2f}")
    print(f"\nFeedback Items Addressed: {summary1['feedback_addressed']}")
    print(f"Branch Points Explored: {summary1['branch_points_explored']}")

    # Analyze learning
    learning = system.analyze_learning()
    if learning:
        print("\n🧠 SYSTEM LEARNING ANALYSIS")
        print(f"Stories Generated: {learning.get('stories_generated', 0)}")
        print(f"Avg Iterations Needed: {learning.get('average_iterations', 0):.1f}")
        print(f"Avg Final Score: {learning.get('average_final_score', 0):.2%}")

    print("\n✨ Iterative story development complete!")


if __name__ == "__main__":
    print("\n🚀 ITERATIVE STORY DEVELOPMENT SYSTEM\n")
    asyncio.run(demo_iterative_system())
