"""
Story Enhancement Loop
Real-time iterative development integrated with character simulation
"""

import asyncio
import json
import aiohttp
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time

# Import our existing components
from narrative_pipeline import NarrativePipeline, SceneDescriptor


class EnhancementFocus(Enum):
    """What aspect to enhance"""

    CONFLICT = "conflict"
    EMOTION = "emotion"
    DIALOGUE = "dialogue"
    ACTION = "action"
    MYSTERY = "mystery"
    RELATIONSHIPS = "relationships"


class SceneQuality(Enum):
    """Scene quality assessment"""

    EXCELLENT = 5
    GOOD = 4
    ADEQUATE = 3
    WEAK = 2
    POOR = 1


@dataclass
class SceneEvaluation:
    """Evaluation of a scene's effectiveness"""

    scene_id: int
    quality: SceneQuality
    strengths: List[str]
    weaknesses: List[str]
    enhancement_suggestions: List[EnhancementFocus]
    specific_feedback: str
    audience_engagement: float  # 0-1


@dataclass
class EnhancementCycle:
    """A single enhancement cycle"""

    cycle_number: int
    focus: EnhancementFocus
    original_scene: SceneDescriptor
    enhanced_scene: SceneDescriptor
    improvement_score: float
    time_taken: float


class StoryEnhancementLoop:
    """Iterative story enhancement with real-time feedback"""

    def __init__(self, model: str = "google/gemma-2-27b"):
        self.model = model
        self.url = "http://localhost:1234/v1/chat/completions"
        self.pipeline = NarrativePipeline(model)
        self.enhancement_cycles: List[EnhancementCycle] = []
        self.scene_evaluations: Dict[int, SceneEvaluation] = {}
        self.quality_threshold = SceneQuality.GOOD
        self.max_enhancement_cycles = 3

    async def call_llm(
        self, prompt: str, context: str = "", temperature: float = 0.8
    ) -> str:
        """LLM call helper"""
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

    async def evaluate_scene(
        self, scene: SceneDescriptor, character_responses: Dict[str, Dict]
    ) -> SceneEvaluation:
        """Evaluate a scene's quality and identify improvements"""

        # Build evaluation context
        context = f"""You are a story editor evaluating narrative quality.
Scene: {scene.situation[:200]}...
Tension level: {scene.tension}
Characters: {', '.join(scene.characters)}

Character responses:
{json.dumps({k: v.get('dialogue', '')[:50] for k, v in character_responses.items()}, indent=2)}

Evaluate this scene's effectiveness and provide specific feedback.
Consider: dramatic tension, character motivation, dialogue quality, pacing, emotional impact.

Respond with JSON:
{{
  "quality_score": 1-5,
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "enhancement_focus": ["conflict/emotion/dialogue/action/mystery/relationships"],
  "specific_feedback": "Detailed feedback",
  "engagement": 0.0-1.0
}}"""

        response = await self.call_llm("Evaluate this scene", context, temperature=0.6)

        # Parse evaluation
        try:
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                data = json.loads(response[json_start:json_end])

                quality = SceneQuality(data.get("quality_score", 3))

                # Map enhancement strings to enums
                enhancement_map = {
                    "conflict": EnhancementFocus.CONFLICT,
                    "emotion": EnhancementFocus.EMOTION,
                    "dialogue": EnhancementFocus.DIALOGUE,
                    "action": EnhancementFocus.ACTION,
                    "mystery": EnhancementFocus.MYSTERY,
                    "relationships": EnhancementFocus.RELATIONSHIPS,
                }

                enhancements = []
                for focus in data.get("enhancement_focus", []):
                    if focus in enhancement_map:
                        enhancements.append(enhancement_map[focus])

                return SceneEvaluation(
                    scene_id=scene.id,
                    quality=quality,
                    strengths=data.get("strengths", []),
                    weaknesses=data.get("weaknesses", []),
                    enhancement_suggestions=enhancements,
                    specific_feedback=data.get("specific_feedback", ""),
                    audience_engagement=data.get("engagement", 0.5),
                )
        except Exception as e:
            print(f"Evaluation parsing error: {e}")

        # Fallback evaluation
        return SceneEvaluation(
            scene_id=scene.id,
            quality=SceneQuality.ADEQUATE,
            strengths=["Basic structure"],
            weaknesses=["Needs development"],
            enhancement_suggestions=[EnhancementFocus.EMOTION],
            specific_feedback="Scene needs enhancement",
            audience_engagement=0.5,
        )

    async def enhance_scene(
        self,
        scene: SceneDescriptor,
        focus: EnhancementFocus,
        evaluation: SceneEvaluation,
    ) -> SceneDescriptor:
        """Enhance a scene based on focus area"""

        enhancement_prompts = {
            EnhancementFocus.CONFLICT: "Add tension through character opposition or obstacles",
            EnhancementFocus.EMOTION: "Deepen emotional stakes and character feelings",
            EnhancementFocus.DIALOGUE: "Make dialogue more natural, revealing, and character-specific",
            EnhancementFocus.ACTION: "Add physical actions and dynamic movement",
            EnhancementFocus.MYSTERY: "Introduce questions, secrets, or uncertainty",
            EnhancementFocus.RELATIONSHIPS: "Develop character connections and dynamics",
        }

        prompt = f"""Enhance this scene with focus on {focus.value}:

Original: {scene.situation}

Weaknesses to address: {', '.join(evaluation.weaknesses)}
Enhancement goal: {enhancement_prompts.get(focus, 'Improve overall quality')}

Provide an enhanced version that:
1. Addresses the weaknesses
2. Maintains story continuity
3. Increases dramatic engagement
4. Stays true to character motivations

Enhanced scene:"""

        enhanced_situation = await self.call_llm(prompt, temperature=0.8)

        if not enhanced_situation:
            return scene

        # Create enhanced scene
        enhanced = SceneDescriptor(
            id=scene.id,
            name=scene.name + f" (Enhanced: {focus.value})",
            situation=enhanced_situation,
            location=scene.location,
            characters=scene.characters,
            tension=(
                min(1.0, scene.tension * 1.1)
                if focus == EnhancementFocus.CONFLICT
                else scene.tension
            ),
            goals=scene.goals,
            emphasis=scene.emphasis,
            sensory=scene.sensory,
        )

        # Add richer sensory details if focusing on emotion
        if focus == EnhancementFocus.EMOTION:
            sensory_prompt = f"For this emotional scene: {enhanced_situation[:100]}... Add visceral sensory details."
            sensory_response = await self.call_llm(sensory_prompt, temperature=0.9)
            if sensory_response:
                enhanced.sensory["emotional"] = sensory_response[:100]

        return enhanced

    async def run_enhancement_cycle(
        self, scene: SceneDescriptor, character_responses: Dict[str, Dict]
    ) -> Tuple[SceneDescriptor, float]:
        """Run a single enhancement cycle on a scene"""

        start_time = time.time()

        # Evaluate current scene
        evaluation = await self.evaluate_scene(scene, character_responses)
        self.scene_evaluations[scene.id] = evaluation

        print("\n  📊 Scene Evaluation:")
        print(f"     Quality: {evaluation.quality.name} ({evaluation.quality.value}/5)")
        print(f"     Strengths: {', '.join(evaluation.strengths[:2])}")
        print(f"     Weaknesses: {', '.join(evaluation.weaknesses[:2])}")
        print(f"     Engagement: {evaluation.audience_engagement:.0%}")

        # Check if enhancement needed
        if evaluation.quality.value >= self.quality_threshold.value:
            print("     ✅ Scene meets quality threshold")
            return scene, 0.0

        # Select enhancement focus
        if evaluation.enhancement_suggestions:
            focus = evaluation.enhancement_suggestions[0]
        else:
            focus = EnhancementFocus.EMOTION  # Default

        print(f"     🎯 Enhancement focus: {focus.value}")

        # Enhance the scene
        enhanced_scene = await self.enhance_scene(scene, focus, evaluation)

        # Re-evaluate enhanced scene
        enhanced_eval = await self.evaluate_scene(enhanced_scene, character_responses)

        improvement = (enhanced_eval.quality.value - evaluation.quality.value) / 5.0

        print(f"     📈 Improvement: {improvement:.0%}")

        # Record cycle
        cycle = EnhancementCycle(
            cycle_number=len(self.enhancement_cycles) + 1,
            focus=focus,
            original_scene=scene,
            enhanced_scene=enhanced_scene,
            improvement_score=improvement,
            time_taken=time.time() - start_time,
        )
        self.enhancement_cycles.append(cycle)

        return enhanced_scene, improvement

    async def iterative_story_enhancement(
        self, title: str, premise: str, characters: List[Dict], num_scenes: int = 4
    ) -> Dict:
        """Run complete iterative story enhancement"""

        print(f"\n{'='*80}")
        print(f"🔄 ITERATIVE STORY ENHANCEMENT: {title}")
        print(f"{'='*80}")
        print(f"Premise: {premise}")
        print(f"Quality Target: {self.quality_threshold.name}")
        print(f"Max Enhancement Cycles: {self.max_enhancement_cycles}")

        # Generate initial story
        print("\n📝 Generating initial story...")
        arc = self.pipeline.build_arc(title, premise, num_scenes)

        scenes = []
        for beat in arc.beats:
            scene = await self.pipeline.craft_scene(beat, characters)
            scenes.append(scene)

        print(f"  ✅ Generated {len(scenes)} initial scenes")

        # Enhancement loop for each scene
        enhanced_scenes = []
        total_improvements = []

        for scene_idx, scene in enumerate(scenes):
            print(f"\n🎬 SCENE {scene_idx + 1}: {scene.name}")
            print(f"📍 {scene.location}")
            print(f"📝 {scene.situation[:150]}...")

            # Simulate character responses (simplified)
            character_responses = {}
            for char in characters:
                if char["id"] in scene.characters:
                    response = await self.pipeline.simulate_character_response(
                        scene, char
                    )
                    character_responses[char["id"]] = response

            # Run enhancement cycles
            current_scene = scene
            scene_improvements = []

            for cycle in range(self.max_enhancement_cycles):
                print(f"\n  🔧 Enhancement Cycle {cycle + 1}")

                enhanced_scene, improvement = await self.run_enhancement_cycle(
                    current_scene, character_responses
                )

                scene_improvements.append(improvement)

                # Stop if good enough or no improvement
                if improvement <= 0:
                    break

                current_scene = enhanced_scene

                # Re-generate character responses for enhanced scene
                if improvement > 0:
                    print("\n  🎭 Re-simulating character responses...")
                    for char in characters:
                        if char["id"] in scene.characters:
                            response = await self.pipeline.simulate_character_response(
                                current_scene, char
                            )
                            character_responses[char["id"]] = response
                            print(
                                f"     {char['name']}: \"{response.get('dialogue', '')[:50]}...\""
                            )

            enhanced_scenes.append(current_scene)
            total_improvements.extend(scene_improvements)

        # Generate summary
        summary = {
            "title": title,
            "total_scenes": len(enhanced_scenes),
            "enhancement_cycles": len(self.enhancement_cycles),
            "average_improvement": (
                sum(total_improvements) / len(total_improvements)
                if total_improvements
                else 0
            ),
            "final_evaluations": {},
        }

        # Final evaluation
        print(f"\n{'='*60}")
        print("📊 FINAL STORY EVALUATION")
        print(f"{'='*60}")

        for scene in enhanced_scenes:
            if scene.id in self.scene_evaluations:
                eval = self.scene_evaluations[scene.id]
                summary["final_evaluations"][scene.name] = {
                    "quality": eval.quality.name,
                    "engagement": eval.audience_engagement,
                }
                print(f"\n{scene.name}:")
                print(f"  Quality: {eval.quality.name}")
                print(f"  Engagement: {eval.audience_engagement:.0%}")

        print("\n📈 Overall Statistics:")
        print(f"  Enhancement Cycles: {len(self.enhancement_cycles)}")
        print(f"  Average Improvement: {summary['average_improvement']:.0%}")

        # Learning insights
        if self.enhancement_cycles:
            focus_counts = {}
            for cycle in self.enhancement_cycles:
                focus_counts[cycle.focus.value] = (
                    focus_counts.get(cycle.focus.value, 0) + 1
                )

            print("\n🧠 Enhancement Focus Areas:")
            for focus, count in sorted(
                focus_counts.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {focus}: {count} cycles")

        return summary

    async def adaptive_enhancement(
        self, scene: SceneDescriptor, target_engagement: float = 0.8
    ) -> SceneDescriptor:
        """Adaptively enhance scene until target engagement reached"""

        current_scene = scene
        attempts = 0
        max_attempts = 5

        while attempts < max_attempts:
            # Mock evaluation for demo
            current_engagement = 0.5 + (attempts * 0.1)  # Gradually improve

            if current_engagement >= target_engagement:
                print(f"  ✅ Target engagement reached: {current_engagement:.0%}")
                break

            # Select focus based on gap
            gap = target_engagement - current_engagement
            if gap > 0.3:
                focus = EnhancementFocus.CONFLICT
            elif gap > 0.2:
                focus = EnhancementFocus.EMOTION
            else:
                focus = EnhancementFocus.DIALOGUE

            print(f"  🎯 Adaptive focus: {focus.value} (gap: {gap:.0%})")

            # Create mock evaluation
            mock_eval = SceneEvaluation(
                scene_id=current_scene.id,
                quality=SceneQuality.ADEQUATE,
                strengths=["Structure"],
                weaknesses=["Needs more " + focus.value],
                enhancement_suggestions=[focus],
                specific_feedback="Enhance " + focus.value,
                audience_engagement=current_engagement,
            )

            # Enhance
            current_scene = await self.enhance_scene(current_scene, focus, mock_eval)
            attempts += 1

        return current_scene


async def demo_enhancement_loop():
    """Demonstrate the enhancement loop system"""

    characters = [
        {
            "id": "investigator",
            "name": "Sam Rivers",
            "role": "protagonist",
            "traits": ["analytical", "obsessive", "empathetic"],
        },
        {
            "id": "witness",
            "name": "Alex Chen",
            "role": "key witness",
            "traits": ["frightened", "observant", "conflicted"],
        },
        {
            "id": "suspect",
            "name": "Jordan Black",
            "role": "antagonist",
            "traits": ["charming", "manipulative", "desperate"],
        },
    ]

    enhancer = StoryEnhancementLoop()

    summary = await enhancer.iterative_story_enhancement(
        title="The Last Testimony",
        premise="An investigator must protect a witness whose testimony could expose a conspiracy",
        characters=characters,
        num_scenes=3,
    )

    print("\n✨ Story enhancement complete!")
    print(f"Final average improvement: {summary['average_improvement']:.0%}")


if __name__ == "__main__":
    print("\n🚀 STORY ENHANCEMENT LOOP SYSTEM\n")
    asyncio.run(demo_enhancement_loop())
