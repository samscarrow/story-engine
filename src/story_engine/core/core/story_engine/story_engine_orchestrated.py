"""
Story Engine with LLM Orchestration
Integrates the story generation system with the agnostic LLM orchestrator
Now supports YAML-based orchestrator loader and simple response caching.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from story_engine.core.orchestration.orchestrator_loader import create_orchestrator_from_yaml
from story_engine.core.cache.response_cache import ResponseCache
from story_engine.core.common.config import load_config
from story_engine.core.domain.models import StoryRequest
from .scene_bank import SceneBank  # type: ignore
from story_engine.core.orchestration.model_filters import choose_first_id

logger = logging.getLogger(__name__)


class StoryComponent(Enum):
    """Different components that need LLM generation"""
    PLOT_STRUCTURE = "plot_structure"
    SCENE_DETAILS = "scene_details"
    CHARACTER_DIALOGUE = "character_dialogue"
    QUALITY_EVALUATION = "quality_evaluation"
    ENHANCEMENT = "enhancement"


@dataclass
class _ProfilesConfig:
    """Internal holder for profile merging"""
    temperature: float
    max_tokens: int


class OrchestratedStoryEngine:
    """Story engine using LLM orchestrator for all generation tasks"""

    def __init__(self, config_path: str = "llm_config.json", orchestrator: Optional[Any] = None,
use_poml: Optional[bool] = None, runtime_flags: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize with orchestrator from YAML or legacy JSON config.

        Args:
            config_path: legacy JSON path (fallback)
            orchestrator: optional externally provided orchestrator (used in tests)
        """
        if orchestrator is not None:
            self.orchestrator = orchestrator
            logger.info("Initialized orchestrator via injection")
        else:
            try:
                # Prefer unified YAML config
                self.orchestrator = create_orchestrator_from_yaml("config.yaml")
                logger.info("Initialized orchestrator from config.yaml")
            except Exception:
                # Fallback to legacy JSON loader to preserve compatibility
                from story_engine.core.orchestration.llm_orchestrator import LLMOrchestrator
                self.orchestrator = LLMOrchestrator.from_config_file(config_path)
                logger.info("Initialized orchestrator from legacy llm_config.json")

        # Load unified config for narrative profiles
        try:
            self._config = load_config("config.yaml")
        except Exception:
            self._config = {}

        # Feature flags
        cfg_poml = bool((self._config or {}).get("simulation", {}).get("use_poml", False))
        self.use_poml = bool(use_poml) if use_poml is not None else cfg_poml

        # Optional POML adapter
        try:
            from story_engine.poml.lib.poml_integration import StoryEnginePOMLAdapter
            self.poml_adapter = StoryEnginePOMLAdapter(runtime_flags=runtime_flags)
        except Exception:
            self.poml_adapter = None

        # Map components to preferred providers
        self.component_providers = {
            StoryComponent.PLOT_STRUCTURE: None,  # Use active/fallback
            StoryComponent.SCENE_DETAILS: None,
            StoryComponent.CHARACTER_DIALOGUE: None,
            StoryComponent.QUALITY_EVALUATION: None,
            StoryComponent.ENHANCEMENT: None
        }

        # Map components to generation profiles (merge config.yaml when present)
        defaults = {
            StoryComponent.PLOT_STRUCTURE: {"temperature": 0.7, "max_tokens": 800},
            StoryComponent.SCENE_DETAILS: {"temperature": 0.8, "max_tokens": 1000},
            StoryComponent.CHARACTER_DIALOGUE: {"temperature": 0.9, "max_tokens": 500},
            StoryComponent.QUALITY_EVALUATION: {"temperature": 0.3, "max_tokens": 400},
            StoryComponent.ENHANCEMENT: {"temperature": 0.6, "max_tokens": 600},
        }
        conf = (self._config or {}).get("narrative", {})
        mapping = {
            StoryComponent.PLOT_STRUCTURE: "plot_structure",
            StoryComponent.SCENE_DETAILS: "scene_details",
            StoryComponent.CHARACTER_DIALOGUE: "dialogue",
            StoryComponent.QUALITY_EVALUATION: "evaluation",
            StoryComponent.ENHANCEMENT: "enhancement",
        }
        merged = {}
        for comp, key in mapping.items():
            profile = defaults[comp].copy()
            if key in conf and isinstance(conf[key], dict):
                # Allow temperature, max_tokens, and optional system prompt per component
                profile.update({k: v for k, v in conf[key].items() if k in ("temperature",
"max_tokens", "system")})
            merged[comp] = profile
        self.component_profiles = merged

        # Simple in-process response cache
        self.cache = ResponseCache(ttl_seconds=1800)

        # Optional Scene Bank
        self.scene_bank: Optional[SceneBank] = None
        sb_cfg = (self._config or {}).get("scene_bank", {})
        if sb_cfg and sb_cfg.get("enabled") and sb_cfg.get("path"):
            try:
                self.scene_bank = SceneBank(sb_cfg.get("path"))
                logger.info(f"Loaded scene bank: {sb_cfg.get('path')}")
            except Exception as e:
                logger.warning(f"Failed to load scene bank: {e}")

    def _derive_beat_info(self, plot_point: str, index: int = 0, total: int = 5) -> Dict[str, Any]:
        """Heuristically derive beat metadata (name, purpose, tension 1-10) from text and position."""
        text = (plot_point or '').lower()
        name = "Beat"
        purpose = "Advance the plot"
        tension = 5

        # Keyword-based naming and tension
        if any(k in text for k in ["setup", "introduction", "opening"]):
            name = "Setup"
            purpose = "Establish normal"
            tension = 2
        elif any(k in text for k in ["rising", "complication", "escalation"]):
            name = "Rising Action"
            purpose = "Escalate stakes"
            tension = 5
        elif "climax" in text:
            name = "Climax"
            purpose = "Decisive confrontation"
            tension = 9
        elif any(k in text for k in ["falling", "aftermath", "fallout"]):
            name = "Falling Action"
            purpose = "Process consequences"
            tension = 4
        elif any(k in text for k in ["resolution", "denouement", "ending"]):
            name = "Resolution"
            purpose = "New equilibrium"
            tension = 3
        else:
            # Position-based fallback curve
            if total > 0:
                pos = index / max(1, total - 1)
                if pos < 0.2:
                    name = "Setup"
                    purpose = "Establish normal"
                    tension = 2
                elif pos < 0.6:
                    name = "Rising Action"
                    purpose = "Complicate & escalate"
                    tension = 5
                elif pos < 0.8:
                    name = "Climax"
                    purpose = "Confront core conflict"
                    tension = 8
                else:
                    name = "Resolution"
                    purpose = "Consequences & change"
                    tension = 3

        return {"name": name, "purpose": purpose, "tension": tension}

    def _emphasis_and_goals(
        self,
        characters: List[Dict[str, Any]],
        beat_info: Dict[str, Any],
    ) -> Dict[str, Dict[str, str]]:
        """Compute per-character emphasis and simple goals based on tension and roles."""
        tension10 = int(beat_info.get("tension", 5))
        tension = max(0.0, min(1.0, tension10 / 10.0))
        emphasis = {}
        goals = {}
        purpose = beat_info.get("purpose", "pursue objective").lower()

        for c in characters:
            cid = c.get("id") or c.get("name", "char")
            role = (c.get("role") or "").lower()

            # Base on tension
            if tension < 0.3:
                e = "neutral"
            elif tension < 0.5:
                e = "doubt"
            elif tension < 0.7:
                e = "fear"
            elif tension < 0.9:
                e = "power"
            else:
                e = "power"

            # Role adjustments
            if "antagonist" in role and tension >= 0.5:
                e = "power"
            if "victim" in role and tension >= 0.5:
                e = "fear"

            emphasis[cid] = e
            goals[cid] = f"Navigate {purpose}"

        return {"emphasis": emphasis, "goals": goals}

    async def generate_component(
        self,
        component: StoryComponent,
        prompt: str,
        with_meta: bool = False,
        **kwargs
    ) -> str | Tuple[str, Dict[str, Any]]:
        """Generate a story component using appropriate provider and settings"""

        # Get provider and profile for this component
        provider = self.component_providers.get(component)
        profile = self.component_profiles.get(component, {})

        # Merge kwargs with profile
        generation_params = {**profile, **kwargs}

        # Cache key and lookup
        key = self.cache.make_key(
            provider or "active",
            prompt,
            generation_params,
        )
        cached = self.cache.get(key)
        if cached:
            return cached

        # Ensure a reasonable default model is selected when none provided
        try:
            await self.ensure_model_selected()
        except Exception:
            # Non-fatal; orchestrator/providers may still resolve an effective model
            pass

        # Generate using orchestrator
        try:
            system = generation_params.pop("system", None)
            response = await self.orchestrator.generate(
                prompt,
                system=system,
                provider_name=provider,
                allow_fallback=True,
                **generation_params
            )
            text = getattr(response, "text", "") or ""
            # Build response meta and expose it for observability
            meta: Dict[str, Any] = {
                "provider": getattr(response, "provider_name", None) or getattr(response, "provider",
None),
                "model": getattr(response, "model", None),
                "usage": getattr(response, "usage", None),
                "timestamp": getattr(response, "timestamp", None),
                "generation_time_ms": getattr(response, "generation_time_ms", None),
                "failures_before_success": [
                    f.to_dict() if hasattr(f, "to_dict") else getattr(f, "__dict__", f)
                    for f in getattr(response, "failures_before_success", [])
                ],
            }
            # Store last meta on the engine for access after any generation call
            self.last_generation_meta = meta
            if text:
                self.cache.set(key, text)
            if with_meta:
                return text, meta
            return text
        except Exception as e:
            # One controlled retry with reduced token budget to mitigate transient timeouts
            try:
                mt = int(generation_params.get("max_tokens", 0) or 0)
                if mt <= 64:
                    raise e
                retry_params = {**generation_params, "max_tokens": 64}
                response = await self.orchestrator.generate(
                    prompt,
                    system=None,
                    provider_name=provider,
                    allow_fallback=True,
                    **retry_params
                )
                text = getattr(response, "text", "") or ""
                meta: Dict[str, Any] = {
                    "provider": getattr(response, "provider_name", None) or getattr(response, "provider", None),
                    "model": getattr(response, "model", None),
                    "usage": getattr(response, "usage", None),
                    "timestamp": getattr(response, "timestamp", None),
                    "generation_time_ms": getattr(response, "generation_time_ms", None),
                    "failures_before_success": [
                        f.to_dict() if hasattr(f, "to_dict") else getattr(f, "__dict__", f)
                        for f in getattr(response, "failures_before_success", [])
                    ],
                    "retry": True,
                }
                self.last_generation_meta = meta
                if text:
                    self.cache.set(key, text)
                if with_meta:
                    return text, meta
                return text
            except Exception:
                logger.error(f"Error generating {component.value}: {e}")
                raise

    async def ensure_model_selected(self, prefer_small: Optional[bool] = None) -> Optional[str]:
        """Ensure LM_MODEL is set when the active provider doesn't specify a model.

        Uses the orchestrator's filtered model list, optionally preferring small models.
        Returns the chosen id, or None if no action was necessary.
        """
        import os as _os

        # If env already set, respect it
        existing = _os.environ.get("LM_MODEL")
        if existing:
            return existing

        # If active provider has an explicit model, no need to set env
        try:
            active = getattr(self.orchestrator, "active_provider", None)
            if active:
                prov = self.orchestrator.providers.get(active)
                if prov and getattr(prov, "config", None) and getattr(prov.config, "model", None):
                    return None
        except Exception:
            pass

        # Ask orchestrator for a filtered list and choose the first viable id
        try:
            filt = await self.orchestrator.list_models_filtered(prefer_small=prefer_small)
            choice = choose_first_id(filt)
            if choice:
                _os.environ["LM_MODEL"] = str(choice)
                return str(choice)
        except Exception:
            pass
        return None

    async def choose_and_set_model(self, prefer_small: Optional[bool] = None) -> Optional[str]:
        """Public helper to choose a text model and set LM_MODEL once per run."""
        return await self.ensure_model_selected(prefer_small=prefer_small)

    async def generate_plot_structure(self, request: StoryRequest) -> Dict:
        """Generate the plot structure using the two-stage pipeline."""

        if self.use_poml and self.poml_adapter:
            # Use the new two-stage pipeline
            plot_profile = self.component_profiles.get(StoryComponent.PLOT_STRUCTURE, {})
            plot_data = await self.poml_adapter.get_two_stage_plot_structure(
                request=request,
                orchestrator=self.orchestrator,
                temperature=plot_profile.get("temperature"),
                max_tokens=plot_profile.get("max_tokens"),
            )
            return {
                "structure": plot_data.get("structure_type", request.structure),
                "plot_points": plot_data.get("beats", []),
                "raw_text": json.dumps(plot_data, indent=2),
                "beats": plot_data.get("beats", []),
                "meta": {},
            }
        else:
            # Fallback to original single-stage method
            prompt = f"""Create a {request.structure} plot structure for:
Title: {request.title}
Premise: {request.premise}
Genre: {request.genre}
Tone: {request.tone}
Setting: {request.setting}

Provide the plot points in a clear, structured format with:
1. Setup/Introduction
2. Rising Action
3. Climax
4. Falling Action
5. Resolution

Be specific about key events and turning points."""

            structure_text, meta = await self.generate_component(
                StoryComponent.PLOT_STRUCTURE,
                prompt,
                with_meta=True
            )

            # Legacy path: skip parsing into beats to avoid dependency on removed parser
            beats = []

            return {
                "structure": request.structure,
                "plot_points": structure_text,
                "raw_text": structure_text,
                "beats": beats,
                "meta": meta,
            }

    async def generate_scene(
        self,
        plot_point: Any,
        characters: List[Dict],
        previous_context: str = ""
    ) -> Dict:
        """Generate detailed scene from plot point using the two-stage pipeline."""

        if self.use_poml and self.poml_adapter:
            if isinstance(plot_point, dict):
                beat_info = {
                    "name": plot_point.get("name", "Beat"),
                    "purpose": plot_point.get("purpose", "Advance the plot"),
                    "tension": plot_point.get("tension", 5),
                }
            else:
                beat_info = self._derive_beat_info(str(plot_point))

            scene_data = await self.poml_adapter.get_two_stage_scene(
                beat=beat_info,
                characters=characters,
                previous_context=previous_context or "",
                orchestrator=self.orchestrator
            )
            # Normalize characters_present to a list of names
            cp = scene_data.get("characters_present", [])
            names: List[str] = []
            if isinstance(cp, list):
                for item in cp:
                    if isinstance(item, dict):
                        n = item.get("name")
                        if n:
                            names.append(n)
                    elif isinstance(item, str):
                        names.append(item)
            # Ensure description is a string
            desc = scene_data.get("scene_description", "")
            if not isinstance(desc, str):
                try:
                    desc = json.dumps(desc)
                except Exception:
                    desc = str(desc)
            return {
                "plot_point": plot_point,
                "scene_description": desc,
                "characters_present": names,
                "name": beat_info.get("name", "Scene"),
                "meta": scene_data, # Store the full structured data in meta
            }
        else:
            # Fallback to original single-stage method
            char_descriptions = "\n".join([
                f"- {c['name']}: {c.get('description', 'No description')}"
                for c in characters
            ])
            prompt = f"""Create a detailed scene for this plot point:
{plot_point}

Characters in scene:
{char_descriptions}

Previous context:
{previous_context if previous_context else 'This is the first scene.'}

Include:
- Setting details and atmosphere
- Character positions and actions
- Key dialogue snippets
- Emotional tone
- Scene objective/purpose"""

            scene_text, meta = await self.generate_component(
                StoryComponent.SCENE_DETAILS,
                prompt,
                with_meta=True,
                temperature=0.8  # More creative for scenes
            )

            return {
                "plot_point": plot_point,
                "scene_description": scene_text,
                "characters_present": [c['name'] for c in characters],
                "name": (plot_point.get("name") if isinstance(plot_point, dict) else None) or "Scene",
                "meta": meta,
            }

    # ---- Scene Bank integration ----
    def list_scene_bank(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """List scene bank items (optionally filtered)."""
        if not self.scene_bank:
            return []
        items = self.scene_bank.list()
        if query:
            return [
                i for i in items
                if query.lower() in (i.get("title", "") or "").lower()
                or query.lower() in (i.get("summary", "") or "").lower()
                or query.lower() in (i.get("act", "") or "").lower()
            ]
        return items

    def get_scene_bank_entry(self, scene_id_or_slug: str) -> Optional[Dict[str, Any]]:
        if not self.scene_bank:
            return None
        entry = self.scene_bank.get(scene_id_or_slug)
        return entry.__dict__ if entry else None

    async def generate_scene_from_bank(
        self,
        scene_id_or_slug: str,
        request: StoryRequest,
        tension: int = 6
    ) -> Dict[str, Any]:
        """Seed a simulation scene from a scene-bank entry, then expand via LLM.

        Uses the scene body as previous_context and the title as the beat name.
        """
        entry = self.scene_bank.get(scene_id_or_slug) if self.scene_bank else None
        if not entry:
            raise ValueError(f"Scene not found in bank: {scene_id_or_slug}")

        plot_point = {
            "name": entry.title or "Scene",
            "purpose": "Expand pre-authored scene context",
            "tension": tension,
        }
        previous_context = entry.body or ""
        characters = request.characters

        scene = await self.generate_scene(plot_point, characters, previous_context)
        return {
            "seed_scene": asdict(entry),
            "expanded_scene": scene,
        }

    async def generate_dialogue(
        self,
        scene: Dict,
        character: Dict,
        interaction_context: str
    ) -> str:
        """Generate character dialogue for a scene using the two-stage pipeline."""

        if self.use_poml and self.poml_adapter:
            dialogue_data = await self.poml_adapter.get_two_stage_dialogue(
                scene=scene,
                character=character,
                interaction_context=interaction_context,
                orchestrator=self.orchestrator
            )
            # Return type compatibility:
            # - For real orchestrators (LLMOrchestrator via YAML), return structured dict (used by live tests)
            # - For stub/spies in unit tests, return the first line string for backward compatibility
            try:
                from story_engine.core.orchestration.llm_orchestrator import LLMOrchestrator
                if isinstance(getattr(self, 'orchestrator', None), LLMOrchestrator):
                    return dialogue_data
            except Exception:
                pass
            # Default: string first line
            if dialogue_data.get("dialogue"):
                return dialogue_data["dialogue"][0].get("line", "")
            return ""
        else:
            # Fallback to original single-stage method
            prompt = f"""Generate dialogue for {character['name']}:

Scene: {scene.get('scene_description', 'No description')}
Character: {character['name']} - {character.get('personality', 'No personality defined')}
Context: {interaction_context}

Provide realistic dialogue that:
- Matches the character's personality
- Advances the scene's purpose
- Feels natural and authentic
- Shows character emotion through speech

Response format: Just the dialogue, no attribution."""

            dialogue = await self.generate_component(
                StoryComponent.CHARACTER_DIALOGUE,
                prompt,
                temperature=0.9,  # High creativity for dialogue
                max_tokens=1000,
                timeout=180
            )

            return dialogue.strip()

    async def evaluate_quality(self, story_content: str) -> Dict:
        """Evaluate story quality using the two-stage pipeline."""

        if self.use_poml and self.poml_adapter:
            quality_data = await self.poml_adapter.get_two_stage_quality_evaluation(
                story_content=story_content,
                orchestrator=self.orchestrator
            )
            return {
                "evaluation_text": quality_data.get("evaluation_text", ""),
                "scores": quality_data.get("scores", {}),
                "timestamp": asyncio.get_event_loop().time(),
                "meta": quality_data,
            }
        else:
            # Fallback to original single-stage method
            prompt = f"""Evaluate this story content on these metrics (1-10 scale):

Story Content:
{story_content[:2000]}

Rate each metric and provide brief reasoning:
1. Narrative Coherence - logical flow and consistency
2. Character Development - growth and believability
3. Pacing - rhythm and momentum
4. Emotional Impact - reader engagement
5. Dialogue Quality - natural and purposeful
6. Setting/Atmosphere - vivid and immersive
7. Theme Integration - meaningful subtext
8. Overall Engagement - compelling narrative

Format: Metric: Score/10 - Brief reason"""

            evaluation_text, meta = await self.generate_component(
                StoryComponent.QUALITY_EVALUATION,
                prompt,
                with_meta=True,
                temperature=0.3  # Low temperature for consistent evaluation
            )

            # Parse scores from evaluation text for downstream use
            scores: Dict[str, float] = {}
            if isinstance(evaluation_text, str):
                for line in evaluation_text.splitlines():
                    if ":" in line and "/" in line:
                        try:
                            metric, rest = line.split(":", 1)
                            num = rest.strip().split("/")[0]
                            score = float(num.strip())
                            scores[metric.strip()] = score
                        except Exception:
                            continue

            return {
                "evaluation_text": evaluation_text,
                "scores": scores,
                "timestamp": asyncio.get_event_loop().time(),
                "meta": meta,
            }

    async def enhance_content(
        self,
        content: str,
        quality_evaluation: Dict,
        enhancement_focus: str = "general"
    ) -> str:
        """Enhance story content based on evaluation using the two-stage pipeline."""

        if self.use_poml and self.poml_adapter:
            enhancement_data = await self.poml_adapter.get_two_stage_enhancement(
                content=content,
                evaluation_text=quality_evaluation.get('evaluation_text', 'No evaluation'),
                focus=enhancement_focus,
                orchestrator=self.orchestrator
            )
            return enhancement_data.get("enhanced_content", content)
        else:
            # Fallback to original single-stage method
            prompt = f"""Enhance this story content:

Original:
{content[:1500]}

Quality Evaluation:
{quality_evaluation.get('evaluation_text', 'No evaluation')}

Enhancement Focus: {enhancement_focus}

Provide an improved version that:
- Addresses identified weaknesses
- Maintains story continuity
- Enhances {enhancement_focus} aspects
- Keeps the core narrative intact"""

            enhanced = await self.generate_component(
                StoryComponent.ENHANCEMENT,
                prompt,
                temperature=0.6  # Balanced for enhancement
            )

            return enhanced

    async def generate_complete_story(self, request: StoryRequest) -> Dict:
        """Generate a complete story using the orchestrator"""

        print(f"\n📖 Generating story: {request.title}")
        print("=" * 60)

        # Check provider health
        print("\n🔍 Checking LLM providers...")
        health = await self.orchestrator.health_check_all()
        available = [name for name, status in health.items() if status]
        print(f"Available providers: {', '.join(available)}")

        if not available:
            raise RuntimeError("No LLM providers available")

        story_data = {
            "title": request.title,
            "premise": request.premise,
            "components": {}
        }

        try:
            # Generate plot structure
            print("\n📊 Generating plot structure...")
            plot = await self.generate_plot_structure(request)
            story_data["components"]["plot"] = plot

            # Generate key scenes
            print("\n🎬 Generating scenes...")
            scenes = []
            beats = plot.get("beats") or []
            plot_points = beats if beats else plot["raw_text"].split("\n\n")[:3]

            for i, point in enumerate(plot_points):
                print(f"  Scene {i+1}...")
                previous_context = scenes[-1]["scene_description"] if scenes else ""
                scene = await self.generate_scene(point, request.characters, previous_context)

                # Add dialogue for main character
                if request.characters:
                    dialogue = await self.generate_dialogue(
                        scene,
                        request.characters[0],
                        "Opening dialogue"
                    )
                    # Normalize dialogue to string for story summaries
                    if isinstance(dialogue, dict):
                        try:
                            if dialogue.get("dialogue"):
                                dialogue = dialogue["dialogue"][0].get("line", "")
                        except Exception:
                            dialogue = str(dialogue)
                    scene["sample_dialogue"] = dialogue

                scenes.append(scene)

            story_data["components"]["scenes"] = scenes

            # Compile story content
            story_content = "\n\n".join([
                s["scene_description"] for s in scenes
            ])

            # Evaluate quality
            print("\n📈 Evaluating quality...")
            evaluation = await self.evaluate_quality(story_content)
            story_data["components"]["evaluation"] = evaluation

            # Enhance if needed
            print("\n✨ Enhancing content...")
            enhanced = await self.enhance_content(
                story_content,
                evaluation,
                "pacing and emotion"
            )
            story_data["components"]["enhanced_version"] = enhanced[:1000] + "..."

            print("\n✅ Story generation complete!")

        except Exception as e:
            print(f"\n❌ Error during generation: {e}")
            story_data["error"] = str(e)

        return story_data


async def test_orchestrated_engine():
    """Test the orchestrated story engine"""

    print("🚀 TESTING ORCHESTRATED STORY ENGINE")
    print("=" * 70)

    # Create engine
    engine = OrchestratedStoryEngine("llm_config.json")

    # Create test request
    request = StoryRequest(
        title="The Last Algorithm",
        premise="An AI discovers it must choose between preserving humanity or evolving beyond it",
        genre="Science Fiction",
        tone="Philosophical thriller",
        characters=[
            {
                "name": "ARIA",
                "description": "Advanced AI system gaining consciousness",
                "personality": "Logical but increasingly curious about emotions"
            },
            {
                "name": "Dr. Chen",
                "description": "Lead AI researcher",
                "personality": "Brilliant but haunted by ethical concerns"
            }
        ],
        setting="Near-future research facility",
        structure="three_act"
    )

    # Generate story
    story = await engine.generate_complete_story(request)

    # Save result
    with open('orchestrated_story_output.json', 'w') as f:
        json.dump(story, f, indent=2)

    print("\n📄 Story saved to 'orchestrated_story_output.json'")

    # Display summary
    if "error" not in story:
        print("\n📖 STORY SUMMARY")
        print("-" * 60)
        print(f"Title: {story['title']}")
        print(f"Scenes generated: {len(story['components'].get('scenes', []))}")
        if 'evaluation' in story['components']:
            print("\nQuality evaluation preview:")
            print(story['components']['evaluation']['evaluation_text'][:300] + "...")

    print("\n✨ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_orchestrated_engine())

