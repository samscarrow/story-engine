"""
End-to-end test using Pontius Pilate material with POML-enabled orchestrated engine.
Stubs the orchestrator to return structured plot, scenes, dialogue, evaluation, enhancement.
"""

import asyncio

from story_engine.core.story_engine.story_engine_orchestrated import (
    OrchestratedStoryEngine,
)
from story_engine.core.domain.models import StoryRequest


class PilateStubResponse:
    def __init__(self, text: str):
        self.text = text
        self.metadata = {}
        self.timestamp = ""


class PilateStubOrchestrator:
    async def generate(self, prompt: str, **kwargs):
        system_prompt = kwargs.get("system", "").lower()
        print(f"DEBUG PilateStubOrchestrator: system_prompt={system_prompt[:100]}...")
        print(f"DEBUG PilateStubOrchestrator: prompt={prompt[:100]}...")
        # Handle structuring prompts first
        if "json" in system_prompt:
            if '"evaluation_text"' in system_prompt:
                return PilateStubResponse(
                    '{"evaluation_text": "Narrative Coherence: 7/10 - Clear causal flow but some jumps.\nCharacter Development: 8/10 - Pilate\'s inner conflict is evident.\nPacing: 7/10 - Steady rise to a firm climax.\nEmotional Impact: 8/10 - Moral tension resonates.\nDialogue Quality: 7/10 - Voices are distinct.\nSetting/Atmosphere: 8/10 - Vivid sense of place.\nTheme Integration: 7/10 - Power vs. conscience present.\nOverall Engagement: 8/10 - Compelling ethical dilemma.", "scores": {"Narrative Coherence": 7, "Character Development": 8, "Pacing": 7, "Emotional Impact": 8, "Dialogue Quality": 7, "Setting/Atmosphere": 8, "Theme Integration": 7, "Overall Engagement": 8}}'
                )
            if '"dialogue":' in system_prompt:
                return PilateStubResponse(
                    '{"dialogue": [{"speaker": "Pilate", "line": "What is truth?", "tone": "weary", "recipient": "himself"}]}'
                )
            if "plot" in system_prompt:
                return PilateStubResponse(
                    '{"structure_type": "three_act", "beats": [{"name": "Setup", "description": "A tense Passover in Jerusalem.", "tension": 3, "purpose": "Introduce conflict"}, {"name": "Rising Action", "description": "Jesus is brought to Pilate.", "tension": 6, "purpose": "Build tension"}, {"name": "Climax", "description": "Pilate makes his decision.", "tension": 9, "purpose": "Resolve conflict"}]}'
                )
            if "enhancement" in system_prompt:
                return PilateStubResponse(
                    '{"enhanced_content": "The air is thick with the scent of dust and fear."}'
                )

        # Creative prompts (existing logic)
        if (
            "Story Overview" in prompt and "Create a" in prompt
        ) or "Setup/Introduction" in prompt:
            text = (
                "Setup: In Jerusalem, Governor Pontius Pilate oversees a tense province as rumors of a prophet spread."
                "\n\nRising Action: Religious leaders bring Jesus before Pilate at dawn; the crowd gathers, pressure mounts as politics and conscience collide."
                "\n\nClimax: Pilate faces the decisive moment—release the prisoner or appease the crowd calling for crucifixion."
                "\n\nFalling Action: After washing his hands, Pilate yields to the mob; soldiers take the condemned away as unrest simmers."
                "\n\nResolution: Order appears restored but the governor remains haunted by doubt and the ramifications of his judgment."
            )
            return PilateStubResponse(text)

        if (
            "Characters Present" in prompt
            or "Scene Creation Instructions" in prompt
            or "Scene:" in prompt
        ):
            return PilateStubResponse(
                "A stone courtyard before the praetorium at early light. Pilate hears accusations as the crowd swells; the air smells of dust and oil."
            )

        if (
            "Rate the following" in prompt
            or "Story Content (Excerpt)" in prompt
            or "expert literary critic" in system_prompt
        ):
            return PilateStubResponse(
                '{"evaluation_text": "Narrative Coherence: 7/10 - Clear causal flow but some jumps.\nCharacter Development: 8/10 - Pilate\'s inner conflict is evident.\nPacing: 7/10 - Steady rise to a firm climax.\nEmotional Impact: 8/10 - Moral tension resonates.\nDialogue Quality: 7/10 - Voices are distinct.\nSetting/Atmosphere: 8/10 - Vivid sense of place.\nTheme Integration: 7/10 - Power vs. conscience present.\nOverall Engagement: 8/10 - Compelling ethical dilemma.", "scores": {"Narrative Coherence": 7, "Character Development": 8, "Pacing": 7, "Emotional Impact": 8, "Dialogue Quality": 7, "Setting/Atmosphere": 8, "Theme Integration": 7, "Overall Engagement": 8}}'
            )

        if "Original Content (Excerpt)" in prompt or "Focus area:" in prompt:
            return PilateStubResponse(
                "The scene tightens: the crowd's roar layers under Pilate's measured words, while silences stretch to heighten dread."
            )

        return PilateStubResponse(
            "Pilate: What is truth? Yet duty binds me tighter than any answer."
        )

    async def health_check_all(self):
        # Simulate healthy provider map
        return {"stub": True}


def test_full_pilate_flow_with_poml():
    orch = PilateStubOrchestrator()
    engine = OrchestratedStoryEngine(orchestrator=orch, use_poml=True)

    request = StoryRequest(
        title="The Trial Before Dawn",
        premise=(
            "A Roman governor must choose between political stability and his sense of justice "
            "when a controversial prophet is brought before him."
        ),
        genre="Historical Drama",
        tone="Grave",
        characters=[
            {"id": "pilate", "name": "Pontius Pilate", "role": "conflicted judge"},
            {"id": "caiaphas", "name": "Caiaphas", "role": "antagonist"},
            {"id": "crowd", "name": "Crowd Representative", "role": "voice of mob"},
        ],
        setting="Jerusalem, during Passover",
        structure="three_act",
    )

    async def run():
        story = await engine.generate_complete_story(request)

        assert "components" in story
        plot = story["components"].get("plot", {})
        assert "beats" in plot and len(plot["beats"]) >= 3
        # Check first beat mapping
        first = plot["beats"][0]
        assert first["name"] in {
            "Setup",
            "Rising Action",
            "Climax",
            "Falling Action",
            "Resolution",
        }
        assert 1 <= int(first["tension"]) <= 10

        # Scenes block
        scenes = story["components"].get("scenes", [])
        assert len(scenes) == len(plot["beats"])  # one scene per beat
        assert all(
            isinstance(s.get("scene_description", ""), str)
            and len(s["scene_description"]) > 0
            for s in scenes
        )
        # Names come from beats
        assert scenes[0]["name"] in {
            "Setup",
            "Rising Action",
            "Climax",
            "Falling Action",
            "Resolution",
        }
        # Dialogue sample attached
        assert isinstance(scenes[0].get("sample_dialogue", ""), str)

        # Evaluation content
        evaluation = story["components"].get("evaluation", {})
        assert "evaluation_text" in evaluation
        assert "Narrative Coherence" in evaluation["evaluation_text"]

        # Enhanced version exists
        assert isinstance(story["components"].get("enhanced_version", ""), str)

    asyncio.run(run())
