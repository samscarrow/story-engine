"""
End-to-end test using Pontius Pilate material with POML-enabled orchestrated engine.
Stubs the orchestrator to return structured plot, scenes, dialogue, evaluation, enhancement.
"""

import asyncio

from core.story_engine.story_engine_orchestrated import OrchestratedStoryEngine
from core.domain.models import StoryRequest


class PilateStubResponse:
    def __init__(self, text: str):
        self.text = text
        self.metadata = {}
        self.timestamp = ""


class PilateStubOrchestrator:
    async def generate(self, prompt: str, **kwargs):
        # Identify plot structure prompts (POML plot template contains these cues)
        if ("Story Overview" in prompt and "Create a" in prompt) or "Setup/Introduction" in prompt:
            text = (
                "Setup: In Jerusalem, Governor Pontius Pilate oversees a tense province as rumors of a prophet spread."\
                "\n\nRising Action: Religious leaders bring Jesus before Pilate at dawn; the crowd gathers, pressure mounts as politics and conscience collide."\
                "\n\nClimax: Pilate faces the decisive moment—release the prisoner or appease the crowd calling for crucifixion."\
                "\n\nFalling Action: After washing his hands, Pilate yields to the mob; soldiers take the condemned away as unrest simmers."\
                "\n\nResolution: Order appears restored but the governor remains haunted by doubt and the ramifications of his judgment."
            )
            return PilateStubResponse(text)

        # Identify scene crafting prompts
        if "Characters Present" in prompt or "Scene Creation Instructions" in prompt or "Scene:" in prompt:
            return PilateStubResponse(
                "A stone courtyard before the praetorium at early light. Pilate hears accusations as the crowd swells; the air smells of dust and oil."
            )

        # Identify evaluation prompts
        if "Rate the following" in prompt or "Story Content (Excerpt)" in prompt:
            return PilateStubResponse(
                "Narrative Coherence: 7/10 - Clear causal flow but some jumps.\n"
                "Character Development: 8/10 - Pilate's inner conflict is evident.\n"
                "Pacing: 7/10 - Steady rise to a firm climax.\n"
                "Emotional Impact: 8/10 - Moral tension resonates.\n"
                "Dialogue Quality: 7/10 - Voices are distinct.\n"
                "Setting/Atmosphere: 8/10 - Vivid sense of place.\n"
                "Theme Integration: 7/10 - Power vs. conscience present.\n"
                "Overall Engagement: 8/10 - Compelling ethical dilemma."
            )

        # Identify enhancement prompts
        if "Original Content (Excerpt)" in prompt or "Focus area:" in prompt:
            return PilateStubResponse(
                "The scene tightens: the crowd's roar layers under Pilate's measured words, while silences stretch to heighten dread."
            )

        # Fallback: treat as dialogue or generic text
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
        assert first["name"] in {"Setup", "Rising Action", "Climax", "Falling Action", "Resolution"}
        assert 1 <= int(first["tension"]) <= 10

        # Scenes block
        scenes = story["components"].get("scenes", [])
        assert len(scenes) == len(plot["beats"])  # one scene per beat
        assert all(isinstance(s.get("scene_description", ""), str) and len(s["scene_description"]) > 0 for s in scenes)
        # Names come from beats
        assert scenes[0]["name"] in {"Setup", "Rising Action", "Climax", "Falling Action", "Resolution"}
        # Dialogue sample attached
        assert isinstance(scenes[0].get("sample_dialogue", ""), str)

        # Evaluation content
        evaluation = story["components"].get("evaluation", {})
        assert "evaluation_text" in evaluation
        assert "Narrative Coherence" in evaluation["evaluation_text"]

        # Enhanced version exists
        assert isinstance(story["components"].get("enhanced_version", ""), str)

    asyncio.run(run())
