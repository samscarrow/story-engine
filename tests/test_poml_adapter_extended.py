"""
Basic tests for additional POML adapter methods: plot structure and evaluation.
These ensure integration wiring returns strings.
"""

from story_engine.poml.lib.poml_integration import StoryEnginePOMLAdapter


def test_plot_structure_prompt_renders_string():
    adapter = StoryEnginePOMLAdapter()
    request = {
        "title": "Test Story",
        "premise": "A challenge must be overcome",
        "genre": "Drama",
        "tone": "Serious",
        "setting": "A small town",
        "structure": "three_act",
    }
    prompt = adapter.get_plot_structure_prompt(request)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_quality_evaluation_prompt_renders_string():
    adapter = StoryEnginePOMLAdapter()
    story_content = "A short scene content example."
    metrics = [
        "Narrative Coherence",
        "Character Development",
        "Pacing",
    ]
    prompt = adapter.get_quality_evaluation_prompt(story_content, metrics)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_enhancement_prompt_renders_string():
    adapter = StoryEnginePOMLAdapter()
    focus = "pacing and emotion"
    evaluation = "Pacing is uneven; dialogue strong."
    prompt = adapter.get_enhancement_prompt(
        content="Original content.",
        evaluation_text=evaluation,
        focus=focus,
        metrics={"pacing": 0.4, "dialogue": 0.8},
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert focus in prompt
    assert evaluation.split(";")[0] in prompt
