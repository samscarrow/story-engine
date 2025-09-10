from story_engine.poml.lib.poml_integration import POMLEngine


def test_path_normalization_accepts_leading_templates():
    engine = POMLEngine()
    # Using leading 'templates/' should still resolve
    out = engine.render('templates/narrative/scene_crafting.poml', {
        'beat': {'name': 'Setup', 'purpose': 'Establish', 'tension': 3},
        'characters': [{'name': 'Alice', 'role': 'protagonist'}],
        'previous_context': ''
    })
    assert isinstance(out, str)
    assert len(out) > 0


def test_render_roles_returns_dict():
    engine = POMLEngine()
    roles = engine.render_roles('simulations/character_response.poml', {
        'character': {'id': 'c1', 'name': 'Alice', 'traits': ['brave'], 'values': [], 'fears': [], 'desires': [], 'memory': {'recent_events': []}, 'emotional_state': {'anger': 0.1, 'doubt': 0.2, 'fear': 0.1, 'compassion': 0.4}},
        'situation': 'A quiet room',
        'emphasis': 'neutral'
    })
    assert set(roles.keys()) == {'system', 'user'}
    assert isinstance(roles['system'], str)
    assert isinstance(roles['user'], str)


def test_path_normalization_accepts_story_engine_prefix():
    engine = POMLEngine()
    # Prefix used by some callers
    out = engine.render('story-engine/narrative/scene_crafting_freeform.poml', {
        'beat': {'name': 'Setup', 'purpose': 'Establish', 'tension': 3},
        'characters': [{'name': 'Alice', 'role': 'protagonist'}],
        'previous_context': ''
    })
    assert isinstance(out, str) and out
