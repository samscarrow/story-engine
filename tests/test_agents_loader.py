from story_engine.agents.loader import list_agents, get_agent_prompt


def test_list_agents_and_get_prompt():
    agents = list_agents()
    assert "engine_integrator" in agents
    text = get_agent_prompt("engine_integrator")
    assert "System prompt" in text or len(text) > 20
