"""
Narrative Pipeline
Streamlined story arc → scene → character simulation pipeline
Now supports optional POML prompts and an injected LLM orchestrator.
"""

import asyncio
import json
import aiohttp
from typing import Dict, List, Optional, Any
import time
import logging

logger = logging.getLogger(__name__)

from story_engine.core.domain.models import NarrativeArc, SceneDescriptor
from story_engine.core.cache.response_cache import ResponseCache
from story_engine.core.common.config import load_config

class NarrativePipeline:
    """Complete pipeline from arc to character responses"""
    
    def __init__(
        self,
        model: str = "google/gemma-2-27b",
        orchestrator: Optional[Any] = None,
        use_poml: Optional[bool] = None,
    ):
        """Initialize pipeline.

        Args:
            model: Fallback model for direct HTTP calls (legacy path)
            orchestrator: Optional LLM orchestrator instance
            use_poml: Optional flag to enable POML prompts
        """
        self.model = model
        self.url = "http://localhost:1234/v1/chat/completions"
        self.current_arc: Optional[NarrativeArc] = None
        self.scenes: List[SceneDescriptor] = []
        self.character_states = {}
        self.story_output = []

        # Orchestrator (preferred) and POML adapter (optional)
        self.orchestrator = orchestrator

        # Load unified configuration early so we can honor feature flags
        try:
            self._config = load_config("config.yaml")
        except Exception:
            self._config = {}

        # Initialize optional POML adapter
        try:
            from story_engine.poml.lib.poml_integration import StoryEnginePOMLAdapter
            self.poml_adapter = StoryEnginePOMLAdapter()
        except Exception:
            self.poml_adapter = None

        # POML feature flag: explicit arg overrides config; default to config value
        cfg_flag = bool((self._config or {}).get("simulation", {}).get("use_poml", False))
        self.use_poml = bool(use_poml) if use_poml is not None else cfg_flag

        # Response cache to avoid repeated identical calls
        self.response_cache = ResponseCache(ttl_seconds=1800)
        defaults = {
            "plot_structure": {"temperature": 0.7, "max_tokens": 800},
            "scene_details": {"temperature": 0.8, "max_tokens": 1000},
            "dialogue": {"temperature": 0.9, "max_tokens": 500},
            "evaluation": {"temperature": 0.3, "max_tokens": 400},
        }
        conf = (self._config or {}).get("narrative", {})
        self._profiles = {k: {**defaults[k], **conf.get(k, {})} for k in defaults.keys()}
    
    async def generate_with_llm(self, prompt: str, context: str = "", temperature: float = 0.8) -> str:
        """LLM call via orchestrator if available, else legacy HTTP path."""
        provider_name = "active" if self.orchestrator is not None else "legacy-http"
        params = {"temperature": temperature, "context": context[:120] if context else ""}
        cache_key = self.response_cache.make_key(provider_name, prompt, params)
        cached = self.response_cache.get(cache_key)
        if cached:
            return cached

        # Preferred path: orchestrator
        if self.orchestrator is not None:
            try:
                # Combine system content and user prompt if context provided
                full_prompt = f"{context}\n\n{prompt}" if context else prompt
                resp = await self.orchestrator.generate(full_prompt, allow_fallback=True, temperature=temperature)
                text = getattr(resp, "text", "") or ""
                if text:
                    self.response_cache.set(cache_key, text)
                return text
            except Exception as e:  # fallback to legacy
                logger.warning(f"Orchestrator generation failed, falling back: {e}")

        # Legacy path (direct HTTP)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": context if context else "You are a creative narrative designer."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 600,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload) as response:
                    data = await response.json()
                    text = data["choices"][0]["message"]["content"]
                    if text:
                        self.response_cache.set(cache_key, text)
                    return text
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""
    
    def build_arc(self, title: str, premise: str, num_beats: int = 5) -> NarrativeArc:
        """Build a simple narrative arc"""
        
        beats = []
        
        # Basic dramatic structure
        beat_templates = [
            {"name": "Setup", "tension": 0.2, "purpose": "Establish normal"},
            {"name": "Catalyst", "tension": 0.4, "purpose": "Disrupt normal"},
            {"name": "Escalation", "tension": 0.6, "purpose": "Raise stakes"},
            {"name": "Crisis", "tension": 0.8, "purpose": "Maximum conflict"},
            {"name": "Resolution", "tension": 0.3, "purpose": "New equilibrium"}
        ]
        
        for i, template in enumerate(beat_templates[:num_beats]):
            beats.append({
                "id": i,
                "name": template["name"],
                "tension": template["tension"],
                "purpose": template["purpose"],
                "premise_connection": premise
            })
        
        arc = NarrativeArc(title=title, beats=beats)
        self.current_arc = arc
        return arc
    
    async def craft_scene(self, beat: Dict, characters: List[Dict], 
                         previous_context: str = "") -> SceneDescriptor:
        """Craft a detailed scene from a story beat"""
        
        # Generate scene details with LLM
        if self.use_poml and self.poml_adapter:
            prompt = self.poml_adapter.get_scene_prompt(
                beat=beat,
                characters=characters,
                previous_context=previous_context,
            )
        else:
            prompt = f"""Create a dramatic scene for:
Beat: {beat['name']} - {beat['purpose']}
Tension level: {beat['tension']}
Characters: {', '.join([c['name'] for c in characters])}
Previous context: {previous_context if previous_context else 'Opening scene'}

Provide a detailed situation description (2-3 sentences) that gives characters clear dramatic opportunities.
Include: location, time of day, immediate conflict or tension, and what's at stake.

Scene situation:"""
        
        # Use configured profile for scene details
        scene_profile = self._profiles.get("scene_details", {"temperature": 0.8})
        situation = await self.generate_with_llm(
            prompt,
            temperature=scene_profile.get("temperature", 0.8),
        )
        
        if not situation:
            # Fallback
            situation = f"{beat['name']}: The characters face {beat['purpose'].lower()}."
        
        # Generate sensory details
        sensory_prompt = (
            f"For this scene: {situation[:100]}... Provide ONE sensory detail each: "
            f"sight, sound, atmosphere. Brief, evocative."
        )
        sensory_response = await self.generate_with_llm(sensory_prompt, temperature=0.9)
        
        sensory = self._parse_sensory(sensory_response)
        
        # Create scene descriptor
        scene = SceneDescriptor(
            id=len(self.scenes),
            name=beat['name'],
            situation=situation,
            location=self._extract_location(situation),
            characters=[c['id'] for c in characters],
            tension=beat['tension'],
            goals={c['id']: f"Navigate {beat['purpose'].lower()}" for c in characters},
            emphasis=self._calculate_emphasis(beat['tension'], characters),
            sensory=sensory
        )
        
        self.scenes.append(scene)
        return scene
    
    def _parse_sensory(self, response: str) -> Dict[str, str]:
        """Parse sensory details from response"""
        
        sensory = {
            "sight": "harsh lighting",
            "sound": "tense silence", 
            "atmosphere": "oppressive"
        }
        
        if response:
            lines = response.lower().split('\n')
            for line in lines:
                if 'sight' in line or 'see' in line or 'visual' in line:
                    sensory["sight"] = line.split(':')[-1].strip()[:50]
                elif 'sound' in line or 'hear' in line or 'audio' in line:
                    sensory["sound"] = line.split(':')[-1].strip()[:50]
                elif 'atmosphere' in line or 'feel' in line or 'mood' in line:
                    sensory["atmosphere"] = line.split(':')[-1].strip()[:50]
        
        return sensory
    
    def _extract_location(self, situation: str) -> str:
        """Extract location from situation"""
        
        # Simple heuristic
        location_words = ["room", "hall", "chamber", "office", "street", "building", 
                         "courtyard", "palace", "temple", "house", "plaza"]
        
        for word in location_words:
            if word in situation.lower():
                return word.capitalize()
        
        return "Unknown Location"
    
    def _calculate_emphasis(self, tension: float, characters: List[Dict]) -> Dict[str, str]:
        """Calculate emphasis for each character based on tension"""
        
        emphasis = {}
        
        for char in characters:
            if tension < 0.3:
                emphasis[char['id']] = "neutral"
            elif tension < 0.5:
                emphasis[char['id']] = "doubt"
            elif tension < 0.7:
                emphasis[char['id']] = "fear"
            elif tension < 0.9:
                emphasis[char['id']] = "power"
            else:
                emphasis[char['id']] = "neutral"
            
            # Modify based on character role
            if char.get('role') == 'antagonist' and tension > 0.5:
                emphasis[char['id']] = "power"
            elif char.get('role') == 'victim' and tension > 0.5:
                emphasis[char['id']] = "fear"
        
        return emphasis
    
    async def simulate_character_response(self, scene: SceneDescriptor, 
                                         character: Dict) -> Dict[str, Any]:
        """Simulate a character's response to a scene"""
        
        # Build character context and prompt
        if self.use_poml and self.poml_adapter:
            # Use POML dialogue generation template
            dialogue_prompt = self.poml_adapter.get_dialogue_prompt(
                character=character,
                scene=scene,
                dialogue_context={
                    "emphasis": scene.emphasis.get(character["id"], "neutral"),
                    "goal": scene.goals.get(character["id"], "respond appropriately"),
                },
            )
            dlg_profile = self._profiles.get("dialogue", {"temperature": 0.9})
            response = await self.generate_with_llm(
                prompt=dialogue_prompt,
                temperature=dlg_profile.get("temperature", 0.9),
            )
        else:
            # Legacy inline prompt
            char_context = f"""You are {character['name']}, {character.get('role', 'character')}.
Traits: {', '.join(character.get('traits', ['complex']))}
Current emotion focus: {scene.emphasis.get(character['id'], 'neutral')}
Goal: {scene.goals.get(character['id'], 'respond appropriately')}

Scene atmosphere: {scene.sensory.get('atmosphere', 'tense')}
You can hear: {scene.sensory.get('sound', 'ambient noise')}
You can see: {scene.sensory.get('sight', 'the scene before you')}

Respond with JSON only:
{{"dialogue": "what you say", "thought": "inner monologue", "action": "physical action"}}"""
            dlg_profile = self._profiles.get("dialogue", {"temperature": 0.9})
            response = await self.generate_with_llm(
                prompt=f"Scene: {scene.situation}\n\nHow do you respond?",
                context=char_context,
                temperature=dlg_profile.get("temperature", 0.9),
            )
        
        # Parse JSON from response
        try:
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                return json.loads(response[json_start:json_end])
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
        
        # Fallback
        return {
            "dialogue": "...",
            "thought": character.get('name', 'Character') + " considers the situation",
            "action": "reacts to the scene"
        }
    
    async def run_full_pipeline(self, title: str, premise: str, 
                               characters: List[Dict], num_scenes: int = 5):
        """Run the complete narrative pipeline"""
        
        print(f"\n{'='*80}")
        print(f"📚 NARRATIVE PIPELINE: {title}")
        print(f"{'='*80}")
        print(f"Premise: {premise}")
        print(f"Characters: {', '.join([c['name'] for c in characters])}")
        print(f"Scenes to generate: {num_scenes}\n")
        
        # Step 1: Build narrative arc
        print("📐 Building narrative arc...")
        arc = self.build_arc(title, premise, num_scenes)
        print(f"  ✅ Arc created with {len(arc.beats)} beats")
        
        # Step 2: Craft scenes from beats
        print("\n🎬 Crafting scenes from beats...")
        previous_context = ""
        
        for beat in arc.beats:
            print(f"\n  Beat {beat['id']+1}: {beat['name']}")
            scene = await self.craft_scene(beat, characters, previous_context)
            print(f"    📍 Location: {scene.location}")
            print(f"    📊 Tension: {scene.tension:.0%}")
            print(f"    📝 {scene.situation[:100]}...")
            
            # Update context for next scene
            previous_context = f"{scene.name}: {scene.situation[:50]}"
        
        # Step 3: Simulate character responses
        print("\n\n🎭 SIMULATING CHARACTER RESPONSES")
        print("="*60)
        
        for scene in self.scenes:
            print(f"\n📖 SCENE {scene.id+1}: {scene.name}")
            print(f"📍 {scene.location}")
            print(f"🎯 Tension: {scene.tension:.0%}")
            print(f"\n📝 {scene.situation}")
            
            if scene.sensory:
                print(f"\n🌍 Atmosphere: {scene.sensory.get('atmosphere', 'N/A')}")
                print(f"👁️ Visual: {scene.sensory.get('sight', 'N/A')}")
                print(f"👂 Sound: {scene.sensory.get('sound', 'N/A')}")
            
            scene_responses = {}
            
            for char_dict in characters:
                char_id = char_dict['id']
                if char_id in scene.characters:
                    print(f"\n💭 {char_dict['name']} ({scene.emphasis.get(char_id, 'neutral')} emphasis):")
                    
                    start = time.time()
                    response = await self.simulate_character_response(scene, char_dict)
                    elapsed = time.time() - start
                    
                    print(f"  💬 \"{response.get('dialogue', 'N/A')[:100]}\"")
                    print(f"  🤔 *{response.get('thought', 'N/A')[:80]}*")
                    print(f"  🎬 {response.get('action', 'N/A')[:60]}")
                    print(f"  ⏱️ {elapsed:.1f}s")
                    
                    scene_responses[char_id] = response
            
            self.story_output.append({
                'scene': scene,
                'responses': scene_responses
            })
        
        # Summary
        print(f"\n{'='*80}")
        print("✅ PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Generated: {len(self.scenes)} scenes")
        print(f"Total responses: {sum(len(s['responses']) for s in self.story_output)}")
        
        return self.story_output


async def demo_pipeline():
    """Demonstrate the narrative pipeline"""
    
    # Define a simple story
    characters = [
        {
            'id': 'detective',
            'name': 'Detective Morgan',
            'role': 'protagonist',
            'traits': ['observant', 'haunted', 'determined']
        },
        {
            'id': 'suspect',
            'name': 'Dr. Blackwood',
            'role': 'antagonist',
            'traits': ['brilliant', 'deceptive', 'desperate']
        },
        {
            'id': 'witness',
            'name': 'Emma Walsh',
            'role': 'key witness',
            'traits': ['nervous', 'truthful', 'traumatized']
        }
    ]
    
    pipeline = NarrativePipeline()
    
    await pipeline.run_full_pipeline(
        title="The Last Witness",
        premise="A detective must protect the only witness to a murder while uncovering a conspiracy",
        characters=characters,
        num_scenes=5
    )


async def demo_historical():
    """Demo with historical scenario"""
    
    characters = [
        {
            'id': 'pilate',
            'name': 'Pontius Pilate',
            'role': 'conflicted judge',
            'traits': ['pragmatic', 'fearful', 'political']
        },
        {
            'id': 'caiaphas',
            'name': 'Caiaphas',
            'role': 'antagonist',
            'traits': ['cunning', 'religious', 'determined']
        },
        {
            'id': 'crowd_leader',
            'name': 'Crowd Representative',
            'role': 'voice of mob',
            'traits': ['angry', 'volatile', 'influenced']
        }
    ]
    
    pipeline = NarrativePipeline()
    
    await pipeline.run_full_pipeline(
        title="The Trial",
        premise="A Roman prefect must decide the fate of an accused prophet while managing political pressures",
        characters=characters,
        num_scenes=5
    )


if __name__ == "__main__":
    import sys
    
    print("\n🚀 NARRATIVE PIPELINE SYSTEM\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "historical":
        asyncio.run(demo_historical())
    else:
        asyncio.run(demo_pipeline())

