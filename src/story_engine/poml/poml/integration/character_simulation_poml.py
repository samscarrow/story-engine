"""
Character Simulation Engine with POML Integration
Modified version of character_simulation_engine_v2.py using POML templates
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import original components
from character_simulation_engine_v2 import (
    CharacterState,
    EmotionalState,
    CharacterMemory,
    SimulationError,
    RetryHandler,
    CacheManager,
)

# Import POML engine
from story_engine.poml.lib.poml_integration import POMLEngine, StoryEnginePOMLAdapter

logger = logging.getLogger(__name__)


class POMLCharacterSimulationEngine:
    """
    Enhanced Character Simulation Engine using POML templates
    Drop-in replacement for the original SimulationEngine class
    """

    def __init__(self, llm_client, config: Dict = None):
        """
        Initialize with POML support

        Args:
            llm_client: LLM client (MockLLM, RealLLM, LMStudioLLM)
            config: Configuration dictionary
        """
        self.llm = llm_client
        self.config = config or {}

        # Initialize POML engine
        poml_config_path = Path(__file__).parent.parent / "config" / "poml_config.yaml"
        self.poml = POMLEngine(config_file=str(poml_config_path))
        self.poml_adapter = StoryEnginePOMLAdapter(self.poml)

        # Keep original components
        self.retry_handler = RetryHandler(
            max_attempts=config.get("max_retries", 3),
            base_delay=config.get("retry_delay", 1.0),
        )

        # Initialize cache if enabled
        cache_config = config.get("cache", {})
        if cache_config.get("enabled", True):
            self.cache = CacheManager(
                ttl_seconds=cache_config.get("ttl_seconds", 3600),
                max_size_mb=cache_config.get("max_size_mb", 100),
            )
        else:
            self.cache = None

        # Tracking
        self.simulation_results = []

        logger.info("POMLCharacterSimulationEngine initialized with POML templates")

    def get_simulation_prompt(
        self,
        character: CharacterState,
        situation: str,
        emphasis: str = "neutral",
        context: Dict = None,
    ) -> str:
        """
        Generate simulation prompt using POML templates

        Args:
            character: Character state object
            situation: Current situation description
            emphasis: Emphasis mode for response
            context: Additional context

        Returns:
            Rendered POML template as prompt string
        """
        # Use POML adapter for backward compatibility
        prompt = self.poml_adapter.get_character_prompt(
            character=character, situation=situation, emphasis=emphasis
        )

        # Add any additional context
        if context:
            context_prompt = self.poml.render(
                "components/context_extension.poml", {"context": context}
            )
            prompt = f"{prompt}\n\n{context_prompt}"

        return prompt

    async def run_simulation(
        self,
        character: CharacterState,
        situation: str,
        emphasis: str = "neutral",
        temperature: Optional[float] = None,
        use_poml: bool = True,
    ) -> Dict:
        """
        Run character simulation with POML templates

        Args:
            character: Character to simulate
            situation: Situation to respond to
            emphasis: Response emphasis mode
            temperature: LLM temperature override
            use_poml: Whether to use POML templates (default: True)

        Returns:
            Simulation results dictionary
        """
        async with self.semaphore:
            try:
                # Calculate temperature if not provided
                if temperature is None:
                    temperature = character.emotional_state.modulate_temperature()

                # Check cache first
                if self.cache:
                    cached_result = await self.cache.get_simulation(
                        character.id, situation, emphasis, temperature
                    )

                    if cached_result:
                        logger.info(
                            f"Using cached simulation for {character.name}/{emphasis}"
                        )
                        return cached_result

                # Generate prompt using POML or fallback to original
                if use_poml:
                    prompt = self.get_simulation_prompt(
                        character=character, situation=situation, emphasis=emphasis
                    )
                    logger.debug("Using POML template for prompt generation")
                else:
                    # Fallback to original string-based prompt
                    prompt = character.get_simulation_prompt(situation, emphasis)
                    logger.debug("Using original string-based prompt")

                # Call LLM with retry logic
                response = await self.retry_handler.execute_with_retry(
                    self.llm.generate_response, prompt, temperature
                )

                # Parse response
                try:
                    response_data = json.loads(response.content)

                    # Validate against POML schema if available
                    if use_poml and self.config.get("validate_schema", True):
                        self._validate_response_schema(response_data)

                    logger.info("Parsed structured response successfully")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse structured response: {e}")
                    # Fallback response
                    response_data = {
                        "dialogue": "I cannot respond properly at this time.",
                        "thought": "System error in processing.",
                        "action": "Pauses uncertainly",
                        "emotional_shift": {
                            "anger": 0,
                            "doubt": 0,
                            "fear": 0,
                            "compassion": 0,
                        },
                    }

                result = {
                    "character_id": character.id,
                    "situation": situation,
                    "emphasis": emphasis,
                    "temperature": temperature,
                    "response": response_data,
                    "metadata": {
                        **response.metadata,
                        "used_poml": use_poml,
                        "template": (
                            "character_response.poml" if use_poml else "string_based"
                        ),
                    },
                    "timestamp": response.timestamp,
                }

                # Update character state if emotional shift provided
                if "emotional_shift" in response_data:
                    await self._apply_emotional_shift(
                        character, response_data["emotional_shift"]
                    )

                # Cache the result
                if self.cache:
                    await self.cache.set_simulation(
                        character.id, situation, emphasis, temperature, result
                    )

                logger.info(
                    f"Simulation completed for {character.name} with emphasis '{emphasis}'"
                )
                return result

            except Exception as e:
                logger.error(f"Simulation failed: {e}")
                raise SimulationError(f"Simulation failed for {character.name}: {e}")

    def _validate_response_schema(self, response_data: Dict):
        """
        Validate response against POML schema

        Args:
            response_data: Response dictionary to validate
        """
        try:
            # Load schema from POML template
            schema_template = self.poml.render(
                "templates/schemas/simulation_response.poml", {}
            )

            # Extract JSON schema from rendered template
            import re

            schema_match = re.search(
                r"<json-schema>(.*?)</json-schema>", schema_template, re.DOTALL
            )

            if schema_match:
                schema = json.loads(schema_match.group(1))

                # Use jsonschema for validation
                from jsonschema import validate

                validate(instance=response_data, schema=schema)
                logger.debug("Response validated against POML schema")

        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")

    async def run_narrative_sequence(
        self, character: CharacterState, story_beats: List[Dict], use_poml: bool = True
    ) -> List[Dict]:
        """
        Run character through narrative sequence using POML templates

        Args:
            character: Character to simulate
            story_beats: List of story beat dictionaries
            use_poml: Whether to use POML templates

        Returns:
            List of simulation results for each beat
        """
        results = []

        for i, beat in enumerate(story_beats):
            # Generate scene using POML
            if use_poml:
                scene_prompt = self.poml.render(
                    "templates/narrative/scene_crafting.poml",
                    {
                        "beat": beat,
                        "characters": [{"name": character.name, "role": "protagonist"}],
                        "previous_context": (
                            results[-1]["response"]["dialogue"] if results else ""
                        ),
                    },
                )

                # Extract situation from scene
                situation = await self._extract_situation_from_scene(scene_prompt)
            else:
                situation = beat.get("situation", f"Scene {i+1}")

            # Run simulation for this beat
            emphasis = beat.get("emphasis", "neutral")
            result = await self.run_simulation(
                character=character,
                situation=situation,
                emphasis=emphasis,
                use_poml=use_poml,
            )

            results.append(result)

            # Update character memory
            character.memory.add_event(f"Scene {i+1}: {situation[:50]}...")

        return results

    async def _extract_situation_from_scene(self, scene_prompt: str) -> str:
        """
        Extract situation description from scene prompt

        Args:
            scene_prompt: Full scene prompt

        Returns:
            Situation description
        """
        # For now, return first paragraph
        # In production, could use LLM to extract
        lines = scene_prompt.split("\n")
        for line in lines:
            if line.strip() and not line.startswith("#"):
                return line.strip()
        return "A tense moment unfolds"

    async def generate_character_variant(
        self, base_character: CharacterState, variant_type: str
    ) -> CharacterState:
        """
        Generate character variant using POML templates

        Args:
            base_character: Base character to modify
            variant_type: Type of variant (e.g., 'ruthless', 'compassionate')

        Returns:
            Modified character state
        """
        # Render variant template
        self.poml.render(
            "templates/characters/character_variants.poml",
            {"base_character": base_character, "variant": variant_type},
        )

        # Parse variant modifications
        # In production, this would be more sophisticated
        import copy

        variant = copy.deepcopy(base_character)

        # Apply variant-specific modifications
        if variant_type == "ruthless":
            variant.emotional_state.compassion *= 0.3
            variant.emotional_state.anger *= 1.5
            variant.traits.append("ruthless")
        elif variant_type == "compassionate":
            variant.emotional_state.compassion *= 1.5
            variant.emotional_state.anger *= 0.5
            variant.traits.append("empathetic")

        return variant

    def get_available_templates(self) -> Dict[str, List[str]]:
        """
        List all available POML templates organized by category

        Returns:
            Dictionary of template categories and their files
        """
        templates = {
            "characters": [],
            "simulations": [],
            "narrative": [],
            "schemas": [],
            "components": [],
        }

        base_path = Path(__file__).parent.parent / "templates"

        for category in templates.keys():
            category_path = base_path / category
            if category_path.exists():
                templates[category] = [f.name for f in category_path.glob("*.poml")]

        return templates

    async def benchmark_poml_performance(
        self,
        character: CharacterState,
        test_situations: List[str],
        iterations: int = 10,
    ) -> Dict:
        """
        Benchmark POML vs string-based prompt performance

        Args:
            character: Character to test
            test_situations: List of test situations
            iterations: Number of iterations per test

        Returns:
            Benchmark results
        """
        import time

        results = {
            "poml": {"times": [], "cache_hits": 0},
            "string": {"times": [], "cache_hits": 0},
        }

        # Clear cache for fair comparison
        if self.cache:
            self.cache.clear()

        for situation in test_situations:
            for i in range(iterations):
                # Test POML
                start = time.time()
                await self.run_simulation(
                    character=character, situation=situation, use_poml=True
                )
                results["poml"]["times"].append(time.time() - start)

                # Test string-based
                start = time.time()
                await self.run_simulation(
                    character=character, situation=situation, use_poml=False
                )
                results["string"]["times"].append(time.time() - start)

        # Calculate statistics
        for method in results:
            times = results[method]["times"]
            results[method]["avg_time"] = sum(times) / len(times)
            results[method]["min_time"] = min(times)
            results[method]["max_time"] = max(times)

        return results


# Factory function for easy integration
def create_poml_engine(
    llm_client, config: Dict = None
) -> POMLCharacterSimulationEngine:
    """
    Create POML-enabled character simulation engine

    Args:
        llm_client: LLM client instance
        config: Configuration dictionary

    Returns:
        POMLCharacterSimulationEngine instance
    """
    return POMLCharacterSimulationEngine(llm_client, config)


# Migration helper
class SimulationEngineMigrator:
    """
    Helper class for migrating from original to POML engine
    """

    @staticmethod
    def migrate_config(old_config: Dict) -> Dict:
        """
        Migrate old configuration to POML-compatible format

        Args:
            old_config: Original configuration

        Returns:
            POML-compatible configuration
        """
        new_config = old_config.copy()

        # Add POML-specific settings
        new_config["validate_schema"] = True
        new_config["poml"] = {
            "cache_templates": True,
            "strict_mode": False,
            "debug": old_config.get("debug", False),
        }

        return new_config

    @staticmethod
    async def compare_outputs(
        engine: POMLCharacterSimulationEngine, character: CharacterState, situation: str
    ) -> Dict:
        """
        Compare POML and string-based outputs

        Args:
            engine: POML engine instance
            character: Character to test
            situation: Test situation

        Returns:
            Comparison results
        """
        # Run both versions
        poml_result = await engine.run_simulation(
            character=character, situation=situation, use_poml=True
        )

        string_result = await engine.run_simulation(
            character=character, situation=situation, use_poml=False
        )

        # Compare results
        comparison = {
            "poml_response": poml_result["response"],
            "string_response": string_result["response"],
            "differences": [],
        }

        # Find differences
        for key in ["dialogue", "thought", "action"]:
            if poml_result["response"][key] != string_result["response"][key]:
                comparison["differences"].append(
                    {
                        "field": key,
                        "poml": poml_result["response"][key],
                        "string": string_result["response"][key],
                    }
                )

        return comparison


# Example usage
if __name__ == "__main__":
    import asyncio

    # Example integration
    async def main():
        # Create mock LLM for testing
        from character_simulation_engine_v2 import MockLLM

        llm = MockLLM()

        # Create POML engine
        config = {"cache": {"enabled": True}, "validate_schema": True}

        engine = create_poml_engine(llm, config)

        # Create test character
        character = CharacterState(
            id="test_char",
            name="Test Character",
            backstory={"origin": "Test origin", "career": "Test career"},
            traits=["brave", "curious"],
            values=["truth", "justice"],
            fears=["failure"],
            desires=["success"],
            emotional_state=EmotionalState(
                anger=0.3, doubt=0.5, fear=0.2, compassion=0.6, confidence=0.7
            ),
            memory=CharacterMemory(),
            current_goal="Test the POML system",
        )

        # Run simulation with POML
        result = await engine.run_simulation(
            character=character,
            situation="You face a critical decision",
            emphasis="doubt",
            use_poml=True,
        )

        print("POML Simulation Result:")
        print(json.dumps(result["response"], indent=2))

        # List available templates
        templates = engine.get_available_templates()
        print("\nAvailable POML Templates:")
        for category, files in templates.items():
            print(f"  {category}: {files}")

    asyncio.run(main())
