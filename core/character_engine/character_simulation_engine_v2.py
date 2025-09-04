"""
Character Simulation Engine v2.0
Production-ready implementation with LLM abstraction and error handling
"""

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Protocol
from datetime import datetime
from pathlib import Path
import random
from enum import Enum
import traceback
import yaml

# Import cache manager
try:
    from cache_manager import SimulationCache, create_cache
except ImportError:
    SimulationCache = None
    create_cache = None
    logging.warning("Cache manager not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# HIGH PRIORITY: LLM Interface Abstraction
# ============================================================================

class LLMResponse:
    """Standardized LLM response format"""
    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

class LLMInterface(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_response(self, 
                               prompt: str, 
                               temperature: float = 0.7,
                               max_tokens: int = 500) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is available"""
        pass

class MockLLM(LLMInterface):
    """Mock LLM for testing and development"""
    
    def __init__(self):
        self.call_count = 0
        
    async def generate_response(self, 
                               prompt: str, 
                               temperature: float = 0.7,
                               max_tokens: int = 500) -> LLMResponse:
        """Generate mock response based on prompt content"""
        self.call_count += 1
        await asyncio.sleep(0.1)  # Simulate API latency
        
        # Parse emphasis from prompt if present
        if "power" in prompt.lower() and temperature > 0.9:
            content = json.dumps({
                "dialogue": "The empire demands order! This chaos ends now!",
                "thought": "They test my authority. I must demonstrate strength.",
                "action": "Strikes table with fist, rises to full height",
                "emotional_shift": {"anger": 0.1, "confidence": 0.15}
            })
        elif "doubt" in prompt.lower():
            content = json.dumps({
                "dialogue": "What is truth? You speak in riddles I cannot decipher...",
                "thought": "This man disturbs me. His certainty in the face of death...",
                "action": "Turns away, unable to maintain eye contact",
                "emotional_shift": {"doubt": 0.2, "fear": 0.1}
            })
        else:
            content = json.dumps({
                "dialogue": "The law is clear. I find no fault, yet the law must be upheld.",
                "thought": "I am trapped between justice and survival.",
                "action": "Paces deliberately, weighing each option",
                "emotional_shift": {"doubt": 0.05, "fear": 0.05}
            })
            
        return LLMResponse(content, {"model": "mock", "temperature": temperature})
    
    async def health_check(self) -> bool:
        """Mock health check always returns True"""
        return True

class OpenAILLM(LLMInterface):
    """OpenAI API implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        # In production, initialize OpenAI client here
        
    async def generate_response(self, 
                               prompt: str, 
                               temperature: float = 0.7,
                               max_tokens: int = 500) -> LLMResponse:
        """Generate response via OpenAI API"""
        try:
            # Production implementation would call OpenAI API
            # response = await openai_client.create_completion(...)
            logger.info(f"OpenAI API call: model={self.model}, temp={temperature}")
            
            # Placeholder for actual API call
            content = "OpenAI response would go here"
            return LLMResponse(content, {"model": self.model, "temperature": temperature})
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def health_check(self) -> bool:
        """Check OpenAI API availability"""
        try:
            # Would ping OpenAI API endpoint
            return True
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
            logger.warning(f'Connection error: {e}')
            return False

        except Exception as e:
            logger.error(f'Unexpected error: {e}')
            return False

class LMStudioLLM(LLMInterface):
    """LMStudio local model implementation with structured output support"""
    
    def __init__(self, endpoint: str = "http://localhost:1234/v1", model: str = None):
        self.endpoint = endpoint
        self.model = model  # Specific model to use
        
        # JSON Schema for character responses
        self.response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "character_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "dialogue": {
                            "type": "string",
                            "description": "What the character says"
                        },
                        "thought": {
                            "type": "string",
                            "description": "Internal monologue"
                        },
                        "action": {
                            "type": "string",
                            "description": "Physical actions taken"
                        },
                        "emotional_shift": {
                            "type": "object",
                            "properties": {
                                "anger": {"type": "number", "minimum": -1, "maximum": 1},
                                "doubt": {"type": "number", "minimum": -1, "maximum": 1},
                                "fear": {"type": "number", "minimum": -1, "maximum": 1},
                                "compassion": {"type": "number", "minimum": -1, "maximum": 1}
                            },
                            "required": ["anger", "doubt", "fear", "compassion"]
                        }
                    },
                    "required": ["dialogue", "thought", "action", "emotional_shift"]
                }
            }
        }
        
    async def generate_response(self, 
                               prompt: str, 
                               temperature: float = 0.7,
                               max_tokens: int = 500) -> LLMResponse:
        """Generate response via LMStudio API with structured output"""
        import aiohttp
        import json
        
        try:
            logger.info(f"LMStudio call: endpoint={self.endpoint}, temp={temperature}")
            
            # Prepare the request with structured output
            headers = {"Content-Type": "application/json"}
            
            system_content = "You are simulating a character. Respond based on the given scenario."
            
            data = {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                "response_format": self.response_schema,  # Use JSON Schema for structured output
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add model if specified
            if self.model:
                data["model"] = self.model
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.endpoint}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract content from response
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # With structured output, content is already a valid JSON string
                        # No need for complex parsing or fixing
                        
                        return LLMResponse(content, {
                            "endpoint": self.endpoint, 
                            "temperature": temperature,
                            "model": result.get("model", "unknown")
                        })
                    else:
                        error_text = await response.text()
                        logger.error(f"LMStudio API error: {response.status} - {error_text}")
                        raise Exception(f"LMStudio API error: {response.status}")
            
        except Exception as e:
            logger.error(f"LMStudio API error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check LMStudio availability"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.endpoint}/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
            logger.warning(f'Connection error: {e}')
            return False

        except Exception as e:
            logger.error(f'Unexpected error: {e}')
            return False

# ============================================================================
# HIGH PRIORITY: Error Handling and Retry Logic
# ============================================================================

class SimulationError(Exception):
    """Base exception for simulation errors"""
    pass

class LLMError(SimulationError):
    """LLM-specific errors"""
    pass

class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                
        logger.error(f"All {self.max_retries} attempts failed")
        raise LLMError(f"Failed after {self.max_retries} retries") from last_exception

# ============================================================================
# ENHANCED CHARACTER SYSTEM WITH ERROR HANDLING
# ============================================================================

@dataclass
class EmotionalState:
    """Tracks character's emotional state with validation"""
    anger: float = 0.0
    doubt: float = 0.0
    fear: float = 0.0
    compassion: float = 0.0
    confidence: float = 0.5
    
    def __post_init__(self):
        """Validate emotional values are in range"""
        for field_name in ['anger', 'doubt', 'fear', 'compassion', 'confidence']:
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                logger.warning(f"Clamping {field_name} from {value} to range [0,1]")
                setattr(self, field_name, max(0, min(1, value)))
    
    def to_dict(self):
        return asdict(self)
    
    def modulate_temperature(self) -> float:
        """Calculate dialogue temperature based on emotional state"""
        try:
            emotional_intensity = (self.anger + self.fear + abs(self.doubt - 0.5)) / 2
            temperature = 0.7 + (emotional_intensity * 0.5)
            return max(0.7, min(1.2, temperature))  # Clamp to valid range
        except Exception as e:
            logger.error(f"Error calculating temperature: {e}")
            return 0.7  # Safe default

@dataclass
class CharacterMemory:
    """Character's working memory with thread-safe operations"""
    recent_events: List[str] = field(default_factory=list)
    key_observations: List[str] = field(default_factory=list)
    unresolved_tensions: List[str] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    
    async def add_event(self, event: str):
        """Thread-safe event addition"""
        async with self._lock:
            self.recent_events.append(event)
            if len(self.recent_events) > 5:
                self.recent_events.pop(0)
            logger.debug(f"Added event to memory: {event}")

@dataclass
class CharacterState:
    """Complete character state with validation"""
    id: str
    name: str
    backstory: Dict[str, Any]
    traits: List[str]
    values: List[str]
    fears: List[str]
    desires: List[str]
    emotional_state: EmotionalState
    memory: CharacterMemory
    current_goal: str
    internal_conflict: Optional[str] = None
    relationship_map: Dict[str, Dict] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate character state integrity"""
        try:
            assert self.id, "Character ID is required"
            assert self.name, "Character name is required"
            assert len(self.traits) > 0, "At least one trait is required"
            assert len(self.values) > 0, "At least one value is required"
            return True
        except AssertionError as e:
            logger.error(f"Character validation failed: {e}")
            return False
    
    def get_simulation_prompt(self, situation: str, emphasis: str = "neutral") -> str:
        """Generate prompt with error handling"""
        try:
            if not self.validate():
                raise ValueError("Invalid character state")
                
            context = f"""You are {self.name}.
            
Background: {self.backstory.get('origin', 'Unknown')}
Career: {self.backstory.get('career', 'Unknown')}
Core Traits: {', '.join(self.traits)}
Values: {', '.join(self.values)}
Fears: {', '.join(self.fears)}
Current Goal: {self.current_goal}

Emotional State:
- Anger: {self.emotional_state.anger:.1f}/10
- Doubt: {self.emotional_state.doubt:.1f}/10  
- Fear: {self.emotional_state.fear:.1f}/10
- Compassion: {self.emotional_state.compassion:.1f}/10

Recent Events: {'; '.join(self.memory.recent_events[-3:]) if self.memory.recent_events else 'None'}
Internal Conflict: {self.internal_conflict or 'None'}

Situation: {situation}
Emphasis: {emphasis}

Respond with a JSON object containing:
- dialogue: What you say
- thought: Your internal monologue
- action: Physical actions or gestures
- emotional_shift: Changes to emotional state (as dictionary)
"""
            return context
            
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            raise SimulationError(f"Failed to generate prompt: {e}")

# ============================================================================
# ENHANCED SIMULATION ENGINE WITH CONCURRENCY CONTROL
# ============================================================================

class SimulationEngine:
    """Production-ready simulation engine with orchestrator/POML integration and caching"""
    
    def __init__(self, 
                 llm_provider: Optional[LLMInterface] = None,
                 max_concurrent: int = 10,
                 retry_handler: Optional[RetryHandler] = None,
                 cache: Optional[SimulationCache] = None,
                 config: Optional[Dict] = None,
                 orchestrator: Optional[Any] = None,
                 use_poml: Optional[bool] = None):
        self.llm = llm_provider
        self.orchestrator = orchestrator
        self.simulation_results = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.retry_handler = retry_handler or RetryHandler()
        self.cache = cache
        self.config = config or {}

        # POML adapter (optional)
        try:
            from poml.lib.poml_integration import StoryEnginePOMLAdapter
            self.poml_adapter = StoryEnginePOMLAdapter()
        except Exception:
            self.poml_adapter = None
        self.use_poml = bool(use_poml if use_poml is not None else self.config.get('simulation', {}).get('use_poml', False))
        self.validate_schema = bool(self.config.get('simulation', {}).get('validate_schema', True))
        
        # Initialize cache if not provided but enabled in config
        if self.cache is None and create_cache and config:
            self.cache = create_cache(config)
        
        backend_name = (
            orchestrator.__class__.__name__ if orchestrator is not None else (
                llm_provider.__class__.__name__ if llm_provider is not None else 'None'
            )
        )
        logger.info(f"Initialized SimulationEngine backend={backend_name} use_poml={self.use_poml}")
        
    async def run_simulation(self, 
                            character: CharacterState,
                            situation: str,
                            emphasis: str = "neutral",
                            temperature: Optional[float] = None,
                            max_tokens: Optional[int] = None) -> Dict:
        """Run single simulation with error handling, caching, and concurrency control"""
        
        async with self.semaphore:  # Limit concurrent simulations
            try:
                # Validate inputs
                if not character.validate():
                    raise SimulationError("Invalid character state")
                    
                if temperature is None:
                    temperature = character.emotional_state.modulate_temperature()
                
                # Check cache first
                if self.cache:
                    cached_result = await self.cache.get_simulation(
                        character.id,
                        situation,
                        emphasis,
                        temperature
                    )
                    
                    if cached_result:
                        logger.info(f"Using cached simulation for {character.name}/{emphasis}")
                        return cached_result
                    
                # Generate prompt (POML when enabled)
                if self.use_poml and self.poml_adapter:
                    prompt = self.poml_adapter.get_character_prompt(
                        character=character,
                        situation=situation,
                        emphasis=emphasis
                    )
                else:
                    prompt = character.get_simulation_prompt(situation, emphasis)

                # Prepare backend call
                if self.orchestrator is not None:
                    async def _call(p: str, t: float, mt: Optional[int]):
                        kwargs = {"allow_fallback": True, "temperature": t}
                        if mt is not None:
                            kwargs["max_tokens"] = mt
                        return await self.orchestrator.generate(p, **kwargs)
                else:
                    async def _call(p: str, t: float, mt: Optional[int]):
                        return await self.llm.generate_response(p, temperature=t, max_tokens=mt or 500)

                # Call backend with retry logic
                response = await self.retry_handler.execute_with_retry(_call, prompt, temperature, max_tokens)
                
                # Parse and validate structured response
                try:
                    raw_text = getattr(response, 'content', None) or getattr(response, 'text', '') or ''
                    response_data = json.loads(raw_text)
                    if self.validate_schema:
                        self._validate_character_response_shape(response_data)
                    logger.info("Parsed structured response successfully")
                except Exception as e:
                    logger.error(f"Failed to parse/validate structured response: {e}")
                    response_data = {
                        "dialogue": "I cannot respond properly at this time.",
                        "thought": "System error in processing.",
                        "action": "Pauses uncertainly",
                        "emotional_shift": {"anger": 0, "doubt": 0, "fear": 0, "compassion": 0}
                    }
                
                result = {
                    "character_id": character.id,
                    "situation": situation,
                    "emphasis": emphasis,
                    "temperature": temperature,
                    "response": response_data,
                    "metadata": {
                        **(getattr(response, 'metadata', {}) or {}),
                        "used_poml": self.use_poml,
                        "template": "character_response.poml" if self.use_poml else "string_based",
                    },
                    "timestamp": getattr(response, 'timestamp', datetime.now().isoformat())
                }
                
                # Update character state if emotional shift provided
                if "emotional_shift" in response_data:
                    await self._apply_emotional_shift(character, response_data["emotional_shift"])
                
                # Cache the result
                if self.cache:
                    await self.cache.set_simulation(
                        character.id,
                        situation,
                        emphasis,
                        temperature,
                        result
                    )
                
                logger.info(f"Simulation completed for {character.name} with emphasis '{emphasis}'")
                return result
                
            except Exception as e:
                logger.error(f"Simulation failed: {e}\n{traceback.format_exc()}")
                raise SimulationError(f"Simulation failed for {character.name}: {e}")
    
    async def _apply_emotional_shift(self, character: CharacterState, shifts: Dict[str, Any]):
        """Apply emotional changes to character"""
        try:
            for emotion, delta in shifts.items():
                # Normalize emotion key to lowercase
                emotion = emotion.lower()
                
                if hasattr(character.emotional_state, emotion):
                    current = getattr(character.emotional_state, emotion)
                    
                    # Parse delta value - handle strings, floats, etc.
                    if isinstance(delta, str):
                        # Remove + sign if present, convert to float
                        delta = float(delta.replace('+', ''))
                    elif not isinstance(delta, (int, float)):
                        logger.warning(f"Unexpected delta type for {emotion}: {type(delta)}")
                        continue
                    
                    # If value is > 1, assume it's absolute not delta
                    if abs(delta) > 1:
                        new_value = max(0, min(1, delta))
                    else:
                        new_value = max(0, min(1, current + delta))
                    
                    setattr(character.emotional_state, emotion, new_value)
                    logger.debug(f"Updated {emotion}: {current:.2f} -> {new_value:.2f}")
        except Exception as e:
            logger.error(f"Failed to apply emotional shift: {e}")
    
    async def run_multiple_simulations(self,
                                      character: CharacterState,
                                      situation: str,
                                      num_runs: int = 10,
                                      emphases: Optional[List[str]] = None,
                                      fixed_temperature: Optional[float] = None,
                                      max_tokens: Optional[int] = None) -> List[Dict]:
        """Run multiple simulations with varied parameters"""
        
        if emphases is None:
            emphases = ["power", "doubt", "fear", "duty", "compassion", "pragmatic"]
            
        temperatures = [0.7, 0.8, 0.9, 1.0, 1.1]
        
        tasks = []
        for i in range(num_runs):
            emphasis = random.choice(emphases)
            temp = fixed_temperature if fixed_temperature is not None else random.choice(temperatures)
            
            # Create variation in emotional state for diversity
            if i > 0:
                # Make a copy to avoid modifying original
                import copy
                char_copy = copy.deepcopy(character)
                char_copy.emotional_state.doubt += random.uniform(-0.1, 0.1)
                char_copy.emotional_state.fear += random.uniform(-0.1, 0.1)
                char_copy.emotional_state.__post_init__()  # Re-validate
            else:
                char_copy = character
            
            task = self.run_simulation(char_copy, situation, emphasis, temp, max_tokens=max_tokens)
            tasks.append(task)
        
        # Gather results with error handling
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Individual simulation failed: {e}")
                # Continue with other simulations
                
        self.simulation_results.extend(results)
        logger.info(f"Completed {len(results)}/{num_runs} simulations successfully")
        return results

    def _validate_character_response_shape(self, payload: Dict[str, Any]) -> None:
        """Minimal schema check for character response shape."""
        if not isinstance(payload, dict):
            raise ValueError("Response is not a JSON object")
        required = ["dialogue", "thought", "action", "emotional_shift"]
        for key in required:
            if key not in payload:
                raise ValueError(f"Missing key in response: {key}")
        if not isinstance(payload.get("emotional_shift", {}), dict):
            raise ValueError("emotional_shift must be an object")

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Demonstrate production-ready character simulation with caching"""
    
    # Load configuration
    config = {}
    config_path = Path("simulation_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Loaded configuration from simulation_config.yaml")
    
    # Initialize LLM provider based on config
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'mock')
    
    if provider == 'mock':
        llm = MockLLM()
    elif provider == 'openai':
        llm = OpenAILLM(
            api_key=llm_config['openai']['api_key'],
            model=llm_config['openai']['model']
        )
    elif provider == 'lmstudio':
        llm = LMStudioLLM(
            endpoint=llm_config['lmstudio']['endpoint']
        )
    else:
        llm = MockLLM()
    
    # Check LLM health
    if not await llm.health_check():
        logger.error("LLM provider is not available")
        return
    
    # Create cache
    cache = None
    if create_cache:
        cache = create_cache(config)
    
    # Create simulation engine with configuration
    sim_config = config.get('simulation', {})
    engine = SimulationEngine(
        llm_provider=llm,
        max_concurrent=sim_config.get('max_concurrent', 10),
        retry_handler=RetryHandler(
            max_retries=sim_config.get('retry', {}).get('max_attempts', 3),
            base_delay=sim_config.get('retry', {}).get('base_delay', 1.0)
        ),
        cache=cache,
        config=config
    )
    
    # Create character
    pilate = CharacterState(
        id="pontius_pilate",
        name="Pontius Pilate",
        backstory={
            "origin": "Roman equestrian from Samnium",
            "career": "Prefect of Judaea (26-36 CE)"
        },
        traits=["pragmatic", "ambitious", "anxious"],
        values=["order", "duty", "Roman law"],
        fears=["rebellion", "imperial disfavor"],
        desires=["peace", "advancement"],
        emotional_state=EmotionalState(
            anger=0.4,
            doubt=0.7,
            fear=0.6,
            compassion=0.3
        ),
        memory=CharacterMemory(
            recent_events=["Crowd demanding justice", "Wife's warning dream"]
        ),
        current_goal="Maintain order without rebellion",
        internal_conflict="Duty to Rome vs. sense of justice"
    )
    
    # Run simulations
    situation = "You face Jesus of Nazareth. The crowd demands action. You must decide."
    
    try:
        results = await engine.run_multiple_simulations(
            pilate, 
            situation, 
            num_runs=3
        )
        
        logger.info(f"Successfully generated {len(results)} simulations")
        
        # Save results
        with open("simulation_results_v2.json", "w") as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
