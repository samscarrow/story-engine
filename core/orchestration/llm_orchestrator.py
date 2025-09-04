"""
LLM Orchestration Layer - Strict Version
No silent failures, no mock fallbacks, explicit error handling
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import aiohttp
import json
import os
from datetime import datetime
import logging
import traceback

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported LLM providers"""
    LMSTUDIO = "lmstudio"
    KOBOLDCPP = "koboldcpp"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class GenerationError(Exception):
    """Specific exception for generation failures"""
    def __init__(self, provider: str, original_error: Exception, details: Dict = None):
        self.provider = provider
        self.original_error = original_error
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(f"Generation failed on {provider}: {original_error}")


@dataclass
class ProviderFailure:
    """Record of a provider failure"""
    provider_name: str
    error_type: str
    error_message: str
    timestamp: str
    request_prompt: str
    traceback: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "provider": self.provider_name,
            "error_type": self.error_type,
            "error": self.error_message,
            "timestamp": self.timestamp,
            "prompt_preview": self.request_prompt[:200] + "..." if len(self.request_prompt) > 200 else self.request_prompt,
            "traceback": self.traceback
        }


@dataclass
class LLMConfig:
    """Universal configuration for LLM providers"""
    provider: ModelProvider
    endpoint: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = None
    timeout: int = 60
    extra_params: Dict[str, Any] = None
    require_explicit_success: bool = True  # Must get valid response
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class LLMResponse:
    """Standardized response format"""
    text: str
    provider: ModelProvider
    provider_name: str
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Dict] = None
    timestamp: str = None
    generation_time_ms: Optional[float] = None
    failures_before_success: List[ProviderFailure] = field(default_factory=list)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # Validate response is not empty or mock
        if not self.text or self.text.strip() == "":
            raise ValueError("Empty response text")
        
        # Check for mock/placeholder patterns
        mock_patterns = [
            "[mock response]",
            "mock llm",
            "placeholder text",
            "lorem ipsum",
            "test response"
        ]
        
        text_lower = self.text.lower()
        for pattern in mock_patterns:
            if pattern in text_lower:
                raise ValueError(f"Mock/placeholder response detected: {pattern}")


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure: Optional[ProviderFailure] = None
        
    @abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate text from prompt - must return real response or raise exception"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health - return detailed status"""
        pass
    
    @abstractmethod
    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format prompt for specific provider"""
        pass
    
    def record_failure(self, error: Exception, prompt: str) -> ProviderFailure:
        """Record a failure for debugging"""
        self.failure_count += 1
        failure = ProviderFailure(
            provider_name=self.__class__.__name__,
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.now().isoformat(),
            request_prompt=prompt,
            traceback=traceback.format_exc()
        )
        self.last_failure = failure
        return failure


class LMStudioProvider(LLMProvider):
    """LMStudio API provider"""
    
    async def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate using LMStudio chat completions API"""
        start_time = asyncio.get_event_loop().time()
        url = f"{self.config.endpoint}/v1/chat/completions"
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model or "default",
            "messages": messages,
            "temperature": kwargs.get('temperature', self.config.temperature),
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise GenerationError(
                            "lmstudio",
                            Exception(f"HTTP {response.status}: {error_text}"),
                            {"status": response.status, "response": error_text}
                        )
                    
                    data = await response.json()
                    
                    # Validate response structure
                    if 'choices' not in data or len(data['choices']) == 0:
                        raise GenerationError(
                            "lmstudio",
                            ValueError("Invalid response structure: no choices"),
                            {"response": data}
                        )
                    # LM Studio may return either OpenAI-style message.content or text
                    choice = data['choices'][0]
                    text = ''
                    if isinstance(choice, dict):
                        # Prefer chat message content
                        text = (
                            (choice.get('message') or {}).get('content')
                            if isinstance(choice.get('message'), dict)
                            else ''
                        )
                        if not text:
                            text = choice.get('text', '') or ''
                    
                    # Create response with validation
                    return LLMResponse(
                        text=text,
                        provider=ModelProvider.LMSTUDIO,
                        provider_name="lmstudio",
                        model=data.get('model'),
                        usage=data.get('usage'),
                        raw_response=data,
                        generation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
                    )
                    
        except GenerationError:
            raise
        except Exception as e:
            failure = self.record_failure(e, prompt)
            logger.error(f"LMStudio generation failed: {failure.to_dict()}")
            raise GenerationError("lmstudio", e, {"failure": failure.to_dict()})
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LMStudio availability with details"""
        try:
            url = f"{self.config.endpoint}/v1/models"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "healthy": True,
                            "models": data.get('data', []),
                            "endpoint": self.config.endpoint
                        }
                    else:
                        return {
                            "healthy": False,
                            "error": f"HTTP {response.status}",
                            "endpoint": self.config.endpoint
                        }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "endpoint": self.config.endpoint
            }
    
    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format for chat completion"""
        if system:
            return f"{system}\n\n{prompt}"
        return prompt


class KoboldCppProvider(LLMProvider):
    """KoboldCpp API provider"""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using KoboldCpp API"""
        start_time = asyncio.get_event_loop().time()
        url = f"{self.config.endpoint}/api/v1/generate"
        
        payload = {
            "prompt": prompt,
            "max_context_length": kwargs.get('max_context', 4096),
            "max_length": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "top_k": kwargs.get('top_k', self.config.top_k),
            "rep_pen": kwargs.get('rep_pen', self.config.repetition_penalty),
            "rep_pen_range": kwargs.get('rep_pen_range', 320),
            "sampler_order": kwargs.get('sampler_order', [6, 0, 1, 3, 4, 2, 5]),
            "stop_sequence": self.config.stop_sequences,
            "trim_stop": True
        }
        
        payload.update(self.config.extra_params)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise GenerationError(
                            "koboldcpp",
                            Exception(f"HTTP {response.status}: {error_text}"),
                            {"status": response.status, "response": error_text}
                        )
                    
                    data = await response.json()
                    
                    # Validate KoboldCpp response
                    if 'results' not in data or len(data['results']) == 0:
                        raise GenerationError(
                            "koboldcpp",
                            ValueError("No results in response"),
                            {"response": data}
                        )
                    
                    text = data['results'][0].get('text', '')
                    
                    return LLMResponse(
                        text=text,
                        provider=ModelProvider.KOBOLDCPP,
                        provider_name="koboldcpp",
                        model=self.config.model,
                        raw_response=data,
                        generation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
                    )
                    
        except GenerationError:
            raise
        except Exception as e:
            failure = self.record_failure(e, prompt)
            logger.error(f"KoboldCpp generation failed: {failure.to_dict()}")
            raise GenerationError("koboldcpp", e, {"failure": failure.to_dict()})
    
    async def health_check(self) -> Dict[str, Any]:
        """Check KoboldCpp availability"""
        try:
            url = f"{self.config.endpoint}/api/v1/model"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "healthy": True,
                            "model": data.get('result'),
                            "endpoint": self.config.endpoint
                        }
                    else:
                        return {
                            "healthy": False,
                            "error": f"HTTP {response.status}",
                            "endpoint": self.config.endpoint
                        }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "endpoint": self.config.endpoint
            }
    
    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format prompt for KoboldCpp"""
        if system:
            return f"### System: {system}\n### User: {prompt}\n### Assistant:"
        return prompt


class LLMOrchestrator:
    """Orchestrator with strict failure handling - no silent fallbacks"""
    
    def __init__(self, fail_on_all_providers: bool = True):
        self.providers: Dict[str, LLMProvider] = {}
        self.active_provider: Optional[str] = None
        self.fail_on_all_providers = fail_on_all_providers
        self.generation_history: List[Dict] = []
        
    def register_provider(self, name: str, config: LLMConfig) -> None:
        """Register a new LLM provider"""
        if config.provider == ModelProvider.LMSTUDIO:
            provider = LMStudioProvider(config)
        elif config.provider == ModelProvider.KOBOLDCPP:
            provider = KoboldCppProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        self.providers[name] = provider
        
        if self.active_provider is None:
            self.active_provider = name
        
        logger.info(f"Registered provider: {name} ({config.provider.value})")
    
    def set_active(self, name: str) -> None:
        """Set the active provider"""
        if name not in self.providers:
            raise ValueError(f"Provider {name} not registered")
        self.active_provider = name
        logger.info(f"Active provider set to: {name}")
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        provider_name: str = None,
        allow_fallback: bool = False,
        fallback_providers: List[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text with explicit failure handling
        
        Args:
            prompt: Input prompt
            provider_name: Specific provider (uses active if None)
            allow_fallback: Explicitly allow fallback to other providers
            fallback_providers: Explicit list of fallback providers
            **kwargs: Generation parameters
        """
        
        # Track all failures
        all_failures: List[ProviderFailure] = []
        
        # Determine providers to try
        providers_to_try = []
        
        if provider_name:
            if provider_name not in self.providers:
                raise ValueError(f"Provider {provider_name} not registered")
            providers_to_try = [provider_name]
        else:
            if not self.active_provider:
                raise ValueError("No active provider set")
            providers_to_try = [self.active_provider]
        
        # Add explicit fallbacks if allowed
        if allow_fallback and fallback_providers:
            for fb in fallback_providers:
                if fb not in self.providers:
                    logger.warning(f"Fallback provider {fb} not registered, skipping")
                elif fb not in providers_to_try:
                    providers_to_try.append(fb)
        
        # Try each provider
        for provider_name in providers_to_try:
            provider = self.providers[provider_name]
            
            logger.info(f"Attempting generation with provider: {provider_name}")
            
            # Health check first
            health = await provider.health_check()
            if not health.get('healthy', False):
                failure = ProviderFailure(
                    provider_name=provider_name,
                    error_type="HealthCheckFailed",
                    error_message=f"Provider unhealthy: {health.get('error', 'Unknown error')}",
                    timestamp=datetime.now().isoformat(),
                    request_prompt=prompt
                )
                all_failures.append(failure)
                logger.error(f"Provider {provider_name} health check failed: {health}")
                continue
            
            # Try generation
            try:
                response = await provider.generate(prompt, system=system, **kwargs)
                
                # Add failure history to response
                response.failures_before_success = all_failures
                
                # Log success with context
                logger.info(
                    f"Generation successful with {provider_name} "
                    f"(after {len(all_failures)} failures) "
                    f"in {response.generation_time_ms:.1f}ms"
                )
                
                # Record in history
                self.generation_history.append({
                    "timestamp": response.timestamp,
                    "provider": provider_name,
                    "prompt_preview": prompt[:100],
                    "success": True,
                    "failures_before": len(all_failures),
                    "generation_time_ms": response.generation_time_ms
                })
                
                return response
                
            except GenerationError as e:
                all_failures.append(ProviderFailure(
                    provider_name=provider_name,
                    error_type=type(e.original_error).__name__,
                    error_message=str(e),
                    timestamp=e.timestamp,
                    request_prompt=prompt,
                    traceback=traceback.format_exc()
                ))
                logger.error(f"Generation failed on {provider_name}: {e}")
                
            except Exception as e:
                all_failures.append(ProviderFailure(
                    provider_name=provider_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    timestamp=datetime.now().isoformat(),
                    request_prompt=prompt,
                    traceback=traceback.format_exc()
                ))
                logger.error(f"Unexpected error on {provider_name}: {e}")
        
        # All providers failed - provide detailed error report
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "prompt_preview": prompt[:200],
            "providers_tried": providers_to_try,
            "all_failures": [f.to_dict() for f in all_failures]
        }
        
        # Record failure in history
        self.generation_history.append({
            "timestamp": error_report["timestamp"],
            "providers_tried": providers_to_try,
            "prompt_preview": prompt[:100],
            "success": False,
            "error_report": error_report
        })
        
        # Save error report
        error_file = f"generation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        error_msg = (
            f"All {len(providers_to_try)} providers failed.\n"
            f"Providers tried: {', '.join(providers_to_try)}\n"
            f"Error report saved to: {error_file}\n"
            f"First failure: {all_failures[0].error_message if all_failures else 'Unknown'}"
        )
        
        raise RuntimeError(error_msg)
    
    async def health_check_all(self) -> Dict[str, Dict]:
        """Detailed health check of all providers"""
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.health_check()
        return results
    
    def get_generation_stats(self) -> Dict:
        """Get statistics about generation history"""
        if not self.generation_history:
            return {"total": 0, "success": 0, "failure": 0}
        
        success = sum(1 for h in self.generation_history if h.get('success', False))
        failure = len(self.generation_history) - success
        
        avg_time = None
        if success > 0:
            times = [h['generation_time_ms'] for h in self.generation_history 
                    if h.get('success') and h.get('generation_time_ms')]
            if times:
                avg_time = sum(times) / len(times)
        
        return {
            "total": len(self.generation_history),
            "success": success,
            "failure": failure,
            "success_rate": (success / len(self.generation_history) * 100) if self.generation_history else 0,
            "avg_generation_time_ms": avg_time,
            "providers_used": list(set(h.get('provider', 'unknown') for h in self.generation_history if h.get('success')))
        }
    
    @classmethod
    def from_config_file(cls, config_path: str, strict: bool = True) -> 'LLMOrchestrator':
        """Create orchestrator from configuration file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        orchestrator = cls(fail_on_all_providers=strict)
        
        for provider_config in config.get('providers', []):
            name = provider_config.pop('name')
            provider_type = ModelProvider(provider_config.pop('provider'))
            provider_config['require_explicit_success'] = True
            llm_config = LLMConfig(provider=provider_type, **provider_config)
            orchestrator.register_provider(name, llm_config)
        
        if 'active' in config:
            orchestrator.set_active(config['active'])
        
        return orchestrator


# Test the strict orchestrator
async def test_orchestrator():
    """Test strict orchestration with no silent failures"""
    
    print("🔒 TESTING LLM ORCHESTRATOR")
    print("=" * 70)
    print("No mock responses, no silent failures")
    print()
    
    orchestrator = LLMOrchestrator(fail_on_all_providers=True)
    
    # Register only real providers
    orchestrator.register_provider(
        "kobold",
        LLMConfig(
            provider=ModelProvider.KOBOLDCPP,
            endpoint="http://localhost:5001",
            temperature=0.5,
            max_tokens=200,
            require_explicit_success=True
        )
    )
    
    orchestrator.register_provider(
        "lmstudio",
        LLMConfig(
            provider=ModelProvider.LMSTUDIO,
            endpoint="http://localhost:1234",
            model="gemma-2-27b",
            temperature=0.7,
            max_tokens=200,
            require_explicit_success=True
        )
    )
    
    # Health check
    print("📊 Provider Health Status:")
    health = await orchestrator.health_check_all()
    for name, status in health.items():
        icon = "✅" if status.get('healthy') else "❌"
        print(f"  {icon} {name}: {status}")
    print()
    
    # Test generation
    test_prompts = [
        "Write one sentence about the ocean.",
        "What is 2+2?",
        "Name three colors."
    ]
    
    for prompt in test_prompts:
        print(f"📝 Prompt: {prompt}")
        try:
            # Try with explicit provider
            response = await orchestrator.generate(
                prompt,
                provider_name="kobold",
                allow_fallback=True,
                fallback_providers=["lmstudio"]
            )
            print(f"✅ Response: {response.text[:100]}")
            if response.failures_before_success:
                print(f"   (After {len(response.failures_before_success)} failures)")
        except Exception as e:
            print(f"❌ Failed: {e}")
        print()
    
    # Show statistics
    stats = orchestrator.get_generation_stats()
    print("📈 Generation Statistics:")
    print(f"  Total attempts: {stats['total']}")
    print(f"  Success: {stats['success']} ({stats['success_rate']:.1f}%)")
    print(f"  Failures: {stats['failure']}")
    if stats['avg_generation_time_ms']:
        print(f"  Avg time: {stats['avg_generation_time_ms']:.1f}ms")
    
    print("\n✨ Orchestration test complete!")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())
