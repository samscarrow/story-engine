"""
LLM Orchestration Layer
Agnostic interface for multiple LLM providers (LMStudio, KoboldCpp, OpenAI, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Dict] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available"""
        pass
    
    @abstractmethod
    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format prompt for specific provider"""
        pass


class LMStudioProvider(LLMProvider):
    """LMStudio API provider"""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using LMStudio chat completions API"""
        url = f"{self.config.endpoint}/v1/chat/completions"
        
        # Merge kwargs with config
        temperature = kwargs.get('temperature', self.config.temperature)
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        
        payload = {
            "model": self.config.model or "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    data = await response.json()
                    
                    return LLMResponse(
                        text=data['choices'][0]['message']['content'],
                        provider=ModelProvider.LMSTUDIO,
                        model=data.get('model'),
                        usage=data.get('usage'),
                        raw_response=data
                    )
        except Exception as e:
            logger.error(f"LMStudio generation error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check LMStudio availability"""
        try:
            url = f"{self.config.endpoint}/v1/models"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format for chat completion"""
        if system:
            return f"{system}\n\n{prompt}"
        return prompt


class KoboldCppProvider(LLMProvider):
    """KoboldCpp API provider"""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using KoboldCpp API"""
        url = f"{self.config.endpoint}/api/v1/generate"
        
        # Build payload with KoboldCpp parameters
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
        
        # Add any extra parameters
        payload.update(self.config.extra_params)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    data = await response.json()
                    
                    # Extract text from results
                    if 'results' in data and len(data['results']) > 0:
                        text = data['results'][0]['text']
                    else:
                        raise ValueError("No results in KoboldCpp response")
                    
                    return LLMResponse(
                        text=text,
                        provider=ModelProvider.KOBOLDCPP,
                        model=self.config.model,
                        raw_response=data
                    )
        except Exception as e:
            logger.error(f"KoboldCpp generation error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check KoboldCpp availability"""
        try:
            url = f"{self.config.endpoint}/api/v1/model"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format prompt for KoboldCpp"""
        if system:
            return f"### System: {system}\n### User: {prompt}\n### Assistant:"
        return prompt


class OllamaProvider(LLMProvider):
    """Ollama API provider"""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using Ollama API"""
        url = f"{self.config.endpoint}/api/generate"
        
        payload = {
            "model": self.config.model or "llama2",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "top_k": kwargs.get('top_k', self.config.top_k),
                "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
                "repeat_penalty": kwargs.get('rep_pen', self.config.repetition_penalty)
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    data = await response.json()
                    
                    return LLMResponse(
                        text=data['response'],
                        provider=ModelProvider.OLLAMA,
                        model=data.get('model'),
                        raw_response=data
                    )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Ollama availability"""
        try:
            url = f"{self.config.endpoint}/api/tags"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format prompt for Ollama"""
        if system:
            return f"System: {system}\nUser: {prompt}\nAssistant:"
        return prompt


class LLMOrchestrator:
    """Main orchestration class for managing multiple LLM providers"""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.active_provider: Optional[str] = None
        self.fallback_chain: List[str] = []
        
    def register_provider(self, name: str, config: LLMConfig) -> None:
        """Register a new LLM provider"""
        # Create appropriate provider instance
        if config.provider == ModelProvider.LMSTUDIO:
            provider = LMStudioProvider(config)
        elif config.provider == ModelProvider.KOBOLDCPP:
            provider = KoboldCppProvider(config)
        elif config.provider == ModelProvider.OLLAMA:
            provider = OllamaProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        self.providers[name] = provider
        
        # Set as active if first provider
        if self.active_provider is None:
            self.active_provider = name
        
        logger.info(f"Registered provider: {name} ({config.provider.value})")
    
    def set_active(self, name: str) -> None:
        """Set the active provider"""
        if name not in self.providers:
            raise ValueError(f"Provider {name} not registered")
        self.active_provider = name
        logger.info(f"Active provider set to: {name}")
    
    def set_fallback_chain(self, provider_names: List[str]) -> None:
        """Set fallback provider chain"""
        for name in provider_names:
            if name not in self.providers:
                raise ValueError(f"Provider {name} not registered")
        self.fallback_chain = provider_names
        logger.info(f"Fallback chain: {' -> '.join(provider_names)}")
    
    async def generate(
        self, 
        prompt: str,
        provider_name: str = None,
        use_fallback: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using specified or active provider
        
        Args:
            prompt: The input prompt
            provider_name: Specific provider to use (optional)
            use_fallback: Whether to use fallback chain on failure
            **kwargs: Additional generation parameters
        """
        # Determine provider to use
        if provider_name:
            if provider_name not in self.providers:
                raise ValueError(f"Provider {provider_name} not registered")
            providers_to_try = [provider_name]
        else:
            providers_to_try = [self.active_provider] if self.active_provider else []
        
        # Add fallback chain if enabled
        if use_fallback:
            providers_to_try.extend([p for p in self.fallback_chain if p not in providers_to_try])
        
        # Try each provider
        for name in providers_to_try:
            provider = self.providers[name]
            
            # Check health first
            try:
                is_healthy = await provider.health_check()
                if not is_healthy:
                    logger.warning(f"Provider {name} health check failed")
                    continue
            except Exception as e:
                logger.warning(f"Provider {name} health check error: {e}")
                continue
            
            # Try generation
            try:
                logger.info(f"Generating with provider: {name}")
                response = await provider.generate(prompt, **kwargs)
                logger.info(f"Generation successful with {name}")
                return response
            except Exception as e:
                logger.error(f"Provider {name} generation failed: {e}")
                continue
        
        raise RuntimeError("All providers failed to generate response")
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered providers"""
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = await provider.health_check()
            except:
                results[name] = False
        return results
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'LLMOrchestrator':
        """Create orchestrator from configuration file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        orchestrator = cls()
        
        # Register providers
        for provider_config in config.get('providers', []):
            name = provider_config.pop('name')
            provider_type = ModelProvider(provider_config.pop('provider'))
            llm_config = LLMConfig(provider=provider_type, **provider_config)
            orchestrator.register_provider(name, llm_config)
        
        # Set active provider
        if 'active' in config:
            orchestrator.set_active(config['active'])
        
        # Set fallback chain
        if 'fallback_chain' in config:
            orchestrator.set_fallback_chain(config['fallback_chain'])
        
        return orchestrator


# Example usage and testing
async def test_orchestrator():
    """Test the orchestrator with multiple providers"""
    
    print("🎯 TESTING LLM ORCHESTRATOR")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = LLMOrchestrator()
    
    # Register LMStudio
    orchestrator.register_provider(
        "lmstudio",
        LLMConfig(
            provider=ModelProvider.LMSTUDIO,
            endpoint="http://localhost:1234",
            model="gemma-2-27b",
            temperature=0.7,
            max_tokens=500
        )
    )
    
    # Register KoboldCpp
    orchestrator.register_provider(
        "kobold",
        LLMConfig(
            provider=ModelProvider.KOBOLDCPP,
            endpoint="http://localhost:5001",
            temperature=0.5,
            max_tokens=400,
            extra_params={"sampler_order": [6, 0, 1, 3, 4, 2, 5]}
        )
    )
    
    # Register Ollama
    orchestrator.register_provider(
        "ollama",
        LLMConfig(
            provider=ModelProvider.OLLAMA,
            endpoint="http://localhost:11434",
            model="llama2",
            temperature=0.6
        )
    )
    
    # Set fallback chain
    orchestrator.set_fallback_chain(["lmstudio", "kobold", "ollama"])
    
    # Test health checks
    print("\n📊 Health Checks:")
    health_results = await orchestrator.health_check_all()
    for name, status in health_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {name}: {'Available' if status else 'Unavailable'}")
    
    # Test generation
    print("\n💬 Testing Generation:")
    test_prompt = "Write a one-sentence story about a dragon."
    
    try:
        response = await orchestrator.generate(test_prompt, max_tokens=100)
        print(f"\n📝 Response from {response.provider.value}:")
        print(f"  {response.text[:200]}")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
    
    print("\n" + "=" * 70)
    print("✨ Orchestrator test complete!")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())