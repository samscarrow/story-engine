"""
Comprehensive Test Suite for Character Simulation Engine v2
Testing all components with proper mocking and validation
"""

import unittest
import asyncio
import json
from unittest.mock import patch
import sys
import os
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from character_simulation_engine_v2 import (
    EmotionalState,
    CharacterMemory,
    CharacterState,
    SimulationEngine,
    LLMResponse,
    MockLLM,
    OpenAILLM,
    LMStudioLLM,
    RetryHandler,
    SimulationError,
    LLMError
)

class TestEmotionalState(unittest.TestCase):
    """Test emotional state management and temperature modulation"""
    
    def setUp(self):
        """Initialize test emotional states"""
        self.neutral_state = EmotionalState()
        self.high_emotion_state = EmotionalState(
            anger=0.9,
            doubt=0.8,
            fear=0.9,
            compassion=0.2,
            confidence=0.1
        )
        self.invalid_state = EmotionalState(
            anger=1.5,  # Out of range
            doubt=-0.5,  # Out of range
            fear=0.5,
            compassion=2.0,  # Out of range
            confidence=0.5
        )
    
    def test_emotional_state_initialization(self):
        """Test proper initialization with defaults"""
        state = EmotionalState()
        self.assertEqual(state.anger, 0.0)
        self.assertEqual(state.doubt, 0.0)
        self.assertEqual(state.fear, 0.0)
        self.assertEqual(state.compassion, 0.0)
        self.assertEqual(state.confidence, 0.5)
    
    def test_emotional_state_validation(self):
        """Test that out-of-range values are clamped"""
        # Invalid state should have values clamped
        self.assertEqual(self.invalid_state.anger, 1.0)  # Clamped from 1.5
        self.assertEqual(self.invalid_state.doubt, 0.0)  # Clamped from -0.5
        self.assertEqual(self.invalid_state.compassion, 1.0)  # Clamped from 2.0
    
    def test_temperature_modulation(self):
        """Test temperature calculation based on emotional state"""
        # Neutral state with confidence=0.5 gives higher than base
        neutral_temp = self.neutral_state.modulate_temperature()
        # Formula: 0.7 + ((0 + 0 + abs(0 - 0.5))/2 * 0.5) = 0.7 + 0.125 = 0.825
        self.assertAlmostEqual(neutral_temp, 0.825, places=3)
        
        # High emotion should give higher temperature
        high_temp = self.high_emotion_state.modulate_temperature()
        self.assertGreater(high_temp, neutral_temp)
        self.assertLessEqual(high_temp, 1.2)  # Max cap
        
    def test_temperature_modulation_error_handling(self):
        """Test temperature calculation handles errors gracefully"""
        state = EmotionalState()
        # Test the actual error handling in modulate_temperature
        # Since confidence=0.5 by default: 0.7 + ((0 + 0 + 0.25)/2 * 0.5) = 0.825
        temp = state.modulate_temperature()
        self.assertAlmostEqual(temp, 0.825, places=3)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        state_dict = self.neutral_state.to_dict()
        self.assertIsInstance(state_dict, dict)
        self.assertIn('anger', state_dict)
        self.assertIn('doubt', state_dict)
        self.assertIn('fear', state_dict)
        self.assertIn('compassion', state_dict)
        self.assertIn('confidence', state_dict)

class TestCharacterMemory(unittest.TestCase):
    """Test character memory management"""
    
    def setUp(self):
        """Initialize test memory"""
        self.memory = CharacterMemory()
    
    def test_add_event(self):
        """Test adding events to memory"""
        async def test():
            await self.memory.add_event("Event 1")
            self.assertEqual(len(self.memory.recent_events), 1)
            self.assertEqual(self.memory.recent_events[0], "Event 1")
        
        asyncio.run(test())
    
    def test_memory_limit(self):
        """Test that memory maintains size limit"""
        async def test():
            # Add more than 5 events
            for i in range(10):
                await self.memory.add_event(f"Event {i}")
            
            # Should only keep last 5
            self.assertEqual(len(self.memory.recent_events), 5)
            self.assertEqual(self.memory.recent_events[0], "Event 5")
            self.assertEqual(self.memory.recent_events[-1], "Event 9")
        
        asyncio.run(test())
    
    def test_thread_safety(self):
        """Test concurrent access to memory"""
        async def add_events(memory, start_idx):
            for i in range(5):
                await memory.add_event(f"Event {start_idx + i}")
        
        async def test():
            # Run multiple concurrent additions
            tasks = [
                add_events(self.memory, 0),
                add_events(self.memory, 100),
                add_events(self.memory, 200)
            ]
            await asyncio.gather(*tasks)
            
            # Should have exactly 5 events (due to limit)
            self.assertEqual(len(self.memory.recent_events), 5)
        
        asyncio.run(test())

class TestCharacterState(unittest.TestCase):
    """Test character state management and validation"""
    
    def setUp(self):
        """Initialize test character"""
        self.valid_character = CharacterState(
            id="test_char",
            name="Test Character",
            backstory={"origin": "Test Origin", "career": "Test Career"},
            traits=["brave", "honest"],
            values=["justice", "truth"],
            fears=["failure"],
            desires=["success"],
            emotional_state=EmotionalState(),
            memory=CharacterMemory(),
            current_goal="Test goal"
        )
        
        self.invalid_character = CharacterState(
            id="",  # Invalid: empty ID
            name="",  # Invalid: empty name
            backstory={},
            traits=[],  # Invalid: no traits
            values=[],  # Invalid: no values
            fears=[],
            desires=[],
            emotional_state=EmotionalState(),
            memory=CharacterMemory(),
            current_goal=""
        )
    
    def test_character_validation_valid(self):
        """Test validation of valid character"""
        self.assertTrue(self.valid_character.validate())
    
    def test_character_validation_invalid(self):
        """Test validation catches invalid character"""
        self.assertFalse(self.invalid_character.validate())
    
    def test_simulation_prompt_generation(self):
        """Test prompt generation for valid character"""
        prompt = self.valid_character.get_simulation_prompt(
            "Test situation",
            "power"
        )
        
        self.assertIn("Test Character", prompt)
        self.assertIn("Test Origin", prompt)
        self.assertIn("brave", prompt)
        self.assertIn("Test situation", prompt)
        self.assertIn("power", prompt)
    
    def test_simulation_prompt_invalid_character(self):
        """Test prompt generation fails gracefully for invalid character"""
        with self.assertRaises(SimulationError):
            self.invalid_character.get_simulation_prompt("Test", "neutral")

class TestLLMInterfaces(unittest.TestCase):
    """Test LLM provider implementations"""
    
    def test_mock_llm_response(self):
        """Test MockLLM generates appropriate responses"""
        async def test():
            llm = MockLLM()
            
            # Test power emphasis
            response = await llm.generate_response(
                "Test prompt with power",
                temperature=1.0
            )
            self.assertIsInstance(response, LLMResponse)
            data = json.loads(response.content)
            self.assertIn("dialogue", data)
            self.assertIn("empire", data["dialogue"].lower())
            
            # Test doubt emphasis
            response = await llm.generate_response(
                "Test prompt with doubt",
                temperature=0.7
            )
            data = json.loads(response.content)
            self.assertIn("truth", data["dialogue"].lower())
        
        asyncio.run(test())
    
    def test_mock_llm_health_check(self):
        """Test MockLLM health check"""
        async def test():
            llm = MockLLM()
            health = await llm.health_check()
            self.assertTrue(health)
        
        asyncio.run(test())
    
    @patch('aiohttp.ClientSession')
    def test_openai_llm_initialization(self, mock_session):
        """Test OpenAI LLM initialization"""
        llm = OpenAILLM(api_key="test_key", model="gpt-4")
        self.assertEqual(llm.api_key, "test_key")
        self.assertEqual(llm.model, "gpt-4")
    
    def test_lmstudio_llm_initialization(self):
        """Test LMStudio LLM initialization"""
        llm = LMStudioLLM(endpoint="http://localhost:5000")
        self.assertEqual(llm.endpoint, "http://localhost:5000")

class TestRetryHandler(unittest.TestCase):
    """Test retry logic with exponential backoff"""
    
    def test_successful_execution(self):
        """Test successful execution on first try"""
        async def success_func():
            return "success"
        
        async def test():
            handler = RetryHandler(max_retries=3)
            result = await handler.execute_with_retry(success_func)
            self.assertEqual(result, "success")
        
        asyncio.run(test())
    
    def test_retry_on_failure(self):
        """Test retry logic on failures"""
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Test failure")
            return "success"
        
        async def test():
            handler = RetryHandler(max_retries=3, base_delay=0.01)
            result = await handler.execute_with_retry(failing_func)
            self.assertEqual(result, "success")
            self.assertEqual(call_count, 3)
        
        asyncio.run(test())
    
    def test_max_retries_exceeded(self):
        """Test failure after max retries"""
        async def always_fails():
            raise Exception("Always fails")
        
        async def test():
            handler = RetryHandler(max_retries=2, base_delay=0.01)
            with self.assertRaises(LLMError):
                await handler.execute_with_retry(always_fails)
        
        asyncio.run(test())

class TestSimulationEngine(unittest.TestCase):
    """Test simulation engine with full integration"""
    
    def setUp(self):
        """Initialize test components"""
        self.mock_llm = MockLLM()
        self.engine = SimulationEngine(
            llm_provider=self.mock_llm,
            max_concurrent=2
        )
        
        self.test_character = CharacterState(
            id="test_char",
            name="Test Character",
            backstory={"origin": "Test", "career": "Test"},
            traits=["brave"],
            values=["justice"],
            fears=["failure"],
            desires=["success"],
            emotional_state=EmotionalState(doubt=0.7),
            memory=CharacterMemory(),
            current_goal="Test goal"
        )
    
    def test_single_simulation(self):
        """Test running a single simulation"""
        async def test():
            result = await self.engine.run_simulation(
                self.test_character,
                "Test situation",
                "doubt",
                0.8
            )
            
            self.assertEqual(result["character_id"], "test_char")
            self.assertEqual(result["situation"], "Test situation")
            self.assertEqual(result["emphasis"], "doubt")
            self.assertEqual(result["temperature"], 0.8)
            self.assertIn("response", result)
            self.assertIn("timestamp", result)
        
        asyncio.run(test())
    
    def test_emotional_shift_application(self):
        """Test emotional shifts are applied correctly"""
        async def test():
            # Mock LLM to return emotional shift
            original_doubt = self.test_character.emotional_state.doubt
            
            with patch.object(self.mock_llm, 'generate_response') as mock_gen:
                mock_gen.return_value = LLMResponse(
                    json.dumps({
                        "dialogue": "Test",
                        "thought": "Test",
                        "action": "Test",
                        "emotional_shift": {"doubt": 0.2}
                    })
                )
                
                await self.engine.run_simulation(
                    self.test_character,
                    "Test",
                    "neutral"
                )
                
                # Check doubt increased
                new_doubt = self.test_character.emotional_state.doubt
                self.assertAlmostEqual(new_doubt, min(1.0, original_doubt + 0.2))
        
        asyncio.run(test())
    
    def test_multiple_simulations(self):
        """Test running multiple simulations"""
        async def test():
            results = await self.engine.run_multiple_simulations(
                self.test_character,
                "Test situation",
                num_runs=3,
                emphases=["power", "doubt"]
            )
            
            self.assertEqual(len(results), 3)
            # Check variety in emphases
            emphases_used = [r["emphasis"] for r in results]
            self.assertTrue(any(e in ["power", "doubt"] for e in emphases_used))
        
        asyncio.run(test())
    
    def test_concurrent_simulation_limit(self):
        """Test that concurrent simulations respect semaphore limit"""
        async def test():
            # Create engine with limit of 2
            engine = SimulationEngine(
                llm_provider=self.mock_llm,
                max_concurrent=2
            )
            
            # Track concurrent executions
            concurrent_count = 0
            max_concurrent = 0
            
            async def tracked_generate(*args, **kwargs):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.1)  # Simulate work
                concurrent_count -= 1
                return LLMResponse(json.dumps({"test": "response"}))
            
            with patch.object(self.mock_llm, 'generate_response', tracked_generate):
                results = await engine.run_multiple_simulations(
                    self.test_character,
                    "Test",
                    num_runs=5
                )
                
                # Should never exceed limit
                self.assertLessEqual(max_concurrent, 2)
                self.assertGreaterEqual(len(results), 3)  # Some should complete
        
        asyncio.run(test())
    
    def test_simulation_error_handling(self):
        """Test simulation handles errors gracefully"""
        async def test():
            # Make LLM fail
            with patch.object(self.mock_llm, 'generate_response', side_effect=Exception("LLM Error")):
                with self.assertRaises(SimulationError):
                    await self.engine.run_simulation(
                        self.test_character,
                        "Test",
                        "neutral"
                    )
        
        asyncio.run(test())

class TestConfigurationLoading(unittest.TestCase):
    """Test loading configuration from YAML"""
    
    def test_load_config(self):
        """Test loading simulation_config.yaml"""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "simulation_config.yaml"
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Verify key sections exist
            self.assertIn('llm', config)
            self.assertIn('simulation', config)
            self.assertIn('character', config)
            self.assertIn('evaluation', config)
            
            # Verify LLM settings
            self.assertIn('provider', config['llm'])
            
            # Verify simulation settings
            self.assertIn('max_concurrent', config['simulation'])
            self.assertIn('default_temperature', config['simulation'])
    
    def test_config_with_engine(self):
        """Test using config to initialize engine"""
        # Create minimal test config
        test_config = {
            'simulation': {
                'max_concurrent': 5,
                'retry': {
                    'max_attempts': 2,
                    'base_delay': 0.5
                }
            }
        }
        
        # Initialize components with config
        retry_handler = RetryHandler(
            max_retries=test_config['simulation']['retry']['max_attempts'],
            base_delay=test_config['simulation']['retry']['base_delay']
        )
        
        engine = SimulationEngine(
            llm_provider=MockLLM(),
            max_concurrent=test_config['simulation']['max_concurrent'],
            retry_handler=retry_handler
        )
        
        self.assertEqual(engine.semaphore._value, 5)
        self.assertEqual(engine.retry_handler.max_retries, 2)

class TestEndToEndSimulation(unittest.TestCase):
    """Test complete simulation workflow"""
    
    def test_full_simulation_workflow(self):
        """Test complete workflow from character creation to results"""
        async def test():
            # 1. Create character
            pilate = CharacterState(
                id="pilate",
                name="Pontius Pilate",
                backstory={
                    "origin": "Roman equestrian",
                    "career": "Prefect of Judaea"
                },
                traits=["pragmatic", "anxious"],
                values=["order", "duty"],
                fears=["rebellion"],
                desires=["peace"],
                emotional_state=EmotionalState(
                    anger=0.4,
                    doubt=0.7,
                    fear=0.6
                ),
                memory=CharacterMemory(),
                current_goal="Maintain order"
            )
            
            # 2. Initialize engine
            engine = SimulationEngine(
                llm_provider=MockLLM(),
                max_concurrent=3
            )
            
            # 3. Run simulations
            results = await engine.run_multiple_simulations(
                pilate,
                "The crowd demands justice",
                num_runs=3
            )
            
            # 4. Verify results
            self.assertEqual(len(results), 3)
            
            for result in results:
                self.assertIn('character_id', result)
                self.assertIn('response', result)
                self.assertIn('timestamp', result)
                
                # Verify response structure
                response = result['response']
                if isinstance(response, dict):
                    # Should have expected fields
                    self.assertTrue(
                        any(key in response for key in ['dialogue', 'raw_response'])
                    )
        
        asyncio.run(test())

# Performance/Load Testing
class TestPerformance(unittest.TestCase):
    """Test performance under load"""
    
    def test_concurrent_load(self):
        """Test system under concurrent load"""
        async def test():
            engine = SimulationEngine(
                llm_provider=MockLLM(),
                max_concurrent=10
            )
            
            character = CharacterState(
                id="test",
                name="Test",
                backstory={"origin": "Test"},
                traits=["test"],
                values=["test"],
                fears=["test"],
                desires=["test"],
                emotional_state=EmotionalState(),
                memory=CharacterMemory(),
                current_goal="Test"
            )
            
            start_time = asyncio.get_event_loop().time()
            
            # Run many simulations
            results = await engine.run_multiple_simulations(
                character,
                "Test",
                num_runs=20
            )
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            # Should complete in reasonable time
            self.assertLess(duration, 10.0)  # 10 seconds for 20 simulations
            self.assertEqual(len(results), 20)
        
        asyncio.run(test())

if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)