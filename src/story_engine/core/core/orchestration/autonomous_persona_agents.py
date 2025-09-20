"""
Autonomous Persona Agents with Self-Modifying POML Templates
Each persona is an autonomous agent that can create, modify, and optimize its own templates
"""

import logging
from llm_observability import get_logger
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod
import uuid

from .unified_llm_orchestrator import LLMPersona
from .llm_orchestrator import LLMResponse, LLMOrchestrator
from story_engine.poml.lib.poml_integration import POMLEngine

logger = logging.getLogger(__name__)


@dataclass
class SimulationContext:
    """Context for a specific simulation run"""

    simulation_id: str
    parent_simulation_id: Optional[str]
    depth: int
    context_data: Dict[str, Any]
    requirements: List[str]
    constraints: Dict[str, Any]
    success_criteria: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemplateMetadata:
    """Metadata about a generated template"""

    template_id: str
    persona: str
    simulation_context_hash: str
    created_at: datetime
    last_used: datetime
    usage_count: int
    success_rate: float
    performance_score: float
    parent_template_id: Optional[str]
    modifications: List[str]


class TemplateRepository:
    """Repository for storing and managing persona-specific templates"""

    def __init__(self, base_path: Path, persona_name: str):
        self.base_path = base_path
        self.persona_name = persona_name
        self.persona_path = base_path / "personas" / persona_name
        self.templates_path = self.persona_path / "templates"
        self.metadata_path = self.persona_path / "metadata"

        # Create directories
        self.templates_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        # Template cache
        self.template_cache = {}
        self.metadata_cache = {}

        logger.info(f"TemplateRepository initialized for {persona_name}")

    def generate_template_id(self, context: SimulationContext) -> str:
        """Generate unique template ID based on context"""
        context_str = json.dumps(context.to_dict(), sort_keys=True)
        return hashlib.sha256(
            f"{self.persona_name}:{context_str}".encode()
        ).hexdigest()[:16]

    async def store_template(
        self, template_id: str, template_content: str, metadata: TemplateMetadata
    ):
        """Store a template and its metadata"""
        template_file = self.templates_path / f"{template_id}.poml"
        metadata_file = self.metadata_path / f"{template_id}.json"

        # Write template
        with open(template_file, "w", encoding="utf-8") as f:
            f.write(template_content)

        # Write metadata
        metadata_dict = asdict(metadata)
        metadata_dict["created_at"] = metadata.created_at.isoformat()
        metadata_dict["last_used"] = metadata.last_used.isoformat()

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

        # Update cache
        self.template_cache[template_id] = template_content
        self.metadata_cache[template_id] = metadata

        logger.info(f"Stored template {template_id} for {self.persona_name}")

    async def load_template(self, template_id: str) -> Optional[str]:
        """Load a template by ID"""
        if template_id in self.template_cache:
            return self.template_cache[template_id]

        template_file = self.templates_path / f"{template_id}.poml"
        if template_file.exists():
            with open(template_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.template_cache[template_id] = content
                return content

        return None

    async def load_metadata(self, template_id: str) -> Optional[TemplateMetadata]:
        """Load template metadata"""
        if template_id in self.metadata_cache:
            return self.metadata_cache[template_id]

        metadata_file = self.metadata_path / f"{template_id}.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert datetime strings back
                data["created_at"] = datetime.fromisoformat(data["created_at"])
                data["last_used"] = datetime.fromisoformat(data["last_used"])
                metadata = TemplateMetadata(**data)
                self.metadata_cache[template_id] = metadata
                return metadata

        return None

    async def find_similar_templates(
        self, context: SimulationContext, similarity_threshold: float = 0.7
    ) -> List[str]:
        """Find templates similar to the given context"""
        similar_templates = []

        # Simple similarity based on context requirements overlap
        for template_file in self.templates_path.glob("*.poml"):
            template_id = template_file.stem
            metadata = await self.load_metadata(template_id)

            if metadata:
                # Calculate similarity (simplified)
                similarity = self._calculate_similarity(context, metadata)
                if similarity >= similarity_threshold:
                    similar_templates.append(template_id)

        # Sort by usage count and performance
        similar_templates.sort(
            key=lambda tid: (
                self.metadata_cache.get(
                    tid,
                    TemplateMetadata(
                        "",
                        "",
                        "",
                        datetime.now(),
                        datetime.now(),
                        0,
                        0.0,
                        0.0,
                        None,
                        [],
                    ),
                ).usage_count,
                self.metadata_cache.get(
                    tid,
                    TemplateMetadata(
                        "",
                        "",
                        "",
                        datetime.now(),
                        datetime.now(),
                        0,
                        0.0,
                        0.0,
                        None,
                        [],
                    ),
                ).performance_score,
            ),
            reverse=True,
        )

        return similar_templates

    def _calculate_similarity(
        self, context: SimulationContext, metadata: TemplateMetadata
    ) -> float:
        """Calculate similarity between context and existing template metadata"""
        # Simplified similarity calculation
        # In practice, this would use more sophisticated matching

        if not hasattr(context, "requirements") or not context.requirements:
            return 0.0

        # Compare based on context hash similarity (simplified)
        if metadata.simulation_context_hash == self.generate_template_id(context):
            return 1.0

        return 0.3  # Default low similarity

    async def update_template_performance(
        self, template_id: str, success: bool, performance_score: float
    ):
        """Update template performance metrics"""
        metadata = await self.load_metadata(template_id)
        if metadata:
            metadata.usage_count += 1
            metadata.last_used = datetime.now()

            # Update success rate
            old_successes = metadata.success_rate * (metadata.usage_count - 1)
            new_successes = old_successes + (1 if success else 0)
            metadata.success_rate = new_successes / metadata.usage_count

            # Update performance score (moving average)
            alpha = 0.3  # Learning rate
            metadata.performance_score = (
                alpha * performance_score + (1 - alpha) * metadata.performance_score
            )

            # Save updated metadata
            await self.store_template(
                template_id, await self.load_template(template_id), metadata
            )


class AutonomousPersonaAgent(ABC):
    """Base class for autonomous persona agents"""

    def __init__(
        self, persona: LLMPersona, orchestrator: LLMOrchestrator, repository_path: Path
    ):
        self.persona = persona
        self.orchestrator = orchestrator
        self.repository = TemplateRepository(repository_path, persona.value)
        self.poml_engine = POMLEngine()

        # Agent state
        self.active_simulations: Dict[str, SimulationContext] = {}
        self.template_generation_history = []
        self.learning_enabled = True

        logger.info(f"AutonomousPersonaAgent initialized for {persona.value}")

    @abstractmethod
    async def generate_base_template(self, context: SimulationContext) -> str:
        """Generate a base POML template for the given context"""
        pass

    @abstractmethod
    async def modify_template(
        self,
        existing_template: str,
        context: SimulationContext,
        modification_reason: str,
    ) -> str:
        """Modify an existing template based on context and reason"""
        pass

    @abstractmethod
    def extract_simulation_requirements(
        self, context_data: Dict[str, Any]
    ) -> List[str]:
        """Extract specific requirements for this persona from context data"""
        pass

    async def run_simulation(
        self,
        context_data: Dict[str, Any],
        parent_simulation_id: Optional[str] = None,
        depth: int = 0,
    ) -> Dict[str, Any]:
        """Run a simulation with dynamic template creation/modification"""

        # Create simulation context
        simulation_id = str(uuid.uuid4())
        requirements = self.extract_simulation_requirements(context_data)

        context = SimulationContext(
            simulation_id=simulation_id,
            parent_simulation_id=parent_simulation_id,
            depth=depth,
            context_data=context_data,
            requirements=requirements,
            constraints=context_data.get("constraints", {}),
            success_criteria=context_data.get("success_criteria", []),
        )

        self.active_simulations[simulation_id] = context

        try:
            # 1. Find or create appropriate template
            template_content = await self._get_or_create_template(context)

            # 2. Execute simulation using the template
            result = await self._execute_simulation(context, template_content)

            # 3. Evaluate result and update template performance
            await self._evaluate_and_learn(context, template_content, result)

            # 4. Handle recursive simulations if needed
            recursive_results = await self._handle_recursive_simulations(
                context, result
            )

            # Combine results
            final_result = {
                "simulation_id": simulation_id,
                "persona": self.persona.value,
                "depth": depth,
                "primary_result": result,
                "recursive_results": recursive_results,
                "template_used": template_content[:200]
                + "...",  # Truncated for logging
                "context": context.to_dict(),
            }

            return final_result

        finally:
            # Clean up
            if simulation_id in self.active_simulations:
                del self.active_simulations[simulation_id]

    async def _get_or_create_template(self, context: SimulationContext) -> str:
        """Get existing similar template or create new one"""

        # 1. Look for existing similar templates
        similar_templates = await self.repository.find_similar_templates(context)

        if similar_templates:
            # Use the best existing template
            best_template_id = similar_templates[0]
            template_content = await self.repository.load_template(best_template_id)

            if template_content:
                logger.info(
                    f"Using existing template {best_template_id} for {self.persona.value}"
                )

                # Check if modification is needed
                if await self._should_modify_template(context, template_content):
                    template_content = await self.modify_template(
                        template_content, context, "Context-specific optimization"
                    )

                    # Store as new template variant
                    new_template_id = self.repository.generate_template_id(context)
                    await self._store_new_template(
                        new_template_id, template_content, context, best_template_id
                    )

                return template_content

        # 2. No suitable template found, generate new one
        logger.info(f"Generating new template for {self.persona.value}")
        template_content = await self.generate_base_template(context)

        # Store the new template
        template_id = self.repository.generate_template_id(context)
        await self._store_new_template(template_id, template_content, context)

        return template_content

    async def _should_modify_template(
        self, context: SimulationContext, template_content: str
    ) -> bool:
        """Determine if template needs modification for this context"""

        # Check if context has specific requirements not covered by template
        template_lower = template_content.lower()

        for requirement in context.requirements:
            if requirement.lower() not in template_lower:
                return True

        # Check for constraints that might need template adjustment
        if context.constraints:
            constraint_keywords = [str(v).lower() for v in context.constraints.values()]
            if not any(keyword in template_lower for keyword in constraint_keywords):
                return True

        return False

    async def _store_new_template(
        self,
        template_id: str,
        template_content: str,
        context: SimulationContext,
        parent_template_id: Optional[str] = None,
    ):
        """Store a new template with metadata"""

        metadata = TemplateMetadata(
            template_id=template_id,
            persona=self.persona.value,
            simulation_context_hash=self.repository.generate_template_id(context),
            created_at=datetime.now(),
            last_used=datetime.now(),
            usage_count=0,
            success_rate=0.0,
            performance_score=0.0,
            parent_template_id=parent_template_id,
            modifications=[],
        )

        await self.repository.store_template(template_id, template_content, metadata)

    async def _execute_simulation(
        self, context: SimulationContext, template_content: str
    ) -> LLMResponse:
        """Execute the simulation using the generated template"""
        import tempfile

        # Use a temporary file to avoid race conditions
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".poml", delete=False
        ) as tmp_file:
            tmp_file.write(template_content)
            tmp_path = Path(tmp_file.name)

        try:
            # Render template with context data
            rendered_prompt = self.poml_engine.render(
                str(tmp_path), context.context_data
            )
        finally:
            # Clean up the temporary file
            if tmp_path.exists():
                tmp_path.unlink()

        # Execute with orchestrator
        response = await self.orchestrator.generate(
            prompt=rendered_prompt,
            temperature=context.context_data.get("temperature", 0.7),
            max_tokens=context.context_data.get("max_tokens", 500),
            allow_fallback=True,
        )

        return response

    async def _evaluate_and_learn(
        self, context: SimulationContext, template_content: str, result: LLMResponse
    ):
        """Evaluate result quality and update template performance"""

        if not self.learning_enabled:
            return

        # Simple evaluation metrics
        success = bool(
            hasattr(result, "text")
            and result.text
            or hasattr(result, "content")
            and result.content
        )

        # Calculate performance score based on various factors
        performance_score = 0.5  # Base score

        if success:
            performance_score += 0.3

            # Check if result meets success criteria
            result_text = getattr(result, "text", "") or getattr(result, "content", "")
            if result_text:
                for criterion in context.success_criteria:
                    if criterion.lower() in result_text.lower():
                        performance_score += 0.1

        # Update template performance
        template_id = self.repository.generate_template_id(context)
        await self.repository.update_template_performance(
            template_id, success, performance_score
        )

        logger.info(
            f"Template performance updated: {template_id}, success={success}, score={performance_score}"
        )

    async def _handle_recursive_simulations(
        self, context: SimulationContext, result: LLMResponse
    ) -> List[Dict[str, Any]]:
        """Handle any recursive simulations needed based on the result"""

        recursive_results = []

        # Check if result suggests need for recursive simulations
        result_text = getattr(result, "text", "") or getattr(result, "content", "")

        if not result_text or context.depth >= 3:  # Prevent infinite recursion
            return recursive_results

        # Simple recursive simulation detection
        recursive_keywords = ["analyze", "elaborate", "detail", "expand"]

        if any(keyword in result_text.lower() for keyword in recursive_keywords):
            # Create recursive simulation context
            recursive_context_data = context.context_data.copy()
            recursive_context_data["previous_result"] = result_text
            recursive_context_data["recursive_focus"] = "elaboration"

            logger.info(
                f"Triggering recursive simulation for {self.persona.value} at depth {context.depth + 1}"
            )

            try:
                recursive_result = await self.run_simulation(
                    recursive_context_data,
                    parent_simulation_id=context.simulation_id,
                    depth=context.depth + 1,
                )
                recursive_results.append(recursive_result)

            except Exception as e:
                    logger.error(f"Recursive simulation failed: {e}")

        return recursive_results

    # Template generation helpers

    async def _generate_meta_template_creation_prompt(
        self, context: SimulationContext
    ) -> str:
        """Generate a prompt for creating POML templates"""

        prompt = f"""You are the {self.persona.value} persona agent. Create a POML template for this simulation context:

Requirements: {', '.join(context.requirements)}
Constraints: {json.dumps(context.constraints, indent=2)}
Success Criteria: {', '.join(context.success_criteria)}

The template should:
1. Be a complete, valid POML document
2. Include appropriate persona-specific instructions
3. Handle the specific requirements and constraints
4. Generate structured output suitable for further processing
5. Include conditional logic for different scenarios

Generate only the POML template code, no explanations."""

        return prompt

    async def _generate_template_modification_prompt(
        self, existing_template: str, context: SimulationContext, reason: str
    ) -> str:
        """Generate a prompt for modifying existing templates"""

        prompt = f"""You are the {self.persona.value} persona agent. Modify this existing POML template:

EXISTING TEMPLATE:
{existing_template}

MODIFICATION CONTEXT:
Requirements: {', '.join(context.requirements)}
Reason: {reason}
New Constraints: {json.dumps(context.constraints, indent=2)}

Modify the template to:
1. Address the new requirements
2. Incorporate the specified constraints
3. Maintain template structure and validity
4. Preserve working elements from the original
5. Add appropriate conditional logic for new scenarios

Generate only the modified POML template code."""

        return prompt


# Concrete implementations for each persona


class CharacterSimulatorAgent(AutonomousPersonaAgent):
    """Character Simulator autonomous agent"""

    async def generate_base_template(self, context: SimulationContext) -> str:
        """Generate character simulation template"""

        meta_prompt = await self._generate_meta_template_creation_prompt(context)

        # Add character-specific guidance
        character_guidance = """
Focus on:
- Individual character psychology and emotional states
- Authentic dialogue and internal thoughts
- Emotional state changes and character development
- Consistent personality expression
- Realistic human responses to situations
"""

        full_prompt = meta_prompt + character_guidance

        response = await self.orchestrator.generate(
            prompt=full_prompt, temperature=0.6, max_tokens=1200, allow_fallback=True
        )

        return getattr(response, "text", "") or getattr(response, "content", "")

    async def modify_template(
        self,
        existing_template: str,
        context: SimulationContext,
        modification_reason: str,
    ) -> str:
        """Modify character simulation template"""

        modification_prompt = await self._generate_template_modification_prompt(
            existing_template, context, modification_reason
        )

        response = await self.orchestrator.generate(
            prompt=modification_prompt,
            temperature=0.5,
            max_tokens=1000,
            allow_fallback=True,
        )

        return getattr(response, "text", "") or getattr(response, "content", "")

    def extract_simulation_requirements(
        self, context_data: Dict[str, Any]
    ) -> List[str]:
        """Extract character-specific requirements"""
        requirements = []

        if "character" in context_data:
            requirements.append("character_consistency")
            requirements.append("emotional_authenticity")

        if "emphasis" in context_data:
            requirements.append(f'emphasis_{context_data["emphasis"]}')

        if "situation" in context_data:
            requirements.append("situational_response")

        return requirements


class SceneArchitectAgent(AutonomousPersonaAgent):
    """Scene Architect autonomous agent"""

    async def generate_base_template(self, context: SimulationContext) -> str:
        """Generate scene architecture template"""

        meta_prompt = await self._generate_meta_template_creation_prompt(context)

        scene_guidance = """
Focus on:
- Dramatic scene structure and pacing
- Environmental details and atmosphere
- Character positioning and staging
- Conflict setup and tension building
- Sensory details and immersion
"""

        full_prompt = meta_prompt + scene_guidance

        response = await self.orchestrator.generate(
            prompt=full_prompt, temperature=0.7, max_tokens=1200, allow_fallback=True
        )

        return getattr(response, "text", "") or getattr(response, "content", "")

    async def modify_template(
        self,
        existing_template: str,
        context: SimulationContext,
        modification_reason: str,
    ) -> str:
        """Modify scene architecture template"""

        modification_prompt = await self._generate_template_modification_prompt(
            existing_template, context, modification_reason
        )

        response = await self.orchestrator.generate(
            prompt=modification_prompt,
            temperature=0.6,
            max_tokens=1000,
            allow_fallback=True,
        )

        return getattr(response, "text", "") or getattr(response, "content", "")

    def extract_simulation_requirements(
        self, context_data: Dict[str, Any]
    ) -> List[str]:
        """Extract scene-specific requirements"""
        requirements = []

        if "beat" in context_data:
            requirements.append("dramatic_structure")
            requirements.append("tension_management")

        if "characters" in context_data:
            requirements.append("character_positioning")

        if "previous_context" in context_data:
            requirements.append("continuity")

        return requirements


# Factory for creating persona agents


class PersonaAgentFactory:
    """Factory for creating autonomous persona agents"""

    AGENT_CLASSES = {
        LLMPersona.CHARACTER_SIMULATOR: CharacterSimulatorAgent,
        LLMPersona.SCENE_ARCHITECT: SceneArchitectAgent,
        # Add more as needed
    }

    @classmethod
    def create_agent(
        cls, persona: LLMPersona, orchestrator: LLMOrchestrator, repository_path: Path
    ) -> AutonomousPersonaAgent:
        """Create an autonomous persona agent"""

        agent_class = cls.AGENT_CLASSES.get(persona)

        if not agent_class:
            raise ValueError(f"No agent implementation for persona {persona}")

        return agent_class(persona, orchestrator, repository_path)

    @classmethod
    def create_all_agents(
        cls, orchestrator: LLMOrchestrator, repository_path: Path
    ) -> Dict[LLMPersona, AutonomousPersonaAgent]:
        """Create all available persona agents"""

        agents = {}

        for persona in cls.AGENT_CLASSES:
            try:
                agents[persona] = cls.create_agent(
                    persona, orchestrator, repository_path
                )
                logger.info(f"Created agent for {persona.value}")
            except Exception as e:
                    logger.error(f"Failed to create agent for {persona.value}: {e}")

        return agents
