"""
Recursive Simulation Engine
Manages complex simulation workflows where persona agents spawn sub-simulations
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from enum import Enum

from .autonomous_persona_agents import PersonaAgentFactory
from .unified_llm_orchestrator import LLMPersona
from .llm_orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)


class SimulationStatus(Enum):
    """Status of a simulation"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationPriority(Enum):
    """Priority levels for simulation scheduling"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SimulationContext:
    """Lightweight context container used by tests and agents.

    This mirrors the minimal fields the tests expect to pass to agents when
    generating templates or running simulations.
    """

    agent_id: str
    simulation_id: str
    context_data: Dict[str, Any]
    parent_simulation_id: Optional[str]
    depth: int


@dataclass
class SimulationRequest:
    """Request for a simulation to be executed"""

    request_id: str
    persona: LLMPersona
    context_data: Dict[str, Any]
    priority: SimulationPriority
    parent_simulation_id: Optional[str]
    depth: int
    max_depth: int
    timeout_seconds: int
    callbacks: List[str]  # Callback function names
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["persona"] = self.persona.value
        result["priority"] = self.priority.value
        return result


@dataclass
class SimulationResult:
    """Result of a completed simulation"""

    request_id: str
    simulation_id: str
    persona: str
    status: SimulationStatus
    result_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    execution_time: float
    template_used: str
    recursive_results: List["SimulationResult"]
    depth: int
    created_at: datetime
    completed_at: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["created_at"] = self.created_at.isoformat()
        result["completed_at"] = (
            self.completed_at.isoformat() if self.completed_at else None
        )
        result["recursive_results"] = [r.to_dict() for r in self.recursive_results]
        return result


class SimulationScheduler:
    """Manages scheduling and execution of simulation requests"""

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.pending_queue: List[SimulationRequest] = []
        self.running_simulations: Dict[str, SimulationRequest] = {}
        self.completed_results: Dict[str, SimulationResult] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Statistics
        self.stats = {
            "total_requests": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "average_execution_time": 0.0,
            "persona_usage": {},
            "depth_distribution": {},
        }

        logger.info(
            f"SimulationScheduler initialized with max_concurrent={max_concurrent}"
        )

    def add_request(self, request: SimulationRequest):
        """Add a simulation request to the queue"""
        self.pending_queue.append(request)
        self.stats["total_requests"] += 1

        # Sort by priority and creation time
        self.pending_queue.sort(
            key=lambda r: (r.priority.value, r.metadata.get("created_at", 0)),
            reverse=True,
        )

        logger.info(
            f"Added simulation request {request.request_id} with priority {request.priority.value}"
        )

    async def get_next_request(self) -> Optional[SimulationRequest]:
        """Get the next request to execute"""
        if not self.pending_queue:
            return None

        # Check if we can run more simulations
        if len(self.running_simulations) >= self.max_concurrent:
            return None

        return self.pending_queue.pop(0)

    def mark_running(self, request: SimulationRequest):
        """Mark request as running"""
        self.running_simulations[request.request_id] = request

        # Update stats
        self.stats["persona_usage"][request.persona.value] = (
            self.stats["persona_usage"].get(request.persona.value, 0) + 1
        )

        self.stats["depth_distribution"][request.depth] = (
            self.stats["depth_distribution"].get(request.depth, 0) + 1
        )

    def mark_completed(self, request_id: str, result: SimulationResult):
        """Mark request as completed"""
        if request_id in self.running_simulations:
            del self.running_simulations[request_id]

        self.completed_results[request_id] = result

        # Update stats
        if result.status == SimulationStatus.COMPLETED:
            self.stats["completed"] += 1
        elif result.status == SimulationStatus.FAILED:
            self.stats["failed"] += 1
        elif result.status == SimulationStatus.CANCELLED:
            self.stats["cancelled"] += 1

        # Update average execution time
        total_completed = self.stats["completed"] + self.stats["failed"]
        if total_completed > 0:
            current_avg = self.stats["average_execution_time"]
            self.stats["average_execution_time"] = (
                current_avg * (total_completed - 1) + result.execution_time
            ) / total_completed

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "pending_requests": len(self.pending_queue),
            "running_simulations": len(self.running_simulations),
            "completed_results": len(self.completed_results),
            "max_concurrent": self.max_concurrent,
            "statistics": dict(self.stats),
        }


class RecursiveSimulationEngine:
    """Main engine for managing recursive simulations with autonomous persona agents"""

    def __init__(
        self,
        orchestrator: LLMOrchestrator,
        repository_path: Optional[Path] = None,
        max_concurrent: int = 5,
        max_depth: int = 5,
        default_timeout: int = 300,
    ):
        self.orchestrator = orchestrator
        # Allow tests to construct engine without passing a repository path
        self.repository_path = repository_path or Path("templates")
        self.max_depth = max_depth
        self.default_timeout = default_timeout

        # Create autonomous persona agents
        self.agents = PersonaAgentFactory.create_all_agents(
            orchestrator, self.repository_path
        )

        # Simulation management
        self.scheduler = SimulationScheduler(max_concurrent)
        self.callback_registry: Dict[str, Callable] = {}

        # Engine state
        self.running = False
        self.simulation_tree: Dict[str, List[str]] = {}  # parent_id -> [child_ids]

        logger.info(
            f"RecursiveSimulationEngine initialized with {len(self.agents)} agents"
        )

    def register_callback(
        self, name: str, callback: Callable[[SimulationResult], None]
    ):
        """Register a callback for simulation events"""
        self.callback_registry[name] = callback
        logger.info(f"Registered callback: {name}")

    async def start(self):
        """Start the simulation engine"""
        if self.running:
            return

        self.running = True
        logger.info("RecursiveSimulationEngine started")

        # Start the main execution loop
        asyncio.create_task(self._execution_loop())

    async def stop(self):
        """Stop the simulation engine"""
        self.running = False
        logger.info("RecursiveSimulationEngine stopped")

    async def submit_simulation(
        self,
        persona: LLMPersona,
        context_data: Dict[str, Any],
        priority: SimulationPriority = SimulationPriority.NORMAL,
        parent_simulation_id: Optional[str] = None,
        depth: int = 0,
        max_depth: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        callbacks: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Submit a simulation request"""

        request_id = str(uuid.uuid4())

        request = SimulationRequest(
            request_id=request_id,
            persona=persona,
            context_data=context_data,
            priority=priority,
            parent_simulation_id=parent_simulation_id,
            depth=depth,
            max_depth=max_depth or self.max_depth,
            timeout_seconds=timeout_seconds or self.default_timeout,
            callbacks=callbacks or [],
            metadata=metadata or {"created_at": datetime.now().timestamp()},
        )

        self.scheduler.add_request(request)

        # Update simulation tree
        if parent_simulation_id:
            if parent_simulation_id not in self.simulation_tree:
                self.simulation_tree[parent_simulation_id] = []
            self.simulation_tree[parent_simulation_id].append(request_id)

        logger.info(
            f"Submitted simulation {request_id} for persona {persona.value} at depth {depth}"
        )

        return request_id

    async def get_simulation_result(
        self, request_id: str
    ) -> Optional[SimulationResult]:
        """Get the result of a simulation"""
        return self.scheduler.completed_results.get(request_id)

    async def get_simulation_tree(self, root_simulation_id: str) -> Dict[str, Any]:
        """Get the complete simulation tree starting from a root simulation"""

        async def build_tree(simulation_id: str) -> Dict[str, Any]:
            result = await self.get_simulation_result(simulation_id)

            tree_node = {
                "simulation_id": simulation_id,
                "result": result.to_dict() if result else None,
                "children": [],
            }

            # Add child simulations
            child_ids = self.simulation_tree.get(simulation_id, [])
            for child_id in child_ids:
                child_tree = await build_tree(child_id)
                tree_node["children"].append(child_tree)

            return tree_node

        return await build_tree(root_simulation_id)

    async def _execution_loop(self):
        """Main execution loop for processing simulation requests"""

        while self.running:
            try:
                # Get next request to execute
                request = await self.scheduler.get_next_request()

                if request:
                    # Execute simulation asynchronously
                    asyncio.create_task(self._execute_simulation(request))
                else:
                    # No pending requests, wait a bit
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(1)

    async def _execute_simulation(self, request: SimulationRequest):
        """Execute a single simulation request"""

        async with self.scheduler.semaphore:
            start_time = datetime.now()
            self.scheduler.mark_running(request)

            try:
                # Get the appropriate agent
                agent = self.agents.get(request.persona)
                if not agent:
                    raise ValueError(
                        f"No agent available for persona {request.persona}"
                    )

                logger.info(
                    f"Executing simulation {request.request_id} with {request.persona.value}"
                )

                # Execute with timeout
                try:
                    simulation_result = await asyncio.wait_for(
                        agent.run_simulation(
                            request.context_data,
                            request.parent_simulation_id,
                            request.depth,
                        ),
                        timeout=request.timeout_seconds,
                    )

                    # Process recursive results
                    recursive_results = await self._process_recursive_results(
                        request, simulation_result
                    )

                    # Create result object
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()

                    result = SimulationResult(
                        request_id=request.request_id,
                        simulation_id=simulation_result.get("simulation_id", ""),
                        persona=request.persona.value,
                        status=SimulationStatus.COMPLETED,
                        result_data=simulation_result,
                        error_message=None,
                        execution_time=execution_time,
                        template_used=simulation_result.get("template_used", ""),
                        recursive_results=recursive_results,
                        depth=request.depth,
                        created_at=start_time,
                        completed_at=end_time,
                    )

                except asyncio.TimeoutError:
                    # Simulation timed out
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()

                    result = SimulationResult(
                        request_id=request.request_id,
                        simulation_id="",
                        persona=request.persona.value,
                        status=SimulationStatus.FAILED,
                        result_data=None,
                        error_message=f"Simulation timed out after {request.timeout_seconds} seconds",
                        execution_time=execution_time,
                        template_used="",
                        recursive_results=[],
                        depth=request.depth,
                        created_at=start_time,
                        completed_at=end_time,
                    )

            except Exception as e:
                # Simulation failed
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

                result = SimulationResult(
                    request_id=request.request_id,
                    simulation_id="",
                    persona=request.persona.value,
                    status=SimulationStatus.FAILED,
                    result_data=None,
                    error_message=str(e),
                    execution_time=execution_time,
                    template_used="",
                    recursive_results=[],
                    depth=request.depth,
                    created_at=start_time,
                    completed_at=end_time,
                )

                logger.error(f"Simulation {request.request_id} failed: {e}")

            # Mark as completed and trigger callbacks
            self.scheduler.mark_completed(request.request_id, result)
            await self._trigger_callbacks(request, result)

    async def _process_recursive_results(
        self, request: SimulationRequest, simulation_result: Dict[str, Any]
    ) -> List[SimulationResult]:
        """Process any recursive simulations triggered by this result"""

        recursive_results = []

        # Check if simulation triggered recursive simulations
        recursive_data = simulation_result.get("recursive_results", [])

        for recursive_sim in recursive_data:
            # Submit recursive simulation request
            recursive_request_id = await self.submit_simulation(
                persona=request.persona,  # Could be different based on result
                context_data=recursive_sim.get("context", {}),
                priority=SimulationPriority.HIGH,  # Recursive sims have high priority
                parent_simulation_id=request.request_id,
                depth=request.depth + 1,
                max_depth=request.max_depth,
                timeout_seconds=request.timeout_seconds
                // 2,  # Shorter timeout for recursive
                callbacks=[],
                metadata={"recursive": True, "parent": request.request_id},
            )

            # Wait for recursive simulation to complete
            # In practice, you might want to handle this asynchronously
            recursive_result = None
            max_wait = 30  # Wait up to 30 seconds for recursive result
            wait_time = 0

            while wait_time < max_wait:
                recursive_result = await self.get_simulation_result(
                    recursive_request_id
                )
                if recursive_result:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            if recursive_result:
                recursive_results.append(recursive_result)

        return recursive_results

    async def _trigger_callbacks(
        self, request: SimulationRequest, result: SimulationResult
    ):
        """Trigger registered callbacks for simulation completion"""

        for callback_name in request.callbacks:
            callback = self.callback_registry.get(callback_name)
            if callback:
                try:
                    (
                        await callback(result)
                        if asyncio.iscoroutinefunction(callback)
                        else callback(result)
                    )
                except Exception as e:
                    logger.error(f"Callback {callback_name} failed: {e}")

    # Monitoring and management methods

    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            "engine_running": self.running,
            "available_agents": list(self.agents.keys()),
            "scheduler_status": self.scheduler.get_status(),
            "simulation_tree_size": len(self.simulation_tree),
            "registered_callbacks": list(self.callback_registry.keys()),
            "max_depth": self.max_depth,
            "default_timeout": self.default_timeout,
        }

    # Back-compat for tests expecting `get_status()`
    def get_status(self) -> Dict[str, Any]:
        """Return a succinct engine status used by tests.

        Keys expected by tests:
        - active_simulations: number of currently running simulations
        - deadlock_detected: whether a deadlock was detected (placeholder False)
        """
        scheduler_status = self.scheduler.get_status()
        return {
            "active_simulations": scheduler_status.get("running_simulations", 0),
            "pending_requests": scheduler_status.get("pending_requests", 0),
            "completed_results": scheduler_status.get("completed_results", 0),
            "deadlock_detected": False,
        }

    async def cancel_simulation(self, request_id: str) -> bool:
        """Cancel a pending simulation"""
        # Find and remove from pending queue
        for i, request in enumerate(self.scheduler.pending_queue):
            if request.request_id == request_id:
                cancelled_request = self.scheduler.pending_queue.pop(i)

                # Create cancelled result
                result = SimulationResult(
                    request_id=request_id,
                    simulation_id="",
                    persona=cancelled_request.persona.value,
                    status=SimulationStatus.CANCELLED,
                    result_data=None,
                    error_message="Cancelled by user",
                    execution_time=0.0,
                    template_used="",
                    recursive_results=[],
                    depth=cancelled_request.depth,
                    created_at=datetime.now(),
                    completed_at=datetime.now(),
                )

                self.scheduler.mark_completed(request_id, result)

                logger.info(f"Cancelled simulation {request_id}")
                return True

        return False

    async def clear_completed_results(self, older_than_hours: int = 24):
        """Clear old completed results to free memory"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        to_remove = []
        for request_id, result in self.scheduler.completed_results.items():
            if result.completed_at and result.completed_at < cutoff_time:
                to_remove.append(request_id)

        for request_id in to_remove:
            del self.scheduler.completed_results[request_id]

        logger.info(f"Cleared {len(to_remove)} old simulation results")

        return len(to_remove)


# Factory function
def create_recursive_simulation_engine(
    orchestrator: LLMOrchestrator,
    repository_path: str | Path,
    max_concurrent: int = 5,
    max_depth: int = 5,
    default_timeout: int = 300,
) -> RecursiveSimulationEngine:
    """Create a RecursiveSimulationEngine with the specified configuration"""

    if isinstance(repository_path, str):
        repository_path = Path(repository_path)

    engine = RecursiveSimulationEngine(
        orchestrator=orchestrator,
        repository_path=repository_path,
        max_concurrent=max_concurrent,
        max_depth=max_depth,
        default_timeout=default_timeout,
    )

    logger.info("RecursiveSimulationEngine created and ready")

    return engine
