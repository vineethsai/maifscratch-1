"""
MAIF-Centric Agent Framework Demo
Demonstrates the complete agent framework with perception, reasoning, planning, execution, and learning.
"""

import asyncio
import json
import time
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import MAIF agent framework
from maif.agents.agentic_framework import (
    AutonomousMAIFAgent,
    MAIFAgentConsortium,
    MAIFAgent,
    AgentState,
    PerceptionSystem,
    ReasoningSystem,
    PlanningSystem,
    ExecutionSystem,
    LearningSystem,
    MemorySystem,
)

from maif_api import MAIF as MAIFArtifact


# Specialized Agent Types
class ResearchAgent(MAIFAgent):
    """Agent specialized in research and analysis."""

    async def run(self):
        """Research agent main loop."""
        logger.info(f"Research agent {self.agent_id} starting")

        research_topics = [
            "MAIF performance optimization",
            "Distributed agent coordination",
            "Semantic compression techniques",
            "Cross-modal attention mechanisms",
        ]

        for topic in research_topics:
            if self.state == AgentState.TERMINATED:
                break

            try:
                # Research phase
                logger.info(f"Researching: {topic}")

                # Perceive research data
                perception = await self.perceive(
                    f"Research data for {topic}: Recent advances show 30% improvement in efficiency",
                    "text",
                )

                # Analyze findings
                reasoning = await self.reason([perception])

                # Plan research actions
                plan = await self.plan(f"analyze {topic}", [reasoning])

                # Execute research
                results = await self.execute(plan)

                # Learn from research
                await self.learn([perception, reasoning, plan, results])

                # Store in knowledge base
                self.memory.store_episodic([perception, reasoning, plan, results])

                logger.info(f"Completed research on: {topic}")

                await asyncio.sleep(2.0)

            except Exception as e:
                logger.error(f"Research error: {e}")

        logger.info(f"Research agent {self.agent_id} completed")


class OptimizationAgent(MAIFAgent):
    """Agent specialized in system optimization."""

    async def run(self):
        """Optimization agent main loop."""
        logger.info(f"Optimization agent {self.agent_id} starting")

        optimization_targets = [
            {"target": "memory_usage", "threshold": 0.8},
            {"target": "processing_speed", "threshold": 0.7},
            {"target": "compression_ratio", "threshold": 0.6},
        ]

        for target in optimization_targets:
            if self.state == AgentState.TERMINATED:
                break

            try:
                # Monitor system
                perception = await self.perceive(
                    f"System metrics for {target['target']}: current value 0.65", "text"
                )

                # Analyze performance
                reasoning = await self.reason([perception])

                # Plan optimizations
                plan = await self.plan(f"optimize {target['target']}", [reasoning])

                # Execute optimizations
                results = await self.execute(plan)

                # Learn from results
                await self.learn([perception, reasoning, plan, results])

                logger.info(f"Optimized: {target['target']}")

                await asyncio.sleep(3.0)

            except Exception as e:
                logger.error(f"Optimization error: {e}")

        logger.info(f"Optimization agent {self.agent_id} completed")


class CoordinatorAgent(MAIFAgent):
    """Agent that coordinates other agents."""

    def __init__(
        self, agent_id: str, workspace_path: str, config: Optional[Dict] = None
    ):
        super().__init__(agent_id, workspace_path, config)
        self.subordinate_agents: List[str] = []

    def add_subordinate(self, agent_id: str):
        """Add subordinate agent."""
        self.subordinate_agents.append(agent_id)

    async def run(self):
        """Coordinator agent main loop."""
        logger.info(f"Coordinator agent {self.agent_id} starting")

        coordination_cycles = 5

        for cycle in range(coordination_cycles):
            if self.state == AgentState.TERMINATED:
                break

            try:
                # Monitor subordinates
                status_reports = []
                for sub_id in self.subordinate_agents:
                    perception = await self.perceive(
                        f"Status report from {sub_id}: Operating normally, 85% efficiency",
                        "text",
                    )
                    status_reports.append(perception)

                # Analyze overall status
                reasoning = await self.reason(status_reports)

                # Plan coordination actions
                plan = await self.plan("coordinate agent activities", [reasoning])

                # Execute coordination
                results = await self.execute(plan)

                # Learn from coordination
                await self.learn(status_reports + [reasoning, plan, results])

                logger.info(f"Coordination cycle {cycle + 1} completed")

                await asyncio.sleep(5.0)

            except Exception as e:
                logger.error(f"Coordination error: {e}")

        logger.info(f"Coordinator agent {self.agent_id} completed")


# Demo Functions
async def demonstrate_single_agent():
    """Demonstrate a single autonomous agent."""
    print("\n=== Single Agent Demo ===")

    workspace = Path("./demo_workspace/single_agent")
    workspace.mkdir(parents=True, exist_ok=True)

    # Create and run agent
    agent = AutonomousMAIFAgent("demo_agent", str(workspace))

    # Run for a limited time
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(15.0)

    # Shutdown
    agent.shutdown()
    task.cancel()

    print("Single agent demo completed")


async def demonstrate_specialized_agents():
    """Demonstrate specialized agent types."""
    print("\n=== Specialized Agents Demo ===")

    workspace = Path("./demo_workspace/specialized_agents")
    workspace.mkdir(parents=True, exist_ok=True)

    # Create specialized agents
    research_agent = ResearchAgent("research_01", str(workspace))
    optimization_agent = OptimizationAgent("optimizer_01", str(workspace))

    # Run agents
    tasks = [
        asyncio.create_task(research_agent.run()),
        asyncio.create_task(optimization_agent.run()),
    ]

    # Wait for completion
    await asyncio.gather(*tasks)

    # Shutdown
    research_agent.shutdown()
    optimization_agent.shutdown()

    print("Specialized agents demo completed")


async def demonstrate_agent_consortium():
    """Demonstrate multi-agent consortium."""
    print("\n=== Agent Consortium Demo ===")

    workspace = Path("./demo_workspace/consortium")
    workspace.mkdir(parents=True, exist_ok=True)

    # Create consortium
    consortium = MAIFAgentConsortium(str(workspace), enable_distribution=True)

    # Create coordinator
    coordinator = CoordinatorAgent("coordinator_01", str(workspace))

    # Create worker agents
    research_agent = ResearchAgent("research_01", str(workspace))
    optimization_agent = OptimizationAgent("optimizer_01", str(workspace))

    # Set up coordination
    coordinator.add_subordinate("research_01")
    coordinator.add_subordinate("optimizer_01")

    # Add to consortium
    consortium.add_agent(coordinator)
    consortium.add_agent(research_agent)
    consortium.add_agent(optimization_agent)

    # Run consortium for limited time
    consortium_task = asyncio.create_task(consortium.run())
    await asyncio.sleep(20.0)

    # Shutdown
    coordinator.shutdown()
    research_agent.shutdown()
    optimization_agent.shutdown()
    consortium_task.cancel()

    print("Agent consortium demo completed")


async def demonstrate_agent_learning():
    """Demonstrate agent learning capabilities."""
    print("\n=== Agent Learning Demo ===")

    workspace = Path("./demo_workspace/learning")
    workspace.mkdir(parents=True, exist_ok=True)

    # Create learning agent
    class LearningDemoAgent(MAIFAgent):
        async def run(self):
            logger.info(f"Learning demo agent {self.agent_id} starting")

            # Initial task attempts
            success_rate = 0.3

            for attempt in range(10):
                if self.state == AgentState.TERMINATED:
                    break

                # Attempt task
                perception = await self.perceive(
                    f"Task attempt {attempt + 1}: {'Success' if asyncio.get_event_loop().time() % 10 < success_rate * 10 else 'Failure'}",
                    "text",
                )

                # Reason about result
                reasoning = await self.reason([perception])

                # Plan improvement
                plan = await self.plan("improve task performance", [reasoning])

                # Execute improvement
                results = await self.execute(plan)

                # Learn from experience
                await self.learn([perception, reasoning, plan, results])

                # Improve success rate through learning
                success_rate = min(0.9, success_rate + 0.1)

                logger.info(
                    f"Attempt {attempt + 1}: Success rate now {success_rate:.1%}"
                )

                await asyncio.sleep(1.0)

            logger.info(f"Learning demo agent {self.agent_id} completed")

    agent = LearningDemoAgent("learner_01", str(workspace))
    await agent.run()
    agent.shutdown()

    print("Agent learning demo completed")


async def demonstrate_memory_system():
    """Demonstrate agent memory capabilities."""
    print("\n=== Memory System Demo ===")

    workspace = Path("./demo_workspace/memory")
    workspace.mkdir(parents=True, exist_ok=True)

    # Create agent with memory focus
    agent = AutonomousMAIFAgent("memory_demo", str(workspace))

    # Store various memories
    for i in range(5):
        perception = await agent.perceive(
            f"Memory test {i}: Important data point", "text"
        )
        agent.memory.store_short_term(perception)
        agent.memory.store_working(f"key_{i}", perception)

    # Create episodic memory
    episode = []
    for i in range(3):
        p = await agent.perceive(f"Episode event {i}", "text")
        episode.append(p)
    agent.memory.store_episodic(episode)

    # Demonstrate recall
    print("\nRecent memories:")
    recent = agent.memory.recall_recent(3)
    for mem in recent:
        print(f"- {mem['summary']}")

    print("\nWorking memory:")
    for i in range(3):
        artifact = agent.memory.recall_working(f"key_{i}")
        if artifact:
            print(f"- key_{i}: {artifact.name}")

    print("\nEpisodic recall:")
    episodes = agent.memory.recall_episodes("Episode")
    for ep in episodes:
        print(f"- Episode {ep['episode_id'][:8]}: {len(ep['artifacts'])} events")

    # Flush memory to MAIF
    agent.memory.flush()
    agent.shutdown()

    print("Memory system demo completed")


async def main():
    """Run all demonstrations."""
    print("MAIF-Centric Agent Framework Demo")
    print("=================================")

    # Run demonstrations
    await demonstrate_single_agent()
    await demonstrate_specialized_agents()
    await demonstrate_agent_consortium()
    await demonstrate_agent_learning()
    await demonstrate_memory_system()

    print("\n=== All Demos Completed ===")
    print("\nKey Features Demonstrated:")
    print("- Autonomous agent operation with perception-reasoning-action loop")
    print("- Specialized agent types (Research, Optimization, Coordinator)")
    print("- Multi-agent consortium with coordination")
    print("- Agent learning from experience")
    print("- Sophisticated memory system with short-term, working, and episodic memory")
    print("- All data persisted using MAIF format")
    print("- Integration with MAIF advanced features (ACAM, HSC, CSB)")
    print("- Distributed coordination support")


if __name__ == "__main__":
    asyncio.run(main())
