"""
CrewAI Crew Runner with MAIF Provenance Tracking.

This module provides the CrewRunner class that executes CrewAI crews
while maintaining full cryptographic provenance in MAIF artifacts.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# Terminal formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"

# Check if CrewAI is available
try:
    from crewai import Agent, Task, Crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

from maif.integrations.crewai import MAIFCrewCallback


class CrewRunner:
    """Runs CrewAI crews with MAIF provenance tracking."""
    
    def __init__(
        self,
        artifact_path: str,
        session_name: str,
    ):
        """Initialize the crew runner.
        
        Args:
            artifact_path: Path to the MAIF artifact file
            session_name: Name of the session
        """
        self.artifact_path = artifact_path
        self.session_name = session_name
        
        # Initialize MAIF callback
        self.callback = MAIFCrewCallback(
            artifact_path=artifact_path,
            agent_id=f"crewai_{session_name}",
        )
        
        self._finalized = False
    
    def run_research_crew(self, topic: str) -> bool:
        """Run a research crew on the given topic.
        
        Args:
            topic: The research topic
            
        Returns:
            True if successful, False otherwise
        """
        if not CREWAI_AVAILABLE:
            print(f"{RED}CrewAI is not installed. Install with: pip install crewai{RESET}")
            return False
        
        print(f"{DIM}Creating agents...{RESET}")
        
        # Create the Researcher agent
        researcher = Agent(
            role="Senior Research Analyst",
            goal=f"Conduct thorough research on: {topic}",
            backstory="""You are an experienced research analyst with expertise in 
            gathering and synthesizing information. You are known for your attention 
            to detail and ability to identify key insights from complex topics.""",
            verbose=True,
            allow_delegation=False,
        )
        
        # Create the Writer agent
        writer = Agent(
            role="Technical Writer",
            goal="Create clear, well-structured documentation from research findings",
            backstory="""You are a skilled technical writer who excels at transforming 
            complex research into accessible documentation. You focus on clarity, 
            organization, and reader comprehension.""",
            verbose=True,
            allow_delegation=False,
        )
        
        print(f"  {GREEN}Created:{RESET} Senior Research Analyst")
        print(f"  {GREEN}Created:{RESET} Technical Writer")
        print()
        
        print(f"{DIM}Creating tasks...{RESET}")
        
        # Create research task
        research_task = Task(
            description=f"""Research the topic: {topic}
            
            Provide a comprehensive analysis including:
            1. Key concepts and definitions
            2. Current state and best practices
            3. Important considerations and challenges
            4. Relevant examples or recommendations
            
            Be thorough but concise in your findings.""",
            expected_output="A detailed research summary with key findings and actionable insights",
            agent=researcher,
        )
        
        # Create writing task
        writing_task = Task(
            description="""Based on the research provided, create documentation that:
            
            1. Introduces the topic clearly
            2. Explains key concepts in accessible language
            3. Discusses important considerations
            4. Provides actionable recommendations
            5. Includes a brief summary
            
            Keep the documentation professional and well-organized.""",
            expected_output="Well-structured documentation suitable for technical readers",
            agent=writer,
        )
        
        print(f"  {GREEN}Created:{RESET} Research Task")
        print(f"  {GREEN}Created:{RESET} Documentation Task")
        print()
        
        print(f"{DIM}Assembling crew...{RESET}")
        
        # Create the crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            task_callback=self.callback.on_task_complete,
            step_callback=self.callback.on_step,
            verbose=True,
        )
        
        print(f"  {GREEN}Crew assembled with 2 agents and 2 tasks{RESET}")
        print()
        
        # Log crew start
        self.callback.on_crew_start(
            crew_name=f"Research Crew: {topic[:30]}...",
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            inputs={"topic": topic},
        )
        
        print(f"{BOLD}Starting crew execution...{RESET}")
        print_separator()
        print()
        
        try:
            # Run the crew
            result = crew.kickoff()
            
            # Log crew completion
            self.callback.on_crew_end(result=result)
            
            print()
            print_separator()
            print()
            print(f"{BOLD}Crew Output:{RESET}")
            print()
            
            # Display the result
            if hasattr(result, 'raw'):
                output = result.raw
            else:
                output = str(result)
            
            # Truncate if too long
            if len(output) > 2000:
                output = output[:2000] + "\n... [truncated]"
            
            print(output)
            
            return True
            
        except Exception as e:
            print(f"\n{RED}Error during crew execution: {e}{RESET}")
            self.callback.on_crew_end(error=e)
            return False
    
    def run_simple_demo(self) -> bool:
        """Run a simple demo without requiring LLM API keys.
        
        This creates a mock crew execution for demonstration purposes.
        
        Returns:
            True if successful
        """
        print(f"{DIM}Running demo mode (no LLM required)...{RESET}")
        print()
        
        # Log crew start
        self.callback.on_crew_start(
            crew_name="Demo Research Crew",
            agents=None,
            tasks=None,
            inputs={"mode": "demo"},
        )
        
        # Simulate task completions
        class MockTaskOutput:
            def __init__(self, description, raw, agent):
                self.description = description
                self.raw = raw
                self.agent = agent
                self.output_format = "raw"
        
        class MockStepOutput:
            def __init__(self, thought, action, action_input, observation):
                self.thought = thought
                self.action = action
                self.action_input = action_input
                self.observation = observation
        
        # Simulate researcher steps
        print(f"{CYAN}[RESEARCHER]{RESET} Starting research...")
        
        steps = [
            MockStepOutput(
                thought="I need to gather information about the topic",
                action="search",
                action_input="AI agent security best practices",
                observation="Found several relevant sources on agent security"
            ),
            MockStepOutput(
                thought="Let me analyze the key points from these sources",
                action="analyze",
                action_input="Analyzing security patterns and recommendations",
                observation="Identified 5 key security practices"
            ),
            MockStepOutput(
                thought="I have enough information to summarize",
                action="Final Answer",
                action_input="Compiling research summary",
                observation=""
            ),
        ]
        
        for step in steps:
            self.callback.on_step(step)
            print(f"  {DIM}Step:{RESET} {step.action}")
        
        # Task 1 complete
        task1 = MockTaskOutput(
            description="Research AI agent security",
            raw="Key findings: 1) Input validation is critical, 2) Principle of least privilege, 3) Audit logging essential, 4) Secure tool execution, 5) Rate limiting recommended",
            agent="researcher"
        )
        self.callback.on_task_complete(task1)
        print(f"  {GREEN}Task completed:{RESET} Research")
        print()
        
        # Simulate writer steps
        print(f"{CYAN}[WRITER]{RESET} Starting documentation...")
        
        steps = [
            MockStepOutput(
                thought="I need to structure the research findings",
                action="outline",
                action_input="Creating document outline",
                observation="Outline created with 5 sections"
            ),
            MockStepOutput(
                thought="Now I'll write each section",
                action="write",
                action_input="Writing documentation sections",
                observation="Documentation draft completed"
            ),
            MockStepOutput(
                thought="Documentation is ready",
                action="Final Answer",
                action_input="Final documentation",
                observation=""
            ),
        ]
        
        for step in steps:
            self.callback.on_step(step)
            print(f"  {DIM}Step:{RESET} {step.action}")
        
        # Task 2 complete
        task2 = MockTaskOutput(
            description="Write security documentation",
            raw="""# AI Agent Security Best Practices

## Overview
This document outlines essential security practices for AI agent systems.

## Key Practices

1. **Input Validation**: Always validate and sanitize all inputs to agents.
2. **Least Privilege**: Agents should have minimal required permissions.
3. **Audit Logging**: Log all agent actions for accountability.
4. **Secure Tools**: Implement proper security for agent tool execution.
5. **Rate Limiting**: Prevent abuse through request throttling.

## Conclusion
Following these practices helps ensure secure AI agent deployments.""",
            agent="writer"
        )
        self.callback.on_task_complete(task2)
        print(f"  {GREEN}Task completed:{RESET} Documentation")
        print()
        
        # Log crew completion
        class MockCrewOutput:
            def __init__(self):
                self.raw = task2.raw
                self.tasks_output = [task1, task2]
        
        self.callback.on_crew_end(result=MockCrewOutput())
        
        print(f"{BOLD}Demo Output:{RESET}")
        print()
        print(task2.raw)
        
        return True
    
    def show_session_summary(self):
        """Show a summary of the current session."""
        stats = self.callback.get_statistics()
        
        print(f"Tasks completed: {stats.get('tasks_completed', 0)}")
        print(f"Steps executed: {stats.get('steps_executed', 0)}")
        print(f"Tool calls: {stats.get('tool_calls', 0)}")
        
        if stats.get('duration_seconds'):
            print(f"Duration: {stats['duration_seconds']:.2f}s")
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize the session and return summary.
        
        Returns:
            Summary dictionary with session statistics
        """
        if self._finalized:
            return {}
        
        stats = self.callback.get_statistics()
        self.callback.finalize()
        self._finalized = True
        
        # Get block count from artifact
        try:
            from maif import MAIFDecoder
            decoder = MAIFDecoder(self.artifact_path)
            decoder.load()
            total_events = len(decoder.blocks)
        except Exception:
            total_events = 0
        
        return {
            "tasks_completed": stats.get("tasks_completed", 0),
            "steps_executed": stats.get("steps_executed", 0),
            "tool_calls": stats.get("tool_calls", 0),
            "errors": stats.get("errors", 0),
            "duration_seconds": stats.get("duration_seconds", 0),
            "total_events": total_events,
        }


def print_separator(char="-", width=60):
    """Print a separator line."""
    print(char * width)

