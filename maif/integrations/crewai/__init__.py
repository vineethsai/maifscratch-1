"""
MAIF CrewAI Integration

Provides MAIF-backed provenance tracking for CrewAI multi-agent workflows.
All agent actions, task completions, and inter-agent communications are
automatically logged to MAIF artifacts with cryptographic signatures.

Quick Start:
    from crewai import Crew, Agent, Task
    from maif.integrations.crewai import MAIFCrewCallback

    callback = MAIFCrewCallback("crew_session.maif")
    
    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task1, task2],
        task_callback=callback.on_task_complete,
        step_callback=callback.on_step,
    )
    
    result = crew.kickoff()
    callback.finalize()

With Context Manager:
    from maif.integrations.crewai import MAIFCrew
    
    with MAIFCrew("crew_session.maif") as callback:
        crew = Crew(
            agents=[...], tasks=[...],
            task_callback=callback.on_task_complete,
            step_callback=callback.on_step,
        )
        result = crew.kickoff()
    # Artifact automatically finalized

Pre-built Patterns:
    from maif.integrations.crewai.patterns import create_research_crew
    
    crew = create_research_crew("research.maif", topic="AI security")
    result = crew.kickoff()

CLI Tools:
    python -m maif.integrations.crewai.cli inspect crew_audit.maif
    python -m maif.integrations.crewai.cli verify crew_audit.maif
    python -m maif.integrations.crewai.cli tasks crew_audit.maif

All agent reasoning steps, tool usage, and task outputs are captured
in the MAIF artifact with full provenance and tamper-evident audit trails.
"""

from maif.integrations.crewai.callback import (
    MAIFCrewCallback,
    MAIFTaskCallback,
    MAIFStepCallback,
)

# Import patterns, context manager, and instrumentation
from maif.integrations.crewai.patterns import (
    MAIFCrew,
    instrument,
    create_research_crew,
    create_qa_crew,
    create_code_review_crew,
    finalize_crew,
    get_artifact_path,
    get_crew_statistics,
)

try:
    from maif.integrations.crewai.memory import MAIFCrewMemory
    _MEMORY_AVAILABLE = True
except ImportError:
    MAIFCrewMemory = None  # type: ignore
    _MEMORY_AVAILABLE = False

__all__ = [
    # Core callback
    "MAIFCrewCallback",
    "MAIFTaskCallback", 
    "MAIFStepCallback",
    # Context manager
    "MAIFCrew",
    # One-liner instrumentation
    "instrument",
    # Pre-built patterns
    "create_research_crew",
    "create_qa_crew",
    "create_code_review_crew",
    # Utilities
    "finalize_crew",
    "get_artifact_path",
    "get_crew_statistics",
]

if _MEMORY_AVAILABLE:
    __all__.append("MAIFCrewMemory")
