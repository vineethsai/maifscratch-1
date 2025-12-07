"""
Pre-built CrewAI patterns with MAIF provenance.

These patterns provide ready-to-use crew configurations for common
use cases, all with built-in cryptographic provenance tracking.

Usage:
    # Pre-built patterns
    from maif.integrations.crewai.patterns import create_research_crew
    
    crew = create_research_crew(
        "research_audit.maif",
        topic="AI security best practices",
        llm=my_llm,
    )
    result = crew.kickoff()
    
    # Instrument existing crews (one-liner!)
    from maif.integrations.crewai import instrument
    
    crew = Crew(agents=[...], tasks=[...])
    crew = instrument(crew, "audit.maif")
    result = crew.kickoff()
"""

from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import functools

from .callback import MAIFCrewCallback

# Check if CrewAI is available
try:
    from crewai import Agent, Task, Crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


# =============================================================================
# One-liner Instrumentation
# =============================================================================

def instrument(
    crew: "Crew",
    artifact_path: Union[str, Path],
    *,
    agent_id: str = "crewai",
    auto_finalize: bool = True,
) -> "Crew":
    """Instrument an existing CrewAI crew with MAIF provenance tracking.
    
    This is the simplest way to add MAIF to an existing crew. Just wrap
    your crew with this function and all actions will be tracked.
    
    Args:
        crew: An existing CrewAI Crew instance
        artifact_path: Path for the MAIF artifact
        agent_id: Agent identifier for provenance
        auto_finalize: If True, automatically finalize after kickoff
        
    Returns:
        The same crew, instrumented with MAIF callbacks
        
    Example:
        from crewai import Crew, Agent, Task
        from maif.integrations.crewai import instrument
        
        # Create your crew as normal
        crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
        
        # Add MAIF tracking with one line
        crew = instrument(crew, "audit.maif")
        
        # Use normally - all actions are tracked
        result = crew.kickoff()
        
        # That's it! Artifact is automatically finalized.
    """
    if not CREWAI_AVAILABLE:
        raise ImportError(
            "CrewAI is required for instrumentation. "
            "Install with: pip install crewai"
        )
    
    # Create callback
    callback = MAIFCrewCallback(artifact_path, agent_id=agent_id)
    
    # Store original kickoff method
    original_kickoff = crew.kickoff
    
    @functools.wraps(original_kickoff)
    def instrumented_kickoff(*args, **kwargs):
        """Wrapped kickoff that adds MAIF tracking."""
        # Log crew start
        callback.on_crew_start(
            crew_name=getattr(crew, 'name', 'Instrumented Crew'),
            agents=getattr(crew, 'agents', None),
            tasks=getattr(crew, 'tasks', None),
            inputs=kwargs.get('inputs'),
        )
        
        try:
            # Run the crew
            result = original_kickoff(*args, **kwargs)
            
            # Log completion
            callback.on_crew_end(result=result)
            
            return result
            
        except Exception as e:
            # Log error
            callback.on_crew_end(error=e)
            raise
            
        finally:
            if auto_finalize:
                callback.finalize()
    
    # Replace kickoff method
    crew.kickoff = instrumented_kickoff
    
    # Set callbacks on crew if it supports them
    if hasattr(crew, '_task_callback'):
        crew._task_callback = callback.on_task_complete
    else:
        crew.task_callback = callback.on_task_complete
        
    if hasattr(crew, '_step_callback'):
        crew._step_callback = callback.on_step
    else:
        crew.step_callback = callback.on_step
    
    # Store callback for later access
    crew._maif_callback = callback
    
    return crew


def _require_crewai():
    """Raise error if CrewAI is not available."""
    if not CREWAI_AVAILABLE:
        raise ImportError(
            "CrewAI is required for these patterns. "
            "Install with: pip install crewai"
        )


# =============================================================================
# Pattern: Research Crew
# =============================================================================

def create_research_crew(
    artifact_path: str,
    topic: str,
    *,
    llm: Optional[Any] = None,
    verbose: bool = True,
    agent_id: str = "research_crew",
) -> "Crew":
    """Create a research crew with MAIF provenance.
    
    This creates a two-agent research crew:
    - Researcher: Gathers and analyzes information
    - Writer: Creates documentation from findings
    
    Args:
        artifact_path: Path for the MAIF artifact
        topic: Research topic
        llm: Optional LLM to use (uses default if not provided)
        verbose: Whether to show verbose output
        agent_id: Agent identifier for provenance
        
    Returns:
        Configured Crew ready to kickoff()
        
    Example:
        crew = create_research_crew(
            "research.maif",
            topic="AI agent security",
            llm=my_llm,
        )
        result = crew.kickoff()
    """
    _require_crewai()
    
    # Create callback
    callback = MAIFCrewCallback(artifact_path, agent_id=agent_id)
    
    # Create agents
    agent_kwargs = {"llm": llm} if llm else {}
    
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Conduct thorough research on: {topic}",
        backstory="""You are an experienced research analyst with expertise in 
        gathering and synthesizing information. You are known for your attention 
        to detail and ability to identify key insights from complex topics.""",
        verbose=verbose,
        allow_delegation=False,
        **agent_kwargs,
    )
    
    writer = Agent(
        role="Technical Writer",
        goal="Create clear, well-structured documentation from research findings",
        backstory="""You are a skilled technical writer who excels at transforming 
        complex research into accessible documentation. You focus on clarity, 
        organization, and reader comprehension.""",
        verbose=verbose,
        allow_delegation=False,
        **agent_kwargs,
    )
    
    # Create tasks
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
    
    # Create crew with MAIF callbacks
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        task_callback=callback.on_task_complete,
        step_callback=callback.on_step,
        verbose=verbose,
    )
    
    # Attach callback for later finalization
    crew._maif_callback = callback
    
    return crew


# =============================================================================
# Pattern: QA Crew
# =============================================================================

def create_qa_crew(
    artifact_path: str,
    context: str,
    *,
    llm: Optional[Any] = None,
    verbose: bool = True,
    agent_id: str = "qa_crew",
) -> "Crew":
    """Create a QA (Question-Answering) crew with MAIF provenance.
    
    This creates a two-agent QA crew:
    - Analyst: Analyzes the context and extracts relevant information
    - Responder: Formulates clear, accurate answers
    
    Args:
        artifact_path: Path for the MAIF artifact
        context: The context/document to answer questions about
        llm: Optional LLM to use
        verbose: Whether to show verbose output
        agent_id: Agent identifier for provenance
        
    Returns:
        Configured Crew ready to kickoff()
    """
    _require_crewai()
    
    callback = MAIFCrewCallback(artifact_path, agent_id=agent_id)
    
    agent_kwargs = {"llm": llm} if llm else {}
    
    analyst = Agent(
        role="Information Analyst",
        goal="Extract and analyze relevant information from provided context",
        backstory="""You are an expert at reading comprehension and information 
        extraction. You can quickly identify key facts, relationships, and 
        relevant details from any text.""",
        verbose=verbose,
        allow_delegation=False,
        **agent_kwargs,
    )
    
    responder = Agent(
        role="Response Specialist",
        goal="Provide clear, accurate, and helpful answers based on analysis",
        backstory="""You are skilled at formulating clear, well-structured 
        responses. You ensure answers are accurate, relevant, and easy to 
        understand while citing sources when appropriate.""",
        verbose=verbose,
        allow_delegation=False,
        **agent_kwargs,
    )
    
    analyze_task = Task(
        description=f"""Analyze the following context and extract key information:
        
        CONTEXT:
        {context[:2000]}  # Truncate for safety
        
        Identify:
        1. Main topics covered
        2. Key facts and data points
        3. Important relationships or connections
        4. Any limitations or caveats""",
        expected_output="A structured analysis of the context with key information extracted",
        agent=analyst,
    )
    
    respond_task = Task(
        description="""Based on the analysis, formulate a comprehensive response that:
        
        1. Addresses the main points clearly
        2. Provides accurate information from the context
        3. Acknowledges any limitations
        4. Is well-organized and easy to follow""",
        expected_output="A clear, accurate response based on the analyzed context",
        agent=responder,
    )
    
    crew = Crew(
        agents=[analyst, responder],
        tasks=[analyze_task, respond_task],
        task_callback=callback.on_task_complete,
        step_callback=callback.on_step,
        verbose=verbose,
    )
    
    crew._maif_callback = callback
    return crew


# =============================================================================
# Pattern: Code Review Crew
# =============================================================================

def create_code_review_crew(
    artifact_path: str,
    code: str,
    language: str = "python",
    *,
    llm: Optional[Any] = None,
    verbose: bool = True,
    agent_id: str = "code_review_crew",
) -> "Crew":
    """Create a code review crew with MAIF provenance.
    
    This creates a three-agent code review crew:
    - Security Analyst: Reviews for security vulnerabilities
    - Quality Analyst: Reviews for code quality and best practices
    - Summarizer: Consolidates findings into actionable feedback
    
    Args:
        artifact_path: Path for the MAIF artifact
        code: The code to review
        language: Programming language
        llm: Optional LLM to use
        verbose: Whether to show verbose output
        agent_id: Agent identifier for provenance
        
    Returns:
        Configured Crew ready to kickoff()
    """
    _require_crewai()
    
    callback = MAIFCrewCallback(artifact_path, agent_id=agent_id)
    
    agent_kwargs = {"llm": llm} if llm else {}
    
    security_analyst = Agent(
        role="Security Analyst",
        goal="Identify security vulnerabilities and potential risks in code",
        backstory="""You are a security expert specializing in code review.
        You can identify common vulnerabilities like injection attacks, 
        authentication issues, and data exposure risks.""",
        verbose=verbose,
        allow_delegation=False,
        **agent_kwargs,
    )
    
    quality_analyst = Agent(
        role="Code Quality Analyst",
        goal="Evaluate code quality, maintainability, and adherence to best practices",
        backstory="""You are an expert in software engineering best practices.
        You review code for readability, maintainability, proper error handling,
        and adherence to language conventions.""",
        verbose=verbose,
        allow_delegation=False,
        **agent_kwargs,
    )
    
    summarizer = Agent(
        role="Review Summarizer",
        goal="Consolidate findings into clear, actionable feedback",
        backstory="""You excel at synthesizing technical feedback into clear,
        prioritized recommendations that developers can act on.""",
        verbose=verbose,
        allow_delegation=False,
        **agent_kwargs,
    )
    
    security_task = Task(
        description=f"""Review this {language} code for security issues:
        
        ```{language}
        {code[:3000]}
        ```
        
        Check for:
        1. Input validation issues
        2. Authentication/authorization flaws
        3. Data exposure risks
        4. Injection vulnerabilities
        5. Cryptographic weaknesses""",
        expected_output="Security analysis with identified vulnerabilities and risk levels",
        agent=security_analyst,
    )
    
    quality_task = Task(
        description=f"""Review this {language} code for quality:
        
        ```{language}
        {code[:3000]}
        ```
        
        Evaluate:
        1. Code readability and organization
        2. Error handling
        3. Naming conventions
        4. Potential bugs or edge cases
        5. Performance considerations""",
        expected_output="Quality analysis with improvement suggestions",
        agent=quality_analyst,
    )
    
    summary_task = Task(
        description="""Based on the security and quality analyses, create a
        prioritized summary of findings:
        
        1. Critical issues that must be addressed
        2. Important improvements to consider
        3. Minor suggestions for polish
        
        Include specific line references or code examples where helpful.""",
        expected_output="Prioritized, actionable code review summary",
        agent=summarizer,
    )
    
    crew = Crew(
        agents=[security_analyst, quality_analyst, summarizer],
        tasks=[security_task, quality_task, summary_task],
        task_callback=callback.on_task_complete,
        step_callback=callback.on_step,
        verbose=verbose,
    )
    
    crew._maif_callback = callback
    return crew


# =============================================================================
# Utility Functions
# =============================================================================

def finalize_crew(crew: "Crew") -> None:
    """Finalize the MAIF artifact for a crew.
    
    Call this when you're done using a crew to seal the artifact.
    
    Args:
        crew: A crew created with one of the pattern functions
    """
    if hasattr(crew, "_maif_callback") and crew._maif_callback:
        crew._maif_callback.finalize()


def get_artifact_path(crew: "Crew") -> Optional[str]:
    """Get the MAIF artifact path for a crew.
    
    Args:
        crew: A crew created with one of the pattern functions
        
    Returns:
        Path to the MAIF artifact, or None if not using MAIF
    """
    if hasattr(crew, "_maif_callback") and crew._maif_callback:
        return crew._maif_callback.get_artifact_path()
    return None


def get_crew_statistics(crew: "Crew") -> Dict[str, Any]:
    """Get statistics from a crew's MAIF callback.
    
    Args:
        crew: A crew created with one of the pattern functions
        
    Returns:
        Dictionary with execution statistics
    """
    if hasattr(crew, "_maif_callback") and crew._maif_callback:
        return crew._maif_callback.get_statistics()
    return {}


# =============================================================================
# Context Manager
# =============================================================================

class MAIFCrew:
    """Context manager for CrewAI crews with automatic MAIF finalization.
    
    Usage:
        with MAIFCrew("audit.maif") as callback:
            crew = Crew(
                agents=[...],
                tasks=[...],
                task_callback=callback.on_task_complete,
                step_callback=callback.on_step,
            )
            result = crew.kickoff()
        # Artifact is automatically finalized
    """
    
    def __init__(
        self,
        artifact_path: str,
        agent_id: str = "crewai",
        **kwargs,
    ):
        """Initialize the context manager.
        
        Args:
            artifact_path: Path for the MAIF artifact
            agent_id: Agent identifier for provenance
            **kwargs: Additional arguments for MAIFCrewCallback
        """
        self.artifact_path = artifact_path
        self.agent_id = agent_id
        self.kwargs = kwargs
        self.callback: Optional[MAIFCrewCallback] = None
    
    def __enter__(self) -> MAIFCrewCallback:
        """Enter context and create callback."""
        self.callback = MAIFCrewCallback(
            self.artifact_path,
            agent_id=self.agent_id,
            **self.kwargs,
        )
        return self.callback
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and finalize artifact."""
        if self.callback:
            if exc_type is not None:
                # Log error if one occurred
                self.callback.on_crew_end(error=exc_val)
            self.callback.finalize()
        return False  # Don't suppress exceptions

