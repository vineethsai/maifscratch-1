# CrewAI Integration

This guide covers integrating MAIF with CrewAI for cryptographic provenance tracking of multi-agent workflows.

## Overview

The MAIF CrewAI integration provides callback handlers that capture every agent action, task completion, and reasoning step in a tamper-evident MAIF artifact. This enables complete audit trails of your crew's execution without modifying your existing agent code.

**Key Benefits:**

- Tamper-evident execution history via hash chains
- Ed25519 signatures on all logged events
- Full agent reasoning trace (thought, action, observation)
- Task completion tracking with outputs
- Persistent agent memory with provenance
- Compatible with all CrewAI workflow patterns

## Installation

```bash
# Install MAIF with CrewAI integration
pip install maif[integrations]

# Or install CrewAI separately
pip install maif crewai
```

**Requirements:**

- Python 3.10+
- CrewAI 0.30.0+

## Quick Start (One-Liner)

The fastest way to add MAIF to an existing crew:

```python
from crewai import Crew, Agent, Task
from maif.integrations.crewai import instrument

# Create your crew as normal
crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])

# Add MAIF tracking with one line
crew = instrument(crew, "audit.maif")

# Use normally - all actions are tracked
result = crew.kickoff()
# Artifact is automatically finalized
```

## Quick Start (Full Control)

```python
from crewai import Agent, Task, Crew
from maif.integrations.crewai import MAIFCrewCallback

# Create the MAIF callback
callback = MAIFCrewCallback("crew_session.maif")

# Define your agents
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="An expert research analyst",
)

writer = Agent(
    role="Writer", 
    goal="Create clear documentation",
    backstory="A technical writer with years of experience",
)

# Define tasks
research_task = Task(
    description="Research the topic thoroughly",
    expected_output="A detailed research summary",
    agent=researcher,
)

write_task = Task(
    description="Write documentation based on research",
    expected_output="Clear, well-structured documentation",
    agent=writer,
)

# Create crew with MAIF callbacks
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    task_callback=callback.on_task_complete,
    step_callback=callback.on_step,
)

# Run the crew
callback.on_crew_start(crew_name="Documentation Crew", agents=crew.agents, tasks=crew.tasks)
result = crew.kickoff()
callback.on_crew_end(result=result)

# Finalize the artifact
callback.finalize()
```

## API Reference

### MAIFCrewCallback

The main callback handler for tracking CrewAI crew execution.

```python
class MAIFCrewCallback:
    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
        track_tokens: bool = True,
        track_timing: bool = True,
    ):
        """
        Initialize the CrewAI callback handler.
        
        Args:
            artifact_path: Path to the MAIF artifact file
            agent_id: Optional identifier (default: "crewai_callback")
            track_tokens: Whether to track token usage if available
            track_timing: Whether to track execution timing
        """
```

#### Crew Lifecycle Methods

##### on_crew_start(crew_name, agents, tasks, inputs)

Log the start of a crew run. Call this before `crew.kickoff()`.

```python
run_id = callback.on_crew_start(
    crew_name="My Crew",
    agents=crew.agents,
    tasks=crew.tasks,
    inputs={"topic": "AI safety"},
)
```

##### on_crew_end(result, error)

Log crew completion. Call this after `crew.kickoff()` returns.

```python
result = crew.kickoff()
callback.on_crew_end(result=result)

# Or if an error occurred
try:
    result = crew.kickoff()
    callback.on_crew_end(result=result)
except Exception as e:
    callback.on_crew_end(error=e)
```

#### Callback Methods

##### on_task_complete(task_output)

Callback for task completion. Pass this to `Crew(task_callback=...)`.

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    task_callback=callback.on_task_complete,
)
```

##### on_step(step_output)

Callback for agent reasoning steps. Pass this to `Crew(step_callback=...)`.

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    step_callback=callback.on_step,
)
```

#### Utility Methods

##### get_statistics()

Get execution statistics.

```python
stats = callback.get_statistics()
print(f"Tasks: {stats['tasks_completed']}")
print(f"Steps: {stats['steps_executed']}")
print(f"Tool calls: {stats['tool_calls']}")
print(f"Duration: {stats['duration_seconds']:.2f}s")
```

##### reset_statistics()

Reset counters for a new run without creating a new artifact.

```python
callback.reset_statistics()
```

##### finalize()

Finalize the MAIF artifact. Call this when done.

```python
callback.finalize()
```

### MAIFTaskCallback

Standalone task callback for simpler usage when you only need task-level tracking.

```python
from maif.integrations.crewai import MAIFTaskCallback

task_callback = MAIFTaskCallback("tasks.maif")

crew = Crew(
    agents=[...],
    tasks=[...],
    task_callback=task_callback,  # Use directly as callable
)

crew.kickoff()
task_callback.finalize()
```

### MAIFStepCallback

Standalone step callback for tracking agent reasoning.

```python
from maif.integrations.crewai import MAIFStepCallback

step_callback = MAIFStepCallback("steps.maif")

crew = Crew(
    agents=[...],
    tasks=[...],
    step_callback=step_callback,  # Use directly as callable
)

crew.kickoff()
step_callback.finalize()
```

### MAIFCrewMemory

Persistent memory storage for CrewAI agents with provenance tracking.

```python
from maif.integrations.crewai import MAIFCrewMemory

memory = MAIFCrewMemory("agent_memory.maif")

# Store a memory
memory_id = memory.save(
    content="User prefers concise responses",
    agent="researcher",
    tags=["preference", "user"],
    importance=0.8,
)

# Search memories
results = memory.search("user preferences", limit=5)

# Get memories by agent
researcher_memories = memory.get_by_agent("researcher")

# Get memories by tags
important = memory.get_by_tags(["important"], match_all=False)

# Get high-importance memories
critical = memory.get_important(min_importance=0.7)

memory.finalize()
```

#### Memory Methods

| Method | Description |
|--------|-------------|
| `save(content, agent, tags, importance)` | Store a memory |
| `search(query, limit, agent, tags)` | Search by content |
| `get_by_agent(agent, limit)` | Get memories for an agent |
| `get_by_tags(tags, match_all, limit)` | Get memories by tags |
| `get_recent(limit, agent)` | Get most recent memories |
| `get_important(limit, min_importance)` | Get high-importance memories |
| `update_importance(memory_id, importance)` | Update memory importance |
| `delete(memory_id)` | Mark memory as deleted |
| `count(agent, tags)` | Count matching memories |
| `clear_agent_memories(agent)` | Clear all memories for an agent |

## Usage Patterns

### Basic Crew Tracking

```python
from crewai import Crew, Agent, Task
from maif.integrations.crewai import MAIFCrewCallback

callback = MAIFCrewCallback("session.maif")

crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    task_callback=callback.on_task_complete,
    step_callback=callback.on_step,
)

callback.on_crew_start(crew_name="My Crew")
result = crew.kickoff()
callback.on_crew_end(result=result)
callback.finalize()
```

### Sequential Crew Runs

Track multiple crew executions in a single artifact:

```python
callback = MAIFCrewCallback("all_sessions.maif")

# First crew run
callback.on_crew_start(crew_name="Research Crew")
result1 = research_crew.kickoff()
callback.on_crew_end(result=result1)
callback.reset_statistics()

# Second crew run
callback.on_crew_start(crew_name="Writing Crew")
result2 = writing_crew.kickoff()
callback.on_crew_end(result=result2)

callback.finalize()
```

### Error Handling

```python
callback = MAIFCrewCallback("session.maif")

try:
    callback.on_crew_start(crew_name="My Crew")
    result = crew.kickoff()
    callback.on_crew_end(result=result)
except Exception as e:
    callback.on_crew_end(error=e)
finally:
    callback.finalize()
```

### Persistent Agent Memory

```python
from maif.integrations.crewai import MAIFCrewMemory

# Context manager ensures finalization
with MAIFCrewMemory("memory.maif") as memory:
    # Store learned information
    memory.save(
        content="The project deadline is December 15th",
        agent="project_manager",
        tags=["deadline", "project"],
        importance=0.9,
    )
    
    memory.save(
        content="Client prefers weekly status updates",
        agent="project_manager", 
        tags=["client", "communication"],
        importance=0.7,
    )

# Later, in another session
memory = MAIFCrewMemory("memory.maif")
deadlines = memory.search("deadline")
for mem in deadlines:
    print(f"{mem['content']} (importance: {mem['importance']})")
```

### Inspecting the Audit Trail

```python
from maif import MAIFDecoder

decoder = MAIFDecoder("session.maif")
decoder.load()

# Verify integrity
is_valid, errors = decoder.verify_integrity()
print(f"Artifact valid: {is_valid}")

# Inspect events
for block in decoder.blocks:
    event_type = block.metadata.get("type", "unknown")
    timestamp = block.metadata.get("timestamp")
    print(f"{event_type}: {timestamp}")
```

## Events Logged

The CrewAI integration logs the following event types:

| Event Type | Description |
|------------|-------------|
| `SESSION_START` | Callback handler initialized |
| `AGENT_START` | Crew kickoff (via `on_crew_start`) |
| `AGENT_END` | Crew completion (via `on_crew_end`) |
| `AGENT_ERROR` | Crew error (via `on_crew_end` with error) |
| `TASK_END` | Task completion (via `on_task_complete`) |
| `AGENT_ACTION` | Agent reasoning step (via `on_step`) |
| `TOOL_END` | Tool invocation result (detected from step) |
| `MEMORY_SAVE` | Memory stored (via `MAIFCrewMemory.save`) |
| `SESSION_END` | Callback handler finalized |

Each event includes:

- Timestamp
- Run ID (linking related events)
- Event-specific data
- Cryptographic signature

## Performance Considerations

- **Callback Overhead**: The callbacks add minimal overhead to crew execution. Most time is spent in LLM calls.

- **Artifact Size**: Each task and step is logged. For long-running crews with many steps, monitor artifact size.

- **Memory Persistence**: `MAIFCrewMemory` loads all memories into RAM on initialization. For very large memory stores, consider periodic archival.

- **Finalization**: Always call `finalize()` to properly seal the artifact with cryptographic signatures.

## Troubleshooting

### Import Error: CrewAI not installed

```
ImportError: No module named 'crewai'
```

**Solution:** Install CrewAI with `pip install crewai`. Note that CrewAI requires Python 3.10+.

### Callbacks Not Firing

If events are not being logged:

1. Verify you passed the callbacks to the Crew constructor
2. Check that `task_callback` and `step_callback` are set correctly
3. Ensure the crew actually executed (check for early errors)

```python
# Correct
crew = Crew(
    task_callback=callback.on_task_complete,  # Method reference
    step_callback=callback.on_step,
)

# Incorrect - don't call the methods
crew = Crew(
    task_callback=callback.on_task_complete(),  # Wrong: calling the method
)
```

### Integrity Check Failed

```python
is_valid, errors = decoder.verify_integrity()
# is_valid is False
```

**Possible causes:**

1. Artifact was modified externally
2. Incomplete write (process crashed during save)
3. Artifact file corrupted

**Solution:** Investigate the errors list for specific issues. The integrity failure indicates tampering or corruption.

### Memory Not Persisting

If memories are not loading across sessions:

1. Ensure you called `finalize()` on the first session
2. Verify the artifact file exists and is readable
3. Check that you're using the same artifact path

```python
# First session
memory1 = MAIFCrewMemory("memory.maif")
memory1.save(content="Test")
memory1.finalize()  # Required for persistence

# Second session
memory2 = MAIFCrewMemory("memory.maif")
print(len(memory2))  # Should show 1
```

## CLI Tools

MAIF provides CLI tools for inspecting and managing CrewAI artifacts:

```bash
# Inspect artifact details
python -m maif.integrations.crewai.cli inspect crew_audit.maif

# Verify artifact integrity
python -m maif.integrations.crewai.cli verify crew_audit.maif

# List completed tasks
python -m maif.integrations.crewai.cli tasks crew_audit.maif

# List agent reasoning steps
python -m maif.integrations.crewai.cli steps crew_audit.maif

# Export to JSON/CSV/Markdown
python -m maif.integrations.crewai.cli export crew_audit.maif --format json

# Generate HTML audit report
python -m maif.integrations.crewai.cli report crew_audit.maif --open

# Query stored memories
python -m maif.integrations.crewai.cli memory crew_audit.maif --search "deadline"
```

### HTML Report

Generate a visual audit report:

```bash
python -m maif.integrations.crewai.cli report session.maif -o report.html --open
```

This creates a dark-themed HTML report showing:
- Integrity verification status
- Event summary statistics
- Task completion details
- Agent reasoning steps timeline

## Pre-built Patterns

Ready-to-use crew configurations with MAIF tracking:

```python
from maif.integrations.crewai.patterns import (
    create_research_crew,
    create_qa_crew,
    create_code_review_crew,
)

# Research crew (2 agents: researcher + writer)
crew = create_research_crew(
    "research.maif",
    topic="AI security best practices",
    llm=my_llm,
)
result = crew.kickoff()

# QA crew (2 agents: analyst + responder)
crew = create_qa_crew(
    "qa.maif",
    context="Your document text here...",
    llm=my_llm,
)

# Code review crew (3 agents: security + quality + summarizer)
crew = create_code_review_crew(
    "review.maif",
    code="def example(): pass",
    language="python",
    llm=my_llm,
)
```

## Related

- [MAIF Overview](../getting-started.md)
- [Framework Integrations](./index.md)
- [LangGraph Integration](./langgraph.md)
- [CrewAI Documentation](https://docs.crewai.com/)

