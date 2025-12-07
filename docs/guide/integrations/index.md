# Framework Integrations

MAIF provides drop-in integrations for popular AI agent frameworks, enabling cryptographic provenance tracking with minimal code changes.

::: tip DeepWiki Documentation
For detailed architecture and implementation docs, see **[DeepWiki - Framework Integrations](https://deepwiki.com/vineethsai/maif/4-framework-integrations)**.
:::

## Available Integrations

| Framework | Status | Description |
|-----------|--------|-------------|
| [LangGraph](./langgraph.md) | Available | State checkpointer with provenance |
| [CrewAI](./crewai.md) | Available | Crew/Agent callbacks, Memory |
| [LangChain](./langchain.md) | Available | Callbacks, VectorStore, Memory |
| [Strands SDK](./strands.md) | Coming Soon | AWS Strands agent callbacks |

## Installation

Install MAIF with all integrations:

```bash
pip install maif[integrations]
```

Or install specific framework support:

```bash
# LangGraph only
pip install maif langgraph

# LangChain only
pip install maif langchain-core

# CrewAI only
pip install maif crewai
```

## Quick Start

Each integration follows the same pattern: create a MAIF-backed handler, attach it to your framework, and finalize when done.

### LangGraph Example

```python
from langgraph.graph import StateGraph
from maif.integrations.langgraph import MAIFCheckpointer

checkpointer = MAIFCheckpointer("state.maif")
app = graph.compile(checkpointer=checkpointer)
result = app.invoke(state, config)
checkpointer.finalize()
```

### LangChain Example

```python
from langchain_openai import ChatOpenAI
from maif.integrations.langchain import MAIFCallbackHandler

handler = MAIFCallbackHandler("session.maif")
llm = ChatOpenAI(callbacks=[handler])
response = llm.invoke("Hello!")
handler.finalize()
```

### CrewAI Example

```python
from crewai import Crew, Agent, Task
from maif.integrations.crewai import MAIFCrewCallback

callback = MAIFCrewCallback("crew.maif")

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    task_callback=callback.on_task_complete,
    step_callback=callback.on_step,
)

callback.on_crew_start(crew_name="My Crew", agents=crew.agents, tasks=crew.tasks)
result = crew.kickoff()
callback.on_crew_end(result=result)
callback.finalize()
```

## Common Patterns

### Context Manager

All integrations support context manager usage for automatic finalization:

```python
with MAIFCheckpointer("state.maif") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    result = app.invoke(state, config)
# Automatically finalized
```

### Inspecting Provenance

After running your agent, inspect the audit trail:

```python
from maif import MAIFDecoder

decoder = MAIFDecoder("session.maif")
decoder.load()

# Verify integrity
is_valid, errors = decoder.verify_integrity()

# Inspect events
for block in decoder.blocks:
    print(f"{block.metadata.get('type')}: {block.metadata.get('timestamp')}")
```

### Multi-Session Support

Use thread IDs to manage multiple sessions in one artifact:

```python
# Session 1
app.invoke(state, {"configurable": {"thread_id": "user-alice"}})

# Session 2
app.invoke(state, {"configurable": {"thread_id": "user-bob"}})
```

## Architecture

All integrations share a common architecture:

```
Framework Event -> MAIF Callback -> MAIFProvenanceTracker -> MAIFEncoder -> .maif file
```

The base classes in `maif.integrations` provide:

- **EventType**: Standardized event types across frameworks
- **MAIFProvenanceTracker**: Core logging functionality
- **BaseMAIFCallback**: Abstract base for framework callbacks
- **Utility functions**: Safe serialization, timestamp formatting

## Interactive Demo

For a comprehensive demonstration of MAIF's governance features with LangGraph, run the interactive demo:

```bash
cd examples/integrations/langgraph_governance_demo
python main.py
```

This demo showcases:
- Multi-agent workflows with access control
- Cryptographic provenance and tamper detection
- Security verification
- Compliance report generation
- Role-based access control simulation

## Contributing

To add support for a new framework, see the [Integration Plan](../../../maif/integrations/INTEGRATION_PLAN.md) for detailed implementation guidance.

