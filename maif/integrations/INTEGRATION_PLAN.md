# MAIF Framework Integrations - Implementation Guide

This document provides detailed guidance for implementing MAIF integrations with AI agent frameworks. It is intended for developers (human or AI agents) who are adding support for new frameworks or maintaining existing integrations.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Base Classes Reference](#base-classes-reference)
3. [Implementation Checklist](#implementation-checklist)
4. [Framework-Specific Guides](#framework-specific-guides)
   - [LangGraph](#langgraph-integration) (Implemented)
   - [LangChain](#langchain-integration) (To implement)
   - [CrewAI](#crewai-integration) (To implement)
   - [Strands SDK](#strands-sdk-integration) (To implement)
5. [Testing Requirements](#testing-requirements)
6. [Documentation Standards](#documentation-standards)

---

## Architecture Overview

### Design Principles

1. **One-Line Activation**: Users should enable MAIF with minimal code changes
2. **Non-Invasive**: Never require users to modify their existing agent code
3. **Framework-Native**: Use each framework's extension patterns (callbacks, hooks, etc.)
4. **Graceful Degradation**: If MAIF is not configured, code should still work
5. **Automatic Provenance**: Capture agent actions without explicit logging calls

### Package Structure

```
maif/integrations/
    __init__.py          # Package exports and lazy loading
    _base.py             # MAIFProvenanceTracker, BaseMAIFCallback, EventType
    _utils.py            # safe_serialize, format_timestamp, etc.
    INTEGRATION_PLAN.md  # This file
    
    langgraph/           # LangGraph integration
        __init__.py
        checkpointer.py  # MAIFCheckpointer
    
    langchain/           # LangChain integration
        __init__.py
        callback.py      # MAIFCallbackHandler
        vectorstore.py   # MAIFVectorStore
        memory.py        # MAIFChatMessageHistory
    
    crewai/              # CrewAI integration
        __init__.py
        callback.py      # MAIFCrewCallback
        memory.py        # MAIFCrewMemory
    
    strands/             # AWS Strands SDK integration
        __init__.py
        callback.py      # MAIFStrandsCallback
```

### Event Flow

```
User Code -> Framework Callback/Hook -> BaseMAIFCallback -> MAIFProvenanceTracker -> MAIFEncoder -> .maif file
```

---

## Base Classes Reference

### EventType (enum)

Standardized event types for consistent logging across frameworks:

```python
from maif.integrations import EventType

# LLM Events
EventType.LLM_START       # LLM invocation started
EventType.LLM_END         # LLM invocation completed
EventType.LLM_ERROR       # LLM error occurred

# Chain/Workflow Events
EventType.CHAIN_START     # Chain execution started
EventType.CHAIN_END       # Chain execution completed
EventType.CHAIN_ERROR     # Chain error occurred

# Tool Events
EventType.TOOL_START      # Tool invocation started
EventType.TOOL_END        # Tool invocation completed
EventType.TOOL_ERROR      # Tool error occurred

# Retrieval Events
EventType.RETRIEVAL_START # Retrieval query started
EventType.RETRIEVAL_END   # Retrieval results received

# Agent Events
EventType.AGENT_START     # Agent execution started
EventType.AGENT_END       # Agent execution completed
EventType.AGENT_ACTION    # Agent took an action
EventType.AGENT_ERROR     # Agent error occurred

# Task Events (CrewAI specific)
EventType.TASK_START      # Task started
EventType.TASK_END        # Task completed
EventType.TASK_ERROR      # Task error occurred

# State Events (LangGraph specific)
EventType.STATE_CHECKPOINT # State checkpoint saved
EventType.STATE_RESTORE    # State restored from checkpoint
EventType.NODE_START       # Graph node started
EventType.NODE_END         # Graph node completed

# Memory Events
EventType.MEMORY_SAVE     # Memory saved
EventType.MEMORY_LOAD     # Memory loaded

# Session Events
EventType.SESSION_START   # Session started
EventType.SESSION_END     # Session ended

# Custom Events
EventType.CUSTOM          # Custom event type
```

### MAIFProvenanceTracker

Core class for logging events to MAIF artifacts:

```python
from maif.integrations import MAIFProvenanceTracker, EventType

# Basic usage
tracker = MAIFProvenanceTracker(
    artifact_path="session.maif",
    agent_id="my-agent"
)

# Log an event
block_id = tracker.log_event(
    event_type=EventType.LLM_START,
    data={
        "model": "gemini-2.0-flash",
        "prompt": "What is the capital of France?",
    },
    metadata={"user_id": "123"},
    run_id="run_abc123",
)

# Log binary data (embeddings, etc.)
tracker.log_binary_event(
    event_type=EventType.RETRIEVAL_END,
    data=embedding_bytes,
    metadata={"dimensions": 384},
)

# Finalize (required to seal the artifact)
tracker.finalize()

# Or use as context manager
with MAIFProvenanceTracker("session.maif") as tracker:
    tracker.log_event(EventType.LLM_START, {"model": "gpt-4"})
    # auto-finalized on exit
```

### BaseMAIFCallback

Abstract base class for framework callbacks:

```python
from maif.integrations import BaseMAIFCallback, EventType

class MyFrameworkCallback(BaseMAIFCallback):
    def __init__(self, artifact_path: str, agent_id: str = None):
        super().__init__(artifact_path, agent_id)
    
    def get_framework_name(self) -> str:
        return "my_framework"
    
    def on_event(self, event_data: dict):
        self.tracker.log_event(
            EventType.CUSTOM,
            data=event_data,
        )
```

### Utility Functions

```python
from maif.integrations import (
    safe_serialize,      # JSON serialize with type handling
    format_timestamp,    # Unix timestamp to ISO string
    generate_run_id,     # Generate unique run ID
    truncate_string,     # Truncate long strings
    extract_error_info,  # Extract exception details
)

# Safe serialization handles UUIDs, datetimes, bytes, custom objects
json_str = safe_serialize({"uuid": some_uuid, "time": datetime.now()})

# Format timestamps
iso_time = format_timestamp(time.time())

# Generate run IDs
run_id = generate_run_id()  # e.g., "a1b2c3d4-..."

# Truncate long strings
short = truncate_string(very_long_string, max_length=1000)

# Extract error info
error_dict = extract_error_info(exception)
# {"error_type": "ValueError", "error_message": "...", "traceback": "..."}
```

---

## Implementation Checklist

When implementing a new framework integration, follow this checklist:

### Phase 1: Setup

- [ ] Create framework directory: `maif/integrations/{framework}/`
- [ ] Create `__init__.py` with exports
- [ ] Create main implementation file(s)

### Phase 2: Implementation

- [ ] Inherit from `BaseMAIFCallback` or use `MAIFProvenanceTracker` directly
- [ ] Implement framework's callback/hook interface
- [ ] Map framework events to `EventType` enum values
- [ ] Handle all event lifecycle (start, end, error)
- [ ] Serialize framework-specific data safely
- [ ] Implement `finalize()` method

### Phase 3: Testing

- [ ] Create `tests/integrations/test_{framework}.py`
- [ ] Write unit tests (no external API required)
- [ ] Write integration tests (marked with `@pytest.mark.integration`)
- [ ] Test error handling paths
- [ ] Test serialization of framework objects

### Phase 4: Documentation

- [ ] Create `docs/guide/integrations/{framework}.md`
- [ ] Write installation instructions
- [ ] Write quick start example
- [ ] Document all public classes and methods
- [ ] Add troubleshooting section

### Phase 5: Examples

- [ ] Create `examples/integrations/{framework}_quickstart.py`
- [ ] Make example self-contained and runnable
- [ ] Include expected output comments

---

## Framework-Specific Guides

### LangGraph Integration

**Status**: Implemented

**File**: `maif/integrations/langgraph/checkpointer.py`

**Primary Class**: `MAIFCheckpointer`

**Interface**: Implements `langgraph.checkpoint.base.BaseCheckpointSaver`

**Usage**:
```python
from langgraph.graph import StateGraph
from maif.integrations.langgraph import MAIFCheckpointer

checkpointer = MAIFCheckpointer("graph_state.maif")
app = graph.compile(checkpointer=checkpointer)
result = app.invoke(initial_state, config={"configurable": {"thread_id": "123"}})
```

**Methods to Implement**:
- `get(config)` / `aget(config)` - Retrieve checkpoint
- `put(config, checkpoint, metadata)` / `aput(...)` - Store checkpoint
- `list(config)` / `alist(config)` - List checkpoints

**Events Logged**:
- `STATE_CHECKPOINT` - When state is saved
- `STATE_RESTORE` - When state is restored

---

### LangChain Integration

**Status**: To implement

**Files**:
- `maif/integrations/langchain/callback.py` - MAIFCallbackHandler
- `maif/integrations/langchain/vectorstore.py` - MAIFVectorStore
- `maif/integrations/langchain/memory.py` - MAIFChatMessageHistory

**Primary Class**: `MAIFCallbackHandler`

**Interface**: Implements `langchain_core.callbacks.BaseCallbackHandler`

**Target Usage**:
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from maif.integrations.langchain import MAIFCallbackHandler

handler = MAIFCallbackHandler("session.maif")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
response = llm.invoke("Hello", config={"callbacks": [handler]})
handler.finalize()
```

**Methods to Implement**:

```python
class MAIFCallbackHandler(BaseCallbackHandler):
    # LLM callbacks
    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        """Log LLM invocation start."""
        
    def on_llm_end(self, response, *, run_id, **kwargs):
        """Log LLM response with generation info."""
        
    def on_llm_error(self, error, *, run_id, **kwargs):
        """Log LLM error."""
    
    # Chain callbacks
    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        """Log chain execution start."""
        
    def on_chain_end(self, outputs, *, run_id, **kwargs):
        """Log chain completion."""
        
    def on_chain_error(self, error, *, run_id, **kwargs):
        """Log chain error."""
    
    # Tool callbacks
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        """Log tool invocation start."""
        
    def on_tool_end(self, output, *, run_id, **kwargs):
        """Log tool result."""
        
    def on_tool_error(self, error, *, run_id, **kwargs):
        """Log tool error."""
    
    # Retriever callbacks
    def on_retriever_start(self, serialized, query, *, run_id, **kwargs):
        """Log retrieval query."""
        
    def on_retriever_end(self, documents, *, run_id, **kwargs):
        """Log retrieved documents."""
```

**Events Logged**:
- `LLM_START`, `LLM_END`, `LLM_ERROR`
- `CHAIN_START`, `CHAIN_END`, `CHAIN_ERROR`
- `TOOL_START`, `TOOL_END`, `TOOL_ERROR`
- `RETRIEVAL_START`, `RETRIEVAL_END`

**Additional Classes**:

1. `MAIFVectorStore` - Migrate from `framework_adapters.py`, improve:
   - Inherit from `langchain_core.vectorstores.VectorStore`
   - Add provenance logging for all operations
   - Implement `from_texts()` class method

2. `MAIFChatMessageHistory` - New implementation:
   - Inherit from `langchain_core.chat_history.BaseChatMessageHistory`
   - Store messages in MAIF with provenance
   - Implement `add_message()`, `clear()`, `messages` property

---

### CrewAI Integration

**Status**: To implement

**Files**:
- `maif/integrations/crewai/callback.py` - MAIFCrewCallback
- `maif/integrations/crewai/memory.py` - MAIFCrewMemory

**Primary Class**: `MAIFCrewCallback`

**Interface**: CrewAI task_callback and step_callback functions

**Target Usage**:
```python
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
```

**Methods to Implement**:

```python
class MAIFCrewCallback(BaseMAIFCallback):
    def on_task_complete(self, task_output):
        """Called when a task completes."""
        self.tracker.log_event(
            EventType.TASK_END,
            data={
                "task_description": task_output.description,
                "output": task_output.raw,
                "agent": task_output.agent,
            }
        )
    
    def on_step(self, step_output):
        """Called on each agent step."""
        self.tracker.log_event(
            EventType.AGENT_ACTION,
            data={
                "thought": step_output.thought,
                "action": step_output.action,
                "action_input": step_output.action_input,
                "observation": step_output.observation,
            }
        )
```

**Events Logged**:
- `TASK_START`, `TASK_END`, `TASK_ERROR`
- `AGENT_START`, `AGENT_END`, `AGENT_ACTION`

**Additional Classes**:

1. `MAIFCrewMemory` - Custom memory for CrewAI:
   - Long-term memory stored in MAIF
   - Searchable via MAIF's semantic capabilities

---

### Strands SDK Integration

**Status**: To implement

**Files**:
- `maif/integrations/strands/callback.py` - MAIFStrandsCallback

**Primary Class**: `MAIFStrandsCallback`

**Interface**: Strands `callback_handler` parameter

**Target Usage**:
```python
from strands import Agent
from maif.integrations.strands import MAIFStrandsCallback

callback = MAIFStrandsCallback("strands_session.maif")
agent = Agent(callback_handler=callback)
response = agent("What is the weather?")
callback.finalize()
```

**Methods to Implement**:

The Strands SDK callback handler should implement these lifecycle methods (verify against current Strands SDK documentation):

```python
class MAIFStrandsCallback:
    def on_agent_start(self, agent_name: str, input_text: str):
        """Called when agent starts processing."""
        
    def on_agent_end(self, agent_name: str, output: str):
        """Called when agent completes."""
        
    def on_tool_start(self, tool_name: str, tool_input: dict):
        """Called when a tool is invoked."""
        
    def on_tool_end(self, tool_name: str, tool_output: str):
        """Called when tool returns."""
        
    def on_model_start(self, model_id: str, messages: list):
        """Called when model inference starts."""
        
    def on_model_end(self, model_id: str, response: str):
        """Called when model inference completes."""
```

**Events Logged**:
- `AGENT_START`, `AGENT_END`, `AGENT_ERROR`
- `TOOL_START`, `TOOL_END`, `TOOL_ERROR`
- `LLM_START`, `LLM_END`, `LLM_ERROR`

**Note**: Verify the exact callback interface from the AWS Strands SDK documentation at:
https://github.com/strands-agents/sdk-python

---

## Testing Requirements

### Test File Structure

```python
# tests/integrations/test_{framework}.py

import pytest
import tempfile
import os

from maif.integrations.{framework} import {MainClass}


class Test{Framework}Integration:
    """Tests for {Framework} integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "test.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    # Unit tests (no API required)
    def test_initialization(self):
        """Test callback/handler initialization."""
        pass
    
    def test_event_logging(self):
        """Test event logging without external calls."""
        pass
    
    def test_serialization(self):
        """Test data serialization."""
        pass
    
    def test_artifact_creation(self):
        """Test MAIF artifact is created correctly."""
        pass
    
    # Integration tests (require API key)
    @pytest.mark.integration
    def test_full_workflow(self):
        """Test complete workflow with real API calls."""
        pass
```

### Shared Test Fixtures

```python
# tests/integrations/conftest.py

import pytest
import os

@pytest.fixture
def gemini_api_key():
    """Get Gemini API key from environment."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key

@pytest.fixture
def temp_artifact_path(tmp_path):
    """Create a temporary path for MAIF artifacts."""
    return str(tmp_path / "test.maif")
```

### Running Tests

```bash
# Run unit tests only
pytest tests/integrations/ -m "not integration"

# Run all tests including integration tests
GEMINI_API_KEY=your_key pytest tests/integrations/

# Run tests for specific framework
pytest tests/integrations/test_langgraph.py -v
```

---

## Documentation Standards

### Guide Structure

Each framework guide in `docs/guide/integrations/{framework}.md` should follow this structure:

```markdown
# {Framework} Integration

## Overview

Brief description of what this integration provides.

## Installation

\`\`\`bash
pip install maif[integrations]
# or
pip install maif {framework-package}
\`\`\`

## Quick Start

\`\`\`python
# Minimal working example
\`\`\`

## API Reference

### {MainClass}

Description of the main class.

#### Constructor

\`\`\`python
{MainClass}(artifact_path: str, agent_id: str = None)
\`\`\`

**Parameters:**
- `artifact_path`: Path to the MAIF artifact file
- `agent_id`: Optional identifier for the agent

#### Methods

##### method_name()

Description and usage.

## Advanced Usage

### Custom Configuration

### Error Handling

### Performance Considerations

## Troubleshooting

### Common Issues

### FAQ

## Examples

Link to example files.
```

### Code Documentation

All public classes and methods must have docstrings following Google style:

```python
def method_name(self, param1: str, param2: int = 10) -> Dict[str, Any]:
    """Brief description of what the method does.
    
    Longer description if needed, explaining behavior, side effects,
    or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2, defaults to 10
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        RuntimeError: When artifact is already finalized
        
    Example:
        >>> obj.method_name("test", param2=20)
        {"result": "success"}
    """
```

---

## Environment Variables

The following environment variables are used for testing:

| Variable | Purpose | Required For |
|----------|---------|--------------|
| `GEMINI_API_KEY` | Google Gemini API key | Integration tests |

**Setup**: Set the environment variable before running tests:
```bash
export GEMINI_API_KEY=your_api_key_here
```

---

## Dependencies

Add these to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
integrations = [
    "langgraph>=0.2.0",
    "langgraph-checkpoint>=1.0.0",
    "langchain-core>=0.3.0",
    "langchain-google-genai>=1.0.0",
    "crewai>=0.30.0",
    "strands-agents>=0.1.0",
]
```

Individual frameworks can also be installed separately with their own groups if needed.

---

## Questions?

If you have questions about implementing a specific integration:

1. Check the framework's official documentation
2. Look at the LangGraph implementation as a reference
3. Review the base classes in `_base.py` and `_utils.py`
4. Test incrementally with unit tests before integration tests

