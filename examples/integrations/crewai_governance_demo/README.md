# MAIF + CrewAI: Enterprise AI Governance Demo

Interactive demonstration of MAIF's enterprise-grade features for CrewAI multi-agent workflows with cryptographic provenance tracking.

## Overview

This demo showcases how MAIF provides:

- **Cryptographic Provenance**: Every agent action is Ed25519 signed and hash-chained
- **Task & Step Tracking**: Complete audit trail of agent reasoning (thought -> action -> observation)
- **Tamper Detection**: Any modification breaks the cryptographic chain
- **Persistent Memory**: Agent memory with provenance across sessions
- **Compliance Reporting**: Generate audit reports from artifacts

## Requirements

```bash
pip install maif[integrations] crewai
```

Note: CrewAI requires Python 3.10 or higher.

## Quick Start

```bash
cd examples/integrations/crewai_governance_demo
python main.py
```

## Features

### 1. Run Research Crew

Execute a multi-agent research workflow with full provenance tracking:
- Researcher agent gathers information
- Writer agent creates documentation
- Every step logged to MAIF artifact

All actions are cryptographically signed and hash-chained.

### 2. Resume Previous Session

Load and inspect previous crew executions from MAIF artifacts. The system:
- Verifies artifact integrity before loading
- Shows session statistics and event counts
- Allows detailed inspection

### 3. Inspect Artifact Provenance

Deep dive into the audit trail with multiple views:
- **Timeline View**: Chronological event sequence
- **Block Details**: Inspect individual blocks and their data
- **Event Type Summary**: Events grouped by type
- **Agent Activity**: Events grouped by agent
- **Task & Step Analysis**: CrewAI-specific events
- **Hash Chain View**: Visualize cryptographic linking

### 4. Security Verification

Run comprehensive security checks:
- File existence and load tests
- Header validation
- Hash chain verification
- Ed25519 signature verification
- Block structure validation

### 5. Tamper Detection Demo

See what happens when artifact data is modified:
1. Creates a test artifact with sample data
2. Verifies original integrity
3. Simulates tampering (byte modification)
4. Shows how MAIF detects the tampering

### 6. Generate Compliance Report

Export audit reports in multiple formats:
- Summary Report (Markdown)
- Detailed Audit Log (JSON)
- Timeline Export (CSV)
- Task Summary Report

### 7. Agent Memory Demo

Demonstrate persistent agent memory:
- Store memories with agent, tags, and importance
- Search memories by content
- Filter by agent or tags
- View importance-ranked memories

## Directory Structure

```
crewai_governance_demo/
    main.py                 # Entry point
    demo_app.py            # Main application and menu
    crew_runner.py         # CrewAI execution with MAIF
    provenance_inspector.py # Artifact inspection
    security_verifier.py    # Security verification
    tamper_demo.py         # Tamper detection demo
    report_generator.py    # Report generation
    memory_demo.py         # Memory persistence demo
    README.md              # This file
    sessions/              # Session artifacts
    reports/               # Generated reports
```

## Architecture

```
User Input (Topic)
    |
    v
+------------------+
|   Crew Runner    |  <-- Initialize MAIF callback
+--------+---------+
         |
    +----+----+
    v         v
+----------+ +---------+
|Researcher| | Writer  |  <-- Agents with specific roles
|  Agent   | |  Agent  |
+----+-----+ +----+----+
     |            |
     +-----+------+
           |
           v
    [Step Callbacks]  <-- Each step logged to MAIF
           |
           v
    [Task Callbacks]  <-- Each task logged to MAIF
           |
           v
+------------------+
|  MAIF Artifact   |  <-- Cryptographically sealed
+------------------+
```

All agent actions are logged via the MAIF callback system.

## Key MAIF Concepts Demonstrated

### Hash Chain

Each block contains a hash of the previous block:

```
Block 1 (session_start)
    |
    | Hash: a3f2b1c8...
    v
Block 2 (agent_start)
    |
    | Hash: 7d4e9f2a...
    v
Block 3 (agent_action)
    |
    | Hash: c8b4f1e2...
    v
Block 4 (task_end)
    ...
```

Modifying any block invalidates all subsequent blocks.

### CrewAI Event Types

The callback logs these event types:

| Event Type | Description |
|------------|-------------|
| `session_start` | Callback handler initialized |
| `agent_start` | Crew kickoff with agents/tasks |
| `agent_action` | Agent reasoning step (thought/action/observation) |
| `tool_end` | Tool invocation result |
| `task_end` | Task completion with output |
| `agent_end` | Crew completion with summary |
| `agent_error` | Error during execution |
| `memory_save` | Memory stored via MAIFCrewMemory |
| `session_end` | Callback handler finalized |

### Persistent Memory

MAIFCrewMemory provides:

```python
from maif.integrations.crewai import MAIFCrewMemory

memory = MAIFCrewMemory("agent_memory.maif")

# Store
memory.save(
    content="User prefers detailed responses",
    agent="assistant",
    tags=["preference"],
    importance=0.8,
)

# Search
results = memory.search("user preferences")

# Persist
memory.finalize()
```

## Example Session

```
$ python main.py

================================================================================
    MAIF + CrewAI: Enterprise AI Governance Demo
================================================================================

MAIN MENU
--------------------------------------------------------------------------------
[1] Run Research Crew              - Multi-agent workflow with provenance
[2] Resume Previous Session        - Load from existing MAIF artifact
[3] Inspect Artifact Provenance    - Deep dive into audit trail
[4] Security Verification          - Verify signatures and integrity
[5] Tamper Detection Demo          - See what happens when data is modified
[6] Generate Compliance Report     - Export audit report
[7] Agent Memory Demo              - Persistent memory with provenance
[8] Exit

Choice: 1

Enter session name: security-research
Enter research topic: AI agent security best practices

Session: security-research
Artifact: sessions/security-research.maif
Topic: AI agent security best practices
--------------------------------------------------------------------------------

Creating agents...
  Created: Senior Research Analyst
  Created: Technical Writer

Creating tasks...
  Created: Research Task
  Created: Documentation Task

Assembling crew...
  Crew assembled with 2 agents and 2 tasks

Starting crew execution...
------------------------------------------------------------
[RESEARCHER] Starting research...
  Step: search
  Step: analyze
  Step: Final Answer
  Task completed: Research

[WRITER] Starting documentation...
  Step: outline
  Step: write
  Step: Final Answer
  Task completed: Documentation

------------------------------------------------------------

Crew execution completed successfully.

Session saved successfully.
  Tasks completed: 2
  Steps executed: 6
  Total events: 12
  Artifact: sessions/security-research.maif
  Integrity: VERIFIED
```

## Running Without LLM API Keys

The demo includes a "demo mode" that simulates crew execution without requiring LLM API keys. This allows you to explore all the provenance and reporting features without external dependencies.

## Related Documentation

- [CrewAI Integration Guide](../../../docs/guide/integrations/crewai.md)
- [LangGraph Governance Demo](../langgraph_governance_demo/)
- [MAIF Security Model](../../../docs/guide/security-model.md)

