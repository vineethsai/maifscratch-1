# Examples

Real-world examples demonstrating MAIF capabilities.

::: tip More Examples on DeepWiki
For a comprehensive list of examples with auto-generated documentation, visit **[DeepWiki - Examples and Use Cases](https://deepwiki.com/vineethsai/maif/5-examples-and-use-cases)**.
:::

## Featured Example

### LangGraph Multi-Agent RAG System

Production-ready multi-agent research assistant with cryptographic provenance.

**Location**: `examples/langgraph/`

**Features**:
- Five specialized agents (Init, Retrieve, Synthesize, Fact-Check, Citation)
- Real ChromaDB vector search with 384-dim embeddings
- Gemini API for generation and verification
- LLM-based fact-checking with iterative refinement
- Complete audit trail in MAIF artifacts
- Interactive console interface
- Multi-turn conversation support

**Quick Start**:
```bash
cd examples/langgraph
echo "GEMINI_API_KEY=your_key" > .env
pip install -r requirements_enhanced.txt
python3 create_kb_enhanced.py
python3 demo_enhanced.py
```

**Documentation**: See [LangGraph RAG Guide](./langgraph-rag.md) for complete details.

---

### CrewAI Research Crew

Multi-agent research workflow with complete audit trails.

**Location**: Uses `maif.integrations.crewai`

**Features**:
- Two specialized agents (Researcher, Writer)
- Task and step-level provenance tracking
- Persistent agent memory with search
- Complete execution audit trail
- Error handling with logging

**Quick Start**:
```python
from crewai import Crew, Agent, Task
from maif.integrations.crewai import MAIFCrewCallback

callback = MAIFCrewCallback("session.maif")
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    task_callback=callback.on_task_complete,
    step_callback=callback.on_step,
)
result = crew.kickoff()
callback.finalize()
```

**Documentation**: See [CrewAI Research Guide](./crewai-research.md) for complete details.

---

## Available Examples

### Hello World
**[Get Started →](./hello-world.md)**

The simplest possible MAIF agent. Perfect for understanding the basics.

**What you'll learn:**
- Creating MAIF artifacts
- Adding text blocks
- Saving and loading
- Basic verification

**Time**: 5 minutes

---

### Multi-Agent System
**[View Example →](./multi-agent.md)**

Multiple agents collaborating through shared MAIF artifacts.

**What you'll learn:**
- Agent coordination
- Shared memory patterns
- Provenance tracking
- Multi-agent workflows

**Time**: 15 minutes

---

### Privacy & Security
**[View Example →](./privacy-demo.md)**

Privacy-preserving agent with encryption and anonymization.

**What you'll learn:**
- AES-GCM encryption
- Differential privacy
- Data anonymization
- Access control

**Time**: 10 minutes

---

### Streaming Data
**[View Example →](./streaming.md)**

High-throughput streaming with memory-mapped I/O.

**What you'll learn:**
- Streaming operations
- Memory-mapped I/O
- Performance optimization
- Large file handling

**Time**: 15 minutes

---

### Financial Agent
**[View Example →](./financial-agent.md)**

Privacy-compliant financial transaction analysis.

**What you'll learn:**
- Regulatory compliance
- Transaction analysis
- Audit trails
- Risk scoring

**Time**: 20 minutes

---

### Distributed Processing
**[View Example →](./distributed.md)**

Distributed agent systems with MAIF synchronization.

**What you'll learn:**
- Distributed coordination
- State synchronization
- Network protocols
- Fault tolerance

**Time**: 25 minutes

---

## Quick Start Examples

### Hello World Agent (30 seconds)

The simplest possible MAIF agent:

```python
from maif_api import create_maif

# Create agent with memory
memory = create_maif("hello-agent")

# Add content
memory.add_text("Hello, MAIF world!", title="Greeting")

# Save with cryptographic signing
memory.save("hello.maif", sign=True)

print("Your first AI agent memory is ready!")
```

### Privacy-Enabled Chat Agent (2 minutes)

A more realistic agent with memory and privacy:

```python
from maif_api import create_maif, load_maif
import os

class PrivateChatAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory_path = f"{agent_id}_memory.maif"
        
        # Load or create memory with privacy
        if os.path.exists(self.memory_path):
            self.memory = load_maif(self.memory_path)
        else:
            self.memory = create_maif(agent_id, enable_privacy=True)
    
    def chat(self, message: str, user_id: str) -> str:
        # Store message with privacy protection
        self.memory.add_text(
            f"User {user_id}: {message}",
            title="User Message",
            encrypt=True,
            anonymize=True  # Remove PII automatically
        )
        
        # Search for relevant context
        context = self.memory.search(message, top_k=3)
        
        # Generate response (integrate your LLM here)
        response = f"I understand you're asking about: {message}"
        
        # Store response
        self.memory.add_text(
            f"Agent: {response}",
            title="Agent Response",
            encrypt=True
        )
        
        return response
    
    def save(self):
        self.memory.save(self.memory_path, sign=True)

# Usage
agent = PrivateChatAgent("support-bot")
response = agent.chat("How do I reset my password?", "user123")
print(response)
agent.save()
```

## Example Categories

### By Experience Level

**Beginner:**
- [Hello World](./hello-world.md) - Your first MAIF agent
- [Privacy Demo](./privacy-demo.md) - Basic privacy features

**Intermediate:**
- [Multi-Agent](./multi-agent.md) - Agent coordination
- [Streaming](./streaming.md) - High-performance I/O
- [Financial Agent](./financial-agent.md) - Production patterns

**Advanced:**
- [LangGraph RAG](./langgraph-rag.md) - Complete multi-agent system
- [CrewAI Research](./crewai-research.md) - Multi-agent workflows with CrewAI
- [Distributed](./distributed.md) - Distributed systems

### By Use Case

**AI/ML Applications:**
- [LangGraph RAG](./langgraph-rag.md) - Research assistant with fact-checking
- [CrewAI Research](./crewai-research.md) - Research and documentation workflow
- [Multi-Agent](./multi-agent.md) - Collaborative agents

**Enterprise:**
- [Financial Agent](./financial-agent.md) - Regulatory compliance
- [Privacy Demo](./privacy-demo.md) - Data protection

**Performance:**
- [Streaming](./streaming.md) - High throughput
- [Distributed](./distributed.md) - Scale-out architecture

## Running the Examples

All examples follow the same pattern:

```bash
# 1. Navigate to repository root
cd /path/to/maif

# 2. Install dependencies (if needed)
pip install -e .

# 3. Run the example
python3 examples/<category>/<example_file>.py
```

For the LangGraph example:
```bash
cd examples/langgraph
pip install -r requirements_enhanced.txt
python3 demo_enhanced.py
```

## Example Structure

Each example includes:
- **Complete, runnable code**
- **Comprehensive error handling**
- **Performance optimizations**
- **Security best practices**
- **Testing and validation**
- **Detailed documentation**

## Contributing Examples

Have a great example to share? We welcome contributions!

1. Create your example in `examples/<category>/`
2. Add documentation in `docs/examples/`
3. Include README with usage instructions
4. Submit a pull request

## Support

- **Documentation**: See the [User Guide](../guide/)
- **API Reference**: Check the [API docs](../api/)
- **Issues**: Report problems on [GitHub](https://github.com/vineethsai/maif/issues)

---

*Every example is designed to be production-ready. Copy, modify, and deploy with confidence.*
