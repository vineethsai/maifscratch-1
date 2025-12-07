# Real-World MAIF Examples

Production-ready examples demonstrating MAIF's cryptographic provenance tracking in real-world AI applications.

## Examples

| Example | Description | Use Case |
|---------|-------------|----------|
| [RAG Chatbot](./rag-chatbot/) | Document Q&A with citations | Customer support, knowledge base |
| [Customer Support](./customer-support-agent/) | Multi-agent triage system | Help desk, ticket routing |
| [Research Assistant](./research-assistant/) | Multi-step research workflow | Analysis, fact-checking |
| [Code Reviewer](./code-reviewer/) | Automated code analysis | CI/CD, security audits |

## Quick Start

```bash
# Install dependencies
pip install maif langgraph

# Run any example
cd rag-chatbot && python main.py
cd customer-support-agent && python main.py
cd research-assistant && python main.py
cd code-reviewer && python main.py
```

## Why MAIF?

All examples include **cryptographic provenance** - every agent action is:

1. **Signed** with Ed25519 (64-byte signatures)
2. **Hash-chained** for tamper detection
3. **Timestamped** for audit trails
4. **Verifiable** at any time

This is essential for:
- Compliance (SOC2, HIPAA, GDPR)
- Debugging and reproducibility
- Customer dispute resolution
- Agent performance tracking

## Architecture Pattern

All examples follow this pattern:

```python
from maif.integrations.langgraph import MAIFCheckpointer

# 1. Create checkpointer
checkpointer = MAIFCheckpointer("workflow.maif")

# 2. Compile graph with checkpointer
app = graph.compile(checkpointer=checkpointer)

# 3. Run workflow (all state changes logged)
result = app.invoke(state, config)

# 4. Finalize (seal the artifact)
checkpointer.finalize()

# 5. Verify integrity anytime
from maif import MAIFDecoder
decoder = MAIFDecoder("workflow.maif")
decoder.load()
is_valid, _ = decoder.verify_integrity()
```

## Customization

Each example can be customized:

- Replace mock LLM with OpenAI/Anthropic/local models
- Add real vector stores (ChromaDB, Pinecone)
- Integrate with your APIs and databases
- Extend with additional agents

