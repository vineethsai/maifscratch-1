---
layout: home

hero:
  name: "MAIF"
  text: "Multimodal Artifact Intelligence Framework"
  tagline: "Secure, verifiable memory for AI agents with built-in privacy, semantic understanding, and cryptographic provenance"
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/vineethsai/maif
    - theme: alt
      text: Explore on DeepWiki
      link: https://deepwiki.com/vineethsai/maif

features:
  - title: Persistent AI Memory
    details: Store agent memories, conversations, and knowledge in a unified, verifiable file format that persists across sessions
  - title: Privacy-by-Design
    details: AES-GCM and ChaCha20 encryption, PII anonymization, and differential privacy built into every operation
  - title: Cryptographic Provenance
    details: Immutable hash chains and digital signatures ensure complete audit trails and tamper detection
  - title: Semantic Understanding
    details: Built-in embeddings, cross-modal attention (ACAM), and semantic search for intelligent content retrieval
  - title: Multi-modal Native
    details: Seamlessly handle text, images, video, audio, embeddings, and structured data in a single artifact
  - title: High Performance
    details: Memory-mapped I/O, streaming support, and optimized compression for production workloads
---

<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #3c82f6 30%, #10b981);

  --vp-home-hero-image-background-image: linear-gradient(-45deg, #3c82f6 50%, #10b981 50%);
  --vp-home-hero-image-filter: blur(44px);
}

@media (min-width: 640px) {
  :root {
    --vp-home-hero-image-filter: blur(56px);
  }
}

@media (min-width: 960px) {
  :root {
    --vp-home-hero-image-filter: blur(68px);
  }
}
</style>

## What is MAIF?

**MAIF (Multimodal Artifact Intelligence Framework)** is a file format and SDK for building AI agents with **persistent, verifiable memory**. Unlike traditional AI systems where context is lost between sessions, MAIF artifacts store everything your agent knows in a cryptographically secure, tamper-evident format.

### The Problem with Traditional AI Memory

```
Traditional AI Agent:
  Session 1: User asks question → Agent responds → Memory lost
  Session 2: User returns → Agent has no context → Starts fresh
  
  No audit trail, no verification, no accountability
```

### The MAIF Solution

```
MAIF-Powered Agent:
  Session 1: User asks question → Agent responds → Saved to .maif file
  Session 2: User returns → Agent loads .maif → Full context restored
  
  + Every action cryptographically signed
  + Complete audit trail
  + Tamper detection built-in
```

## Quick Start

Get your first MAIF artifact running in 30 seconds:

```bash
# Clone and install
git clone https://github.com/vineethsai/maif.git
cd maif
pip install -e .
```

```python
from maif_api import create_maif, load_maif

# Create an artifact for your agent
memory = create_maif("my-agent")

# Add content with automatic integrity protection
memory.add_text("User prefers detailed explanations", title="Preference")
memory.add_text("Previous topic was machine learning", title="Context")

# Save with cryptographic signing
memory.save("agent_memory.maif", sign=True)

# Later: Load and verify
loaded = load_maif("agent_memory.maif")
assert loaded.verify_integrity()  # Tamper detection

# Search your agent's memory
results = memory.search("user preferences")
```

**That's it.** Your AI agent now has persistent, verifiable memory.

## Framework Integrations

MAIF integrates seamlessly with popular AI agent frameworks:

| Framework | Status | What You Get |
|-----------|--------|--------------|
| [**LangGraph**](/guide/integrations/langgraph) | Available | Checkpointer with cryptographic provenance |
| [**LangChain**](/guide/integrations/langchain) | Available | Callbacks, VectorStore, ChatMessageHistory |
| [**CrewAI**](/guide/integrations/crewai) | Available | Crew/Agent callbacks, persistent memory |
| [**Strands SDK**](/guide/integrations/strands) | Coming Soon | AWS Strands agent integration |

```python
# LangGraph - One line to add provenance
from maif.integrations.langgraph import MAIFCheckpointer
app = graph.compile(checkpointer=MAIFCheckpointer("session.maif"))

# LangChain - Track all LLM calls
from maif.integrations.langchain import MAIFCallbackHandler
llm = ChatOpenAI(callbacks=[MAIFCallbackHandler("llm.maif")])

# CrewAI - Full crew provenance
from maif.integrations.crewai import MAIFCrewCallback
crew = Crew(agents=[...], task_callback=callback.on_task_complete)
```

[View All Integrations →](/guide/integrations/)

## Real-World Use Case: Multi-Agent RAG System

MAIF powers production AI systems. See our complete example: a research assistant with 5 specialized agents, each action cryptographically logged.

```bash
cd examples/langgraph
pip install -r requirements_enhanced.txt
python3 demo_enhanced.py
```

**Features:**
- **5 Agents**: Init, Retrieve, Synthesize, Fact-Check, Citation
- **Real Vector Search**: ChromaDB with sentence-transformers
- **LLM Integration**: Gemini API for generation and verification
- **Complete Audit Trail**: Every query, retrieval, and response logged to MAIF

[View LangGraph Example →](/examples/langgraph-rag)

## Core Features

### 1. Persistent Memory

```python
from maif_api import create_maif

# Create memory for your agent
agent_memory = create_maif("support-bot")

# Store conversation context
agent_memory.add_text("User: How do I reset my password?")
agent_memory.add_text("Agent: Go to Settings > Security > Reset Password")

# Persist across sessions
agent_memory.save("support_bot.maif")
```

### 2. Privacy Protection

```python
from maif_api import create_maif

# Enable privacy features
secure_memory = create_maif("hipaa-agent", enable_privacy=True)

# Encrypt sensitive content
secure_memory.add_text(
    "Patient John Doe, SSN: 123-45-6789",
    title="Patient Record",
    encrypt=True,      # AES-GCM encryption
    anonymize=True     # Automatic PII removal
)

# Get privacy report
report = secure_memory.get_privacy_report()
```

### 3. Cryptographic Provenance

Every block is signed with Ed25519 and cryptographically linked:

```python
from maif_api import load_maif

# Load and verify (all signatures and hashes checked)
artifact = load_maif("important_data.maif")

# Check for tampering
if artifact.verify_integrity():
    print("Data is authentic and unmodified")
else:
    print("Data has been tampered with!")
```

### 4. Semantic Search

Find relevant content using semantic similarity:

```python
from maif_api import load_maif

# Load knowledge base
kb = load_maif("knowledge.maif")

# Semantic search
results = kb.search("machine learning best practices", top_k=5)

for result in results:
    print(f"Found: {result['text'][:100]}...")
```

### 5. Multi-modal Content

Store text, images, video, and embeddings together:

```python
from maif_api import create_maif

# Create multimodal artifact
artifact = create_maif("media-agent")

# Add different content types
artifact.add_text("Product description: High-quality headphones")
artifact.add_image("product_photo.jpg", title="Product Image")
artifact.add_multimodal({
    "text": "Review summary",
    "rating": 4.5,
    "features": ["wireless", "noise-cancelling"]
}, title="Product Review")

artifact.save("product_catalog.maif")
```

## Architecture

MAIF files are **self-contained** — everything in a single `.maif` file:

```
┌─────────────────────────────────────────────────────────┐
│                    MAIF Artifact (.maif)                │
├─────────────────────────────────────────────────────────┤
│  File Header: Ed25519 Public Key + Merkle Root          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Block 1 │→│ Block 2 │→│ Block 3 │→│ Block n │       │
│  │  Text   │ │  Image  │ │Embedding│ │Metadata │       │
│  │ sig+hash│ │ sig+hash│ │ sig+hash│ │ sig+hash│       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
│       │           │           │           │             │
│       └───────────┴───────────┴───────────┘             │
│                         ↓                               │
│       Hash Chain + Ed25519 Signatures (64 bytes each)   │
├─────────────────────────────────────────────────────────┤
│  Embedded Provenance Chain (complete audit trail)       │
└─────────────────────────────────────────────────────────┘
```

## Who Should Use MAIF?

| Use Case | Why MAIF? |
|----------|-----------|
| **AI Agent Developers** | Persistent memory that survives restarts |
| **Enterprise AI** | Audit trails for compliance (GDPR, HIPAA) |
| **Multi-Agent Systems** | Shared memory with provenance tracking |
| **Research** | Reproducible experiments with verification |
| **Healthcare AI** | Privacy-preserving patient data handling |
| **Financial AI** | Tamper-evident transaction logging |

## Getting Started

<div class="tip custom-block" style="padding-top: 8px">

**Next Steps:**
1. **[Installation Guide →](/guide/installation)** - Set up MAIF
2. **[Quick Start →](/guide/quick-start)** - 5-minute tutorial
3. **[Examples →](/examples/)** - Real-world use cases
4. **[API Reference →](/api/)** - Complete documentation

</div>

## Featured Example

The **LangGraph Multi-Agent RAG System** demonstrates MAIF in a production scenario:

- Research assistant with fact-checking
- 5 specialized agents working together
- Every action logged to MAIF artifacts
- Complete audit trail and provenance

[View the LangGraph Example →](/examples/langgraph-rag)

---

<div style="text-align: center; margin: 2rem 0;">

**MAIF: Trustworthy Memory for AI Agents**

*Open source. Production ready. Cryptographically secure.*

</div>
