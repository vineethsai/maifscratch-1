# LangGraph Multi-Agent RAG System

Complete production-ready multi-agent RAG system with cryptographic provenance.

## Introduction

This example demonstrates a sophisticated multi-agent architecture that combines:
- **LangGraph** for agent orchestration and state management
- **MAIF** for cryptographic provenance and audit trails
- **ChromaDB** for semantic similarity search
- **Gemini API** for natural language generation
- **Five specialized agents** working in coordination

The system answers questions by searching a local knowledge base, generating answers, verifying facts, and providing citations—all while maintaining a complete, tamper-evident audit trail.

## Architecture

### Agent Pipeline

The system consists of five specialized agents:

**1. Init Session Agent**
- Creates or loads MAIF session artifact
- Initializes session metadata
- Sets up cryptographic hash chain foundation

**2. Retriever Agent**
- Generates query embeddings (384 dimensions)
- Performs semantic similarity search in ChromaDB
- Ranks results by relevance score
- Logs retrieval events to MAIF

**3. Synthesizer Agent**
- Constructs prompts from retrieved context
- Calls Gemini API for answer generation
- Handles both initial generation and revisions
- Logs all responses with metadata

**4. Fact-Checker Agent**
- Extracts verifiable claims from answers
- Uses Gemini to verify each claim semantically
- Returns confidence scores and verification status
- Triggers revision if confidence below threshold

**5. Citation Agent**
- Maps claims to supporting sources
- Formats inline citations
- Adds references section
- Logs citation metadata

### Data Flow

```
User Question
    ↓
Init Session (creates MAIF artifact)
    ↓
Retrieve (ChromaDB search → MAIF log)
    ↓
Synthesize (Gemini generates → MAIF log)
    ↓
Fact-Check (LLM verifies → MAIF log)
    ↓
├─ Confidence ≥ 75% → Citation Agent → Done
└─ Confidence < 75% → Synthesize (revision) → Fact-Check (loop)
```

### Provenance Tracking

Every agent action creates a block in the session MAIF artifact:

```
Block 1: session_init
  └─ hash: a1b2c3...
      ↓ previous_hash
Block 2: user_message
  └─ hash: d4e5f6...
      ↓ previous_hash
Block 3: retrieval_event
  └─ hash: g7h8i9...
      ↓ previous_hash
Block 4: model_response
  └─ hash: j1k2l3...
      ↓ previous_hash
Block 5: verification
  └─ hash: m4n5o6...
```

Any tampering breaks the chain and is immediately detectable.

## Installation

### Prerequisites
- Python 3.9 or higher
- 4GB RAM minimum
- Internet connection for API calls and model downloads

### Setup Instructions

```bash
# Navigate to example directory
cd examples/langgraph

# Install dependencies
pip install chromadb sentence-transformers langgraph \
            langgraph-checkpoint-sqlite google-generativeai \
            python-dotenv tqdm requests

# Configure API key
echo "GEMINI_API_KEY=your_gemini_api_key" > .env

# Create knowledge base (one-time setup)
python3 create_kb_enhanced.py
```

The knowledge base creation process:
1. Loads 3 sample documents (14 chunks total)
2. Generates 384-dimensional embeddings using sentence-transformers
3. Indexes embeddings in ChromaDB for fast retrieval
4. Creates MAIF artifacts with embedded vectors
5. Takes approximately 30 seconds on modern hardware

## Usage

### Interactive Mode

```bash
python3 demo_enhanced.py
```

The interactive console provides eight options:

**1. Ask a Question**
- Enter any question about climate change
- Watch real-time agent execution
- Receive verified answer with citations
- All actions logged to MAIF

**2. View Session History**
- Read complete audit trail from MAIF artifact
- See all agent actions with timestamps
- Verify cryptographic integrity

**3. Inspect MAIF Artifact**
- View block count and types
- Examine hash chain linkage
- Check file size and metadata

**4. Multi-Agent Statistics**
- Agent execution counts
- Verification confidence metrics
- Block creation statistics

**5. Vector DB Statistics**
- Total chunks indexed
- Embedding dimensions
- Collection information

**6. Start New Session**
- Create fresh MAIF artifact
- Reset conversation context

**7. Multi-Turn Conversation**
- Ask follow-up questions
- Maintain conversation context
- All turns logged to same artifact

**8. Exit**
- Display session summary
- Show artifact location

### Programmatic Usage

```python
from graph_enhanced import create_enhanced_app

# Initialize application
app = create_enhanced_app()

# Configure session
config = {"configurable": {"thread_id": "session_123"}}

# Run query
result = app.invoke({
    "question": "What causes climate change?",
    "session_id": "session_123",
    "session_artifact_path": "",
    "kb_artifact_paths": {},
    "retrieved_chunks": [],
    "current_turn_block_ids": [],
    "verification_status": None,
    "needs_revision": False,
    "iteration_count": 0,
    "max_iterations": 3,
    "messages": []
}, config)

# Access results
answer = result['answer']
confidence = result['verification_results']['confidence']
citations = result['citations']
```

### Multi-Turn Conversations

```python
# Turn 1
result1 = app.invoke({
    "question": "What causes climate change?",
    "session_id": "session_123",
    # ... other fields
}, config)

# Turn 2 (same session)
result2 = app.invoke({
    "question": "How can we mitigate it?",
    "session_id": "session_123",
    "session_artifact_path": result1['session_artifact_path'],
    # ... other fields
}, config)

# Both turns logged to same MAIF artifact
```

## Configuration

### Environment Variables

Create `.env` file:
```bash
GEMINI_API_KEY=your_api_key_here
```

### Adjustable Parameters

**Fact-Checking Threshold** (`nodes/fact_check_enhanced.py`):
```python
if confidence >= 0.75:  # Adjust this value
    verification_status = 'verified'
```

**Maximum Iterations** (in initial state):
```python
"max_iterations": 3  # Number of revision cycles
```

**Retrieval Count** (`nodes/retrieve_enhanced.py`):
```python
chunks = vector_db.search(question, top_k=5)  # Number of chunks
```

**Embedding Model** (`vector_db.py`):
```python
embedding_model: str = "all-MiniLM-L6-v2"  # Change model
```

## Customization

### Adding Custom Documents

```python
from vector_db import get_vector_db
from maif_utils import KBManager

# Prepare chunks
chunks = [
    {
        "text": "Your document content...",
        "metadata": {"section": "intro", "page": 1}
    },
    # ... more chunks
]

# Add to vector database
vdb = get_vector_db()
vdb.add_documents("doc_id", chunks, document_metadata={
    "title": "Document Title",
    "author": "Author Name"
})

# Create MAIF artifact
kb_manager = KBManager()
kb_manager.create_kb_artifact("doc_id", chunks)
```

### Changing LLM Provider

Edit `nodes/synthesize.py`:

```python
# Replace Gemini with OpenAI
import openai

def call_llm_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Adding New Agents

1. Create node file:
```python
def summary_agent_node(state: RAGState) -> RAGState:
    summary = generate_summary(state['answer'])
    
    session_manager = SessionManager()
    block_id = session_manager.log_model_response(
        state['session_artifact_path'],
        summary,
        "summary_model",
        metadata={"node": "summary"}
    )
    
    state['summary'] = summary
    state['current_turn_block_ids'].append(block_id)
    return state
```

2. Add to graph:
```python
graph.add_node("summary", summary_agent_node)
graph.add_edge("cite", "summary")
```

## Performance

Measured on M1 Mac, 16GB RAM:

| Operation | Time | Details |
|-----------|------|---------|
| Query embedding | 200ms | Sentence-transformers, cached model |
| Vector search | 50ms | ChromaDB, 14 chunks |
| Gemini API call | 2-3s | Network dependent |
| Claim verification | 2s | Per claim, parallelized |
| MAIF block write | <10ms | Append operation |
| Hash verification | <1ms | SHA-256 computation |
| **Total pipeline** | 10-15s | Including 1-2 iterations |

Scales to:
- 100,000+ chunks in ChromaDB
- Multi-turn conversations (10+ turns)
- Concurrent sessions (thread-safe operations)

## Troubleshooting

### Vector Database Issues

**Empty database:**
```bash
python3 create_kb_enhanced.py
```

**Slow searches:**
- Check chunk count (should be <100K for optimal performance)
- Verify memory-mapped I/O is enabled
- Consider reducing embedding dimensions

### API Issues

**Rate limits:**
- Add delays between requests
- Reduce max_iterations
- Implement exponential backoff

**Authentication errors:**
- Verify GEMINI_API_KEY in .env
- Check API key permissions
- Ensure internet connectivity

### Import Errors

**Missing modules:**
```bash
pip install -r requirements_enhanced.txt
```

**Path issues:**
Run from correct directory or adjust sys.path

### MAIF Artifacts

**Integrity failures:**
- Do not manually edit .maif files
- Ensure atomic writes (no interruptions)
- Check disk space

**Large artifact sizes:**
- Implement session rotation
- Archive old sessions
- Use compression

## Advanced Topics

### Distributed Deployment

For high-traffic scenarios:

```python
# Use Redis for checkpointer
from langgraph.checkpoint.redis import RedisSaver

checkpointer = RedisSaver(redis_url="redis://localhost:6379")
app = graph.compile(checkpointer=checkpointer)
```

### Monitoring and Observability

```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

# Add custom metrics
from prometheus_client import Counter, Histogram

queries_total = Counter('rag_queries_total', 'Total queries processed')
query_duration = Histogram('rag_query_duration_seconds', 'Query processing time')
```

### Security Hardening

```python
# Enable encryption for MAIF artifacts
from maif.privacy import PrivacyEngine, EncryptionMode

privacy_engine = PrivacyEngine()
encoder = MAIFEncoder(
    enable_privacy=True,
    privacy_engine=privacy_engine
)

# Add encrypted blocks
encoder.add_text_block(
    sensitive_text,
    encryption_mode=EncryptionMode.AES_GCM
)
```

## Related Examples

- **Basic Examples**: `../basic/` - Simple MAIF operations
- **Security Examples**: `../security/` - Privacy and encryption
- **Advanced Examples**: `../advanced/` - Multi-agent patterns

## References

- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- ChromaDB Documentation: https://docs.trychroma.com/
- Sentence-Transformers: https://www.sbert.net/
- MAIF Core Documentation: `../../docs/`

## Citation

```bibtex
@software{maif_langgraph_rag_2024,
  title={LangGraph + MAIF Multi-Agent Research Assistant},
  author={MAIF Development Team},
  year={2024},
  url={https://github.com/vineethsai/maifscratch-1/tree/main/examples/langgraph},
  note={Production-ready multi-agent RAG with cryptographic provenance}
}
```

