# LangGraph + MAIF Multi-Agent Research Assistant

Production-ready multi-agent RAG system combining LangGraph orchestration with MAIF cryptographic provenance.

## Overview

This system demonstrates a complete multi-agent architecture where:
- **LangGraph** manages agent coordination and control flow
- **MAIF** provides cryptographic provenance for all operations
- **ChromaDB** enables semantic similarity search
- **Gemini API** powers answer generation and fact-checking
- **Five specialized agents** collaborate to produce verified, cited answers

### Architecture

```
User Question
    ↓
[Init Session] → Creates MAIF artifact
    ↓
[Retriever] → ChromaDB semantic search (384-dim embeddings)
    ↓
[Synthesizer] → Gemini generates answer
    ↓
[Fact-Checker] → LLM verifies claims
    ↓ (if confidence < 75%)
[Synthesizer] → Revises answer (iterative refinement)
    ↓
[Citation Agent] → Adds source attribution
    ↓
Final Answer + Complete Audit Trail
```

All agent actions are logged to a MAIF artifact with cryptographic hash chains, creating a tamper-evident audit trail.

## Features

### Core Capabilities
- **Semantic Search**: ChromaDB with sentence-transformers embeddings (384 dimensions)
- **Answer Generation**: Gemini 2.0 Flash API integration
- **Fact Verification**: LLM-based claim verification with confidence scores
- **Iterative Refinement**: Automatic answer revision when confidence is low
- **Citation Management**: Automatic source attribution and reference formatting
- **Multi-Turn Conversations**: Full conversation history with context maintenance

### Provenance & Audit
- **Cryptographic Hash Chains**: Every block linked to previous via SHA-256
- **Complete Audit Trail**: All agent actions logged with timestamps
- **Tamper Detection**: Any modification breaks the hash chain
- **Session Artifacts**: Complete conversation history in MAIF format
- **KB Artifacts**: Document chunks with embeddings stored in MAIF

### Production Features
- **Thread-Safe**: Concurrent access to MAIF artifacts
- **Error Handling**: Graceful fallbacks for API failures
- **Rate Limiting**: Controlled Gemini API usage
- **Progress Indicators**: Real-time feedback during operations
- **Environment-Based Config**: API keys from .env files

## Installation

### Requirements
- Python 3.9+
- 4GB RAM minimum
- Internet connection (for Gemini API and embedding model)

### Setup

```bash
cd examples/langgraph

# 1. Install dependencies
pip install -r requirements_enhanced.txt

# Alternatively, install individually:
pip install chromadb sentence-transformers langgraph \
            langgraph-checkpoint-sqlite google-generativeai \
            python-dotenv tqdm requests

# 2. Configure API key
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env

# 3. Create knowledge base
python3 create_kb_enhanced.py
```

The KB creation script will:
- Generate real embeddings for 14 document chunks
- Index them in ChromaDB for semantic search
- Create MAIF artifacts with embedded vectors
- Takes approximately 30 seconds

## Usage

### Interactive Demo

```bash
python3 demo_enhanced.py
```

This launches an interactive console with eight menu options:

1. **Ask a Question** - Full RAG pipeline with fact-checking
2. **View Session History** - Complete audit trail from MAIF
3. **Inspect MAIF Artifact** - Cryptographic provenance details
4. **Multi-Agent Statistics** - Agent activity metrics
5. **Vector DB Statistics** - ChromaDB database info
6. **Start New Session** - Fresh MAIF artifact
7. **Multi-Turn Conversation** - Conversational interface
8. **Exit** - Session summary

### Example Session

```
Question: What are the main causes of climate change?

Retriever Agent:
  - Generates 384-dim query embedding
  - Searches ChromaDB: 5 chunks found
  - Best match: 0.776 similarity
  - Logs retrieval event to MAIF

Synthesizer Agent:
  - Builds prompt with retrieved context
  - Calls Gemini API for answer generation
  - Logs response to MAIF (969 characters)

Fact-Checker Agent:
  - Extracts 7 verifiable claims
  - Uses Gemini to verify each claim semantically
  - Iteration 1: 71.4% confidence → triggers revision
  - Logs verification to MAIF

Synthesizer Agent (Revision):
  - Revises based on feedback
  - Iteration 2: 83.3% confidence → accepted
  - Logs revised answer to MAIF

Citation Agent:
  - Maps claims to source documents
  - Adds reference section
  - Logs citations to MAIF

Result: Answer with 5 citations + complete provenance trail
```

### Programmatic Usage

```python
from graph_enhanced import create_enhanced_app

# Create application
app = create_enhanced_app()

# Run query
result = app.invoke({
    "question": "What causes climate change?",
    "session_id": "user_123",
    "session_artifact_path": "",
    "kb_artifact_paths": {},
    "retrieved_chunks": [],
    "current_turn_block_ids": [],
    "verification_status": None,
    "needs_revision": False,
    "iteration_count": 0,
    "max_iterations": 3,
    "messages": []
}, config={"configurable": {"thread_id": "user_123"}})

print(result['answer'])
```

## Components

### Core Modules

#### State Management (`state.py`)
Defines `RAGState` TypedDict with:
- User interaction fields (question, answer, messages)
- MAIF persistence layer (session paths, KB paths)
- Working memory (retrieved chunks, current blocks)
- Verification state (results, status, iteration count)

#### MAIF Utilities (`maif_utils.py`)
- `SessionManager`: Create and log to session artifacts
- `KBManager`: Create KB artifacts with document chunks
- Logging methods for all agent actions

#### Vector Database (`vector_db.py`)
- ChromaDB persistent client
- Sentence-transformers integration (all-MiniLM-L6-v2)
- Embedding generation and indexing
- Semantic similarity search

#### Enhanced Fact-Checking (`enhanced_fact_check.py`)
- LLM-based claim verification using Gemini
- Parallel verification (2 claims at once)
- Confidence scoring per claim
- Fallback to keyword matching

### Agent Nodes

#### Init Session (`nodes/init_session.py`)
Creates or loads MAIF session artifact. Initializes session metadata and state fields.

#### Retriever (`nodes/retrieve_enhanced.py`)
- Generates query embeddings using sentence-transformers
- Performs semantic search in ChromaDB
- Logs retrieval events with document IDs and similarity scores
- Returns top-k most relevant chunks

#### Synthesizer (`nodes/synthesize.py`)
- Builds prompts from retrieved chunks
- Calls Gemini API for answer generation
- Handles both initial generation and revision
- Logs model responses with iteration metadata

#### Fact-Checker (`nodes/fact_check_enhanced.py`)
- Extracts verifiable claims from generated answers
- Uses Gemini to verify each claim against sources
- Returns verdict (SUPPORTED/CONTRADICTED/UNVERIFIED) per claim
- Triggers revision if confidence below threshold (75%)

#### Citation Agent (`nodes/cite.py`)
- Maps verified claims to supporting source chunks
- Formats inline citations
- Adds references section
- Logs citation metadata

### Graph Construction (`graph_enhanced.py`)

LangGraph orchestrates agent execution with:
- Sequential node execution
- Conditional routing (fact-check → revise or cite)
- State management across nodes
- SqliteSaver checkpointer for resumability

## Configuration

### Environment Variables

Create `.env` file:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Adjustable Parameters

#### Fact-Checking Threshold
Edit `nodes/fact_check_enhanced.py` (line 67):
```python
if confidence >= 0.75:  # Adjust verification threshold
    verification_status = 'verified'
```

#### Maximum Iterations
In initial state or edit node:
```python
"max_iterations": 3  # Maximum revision cycles
```

#### Retrieval Count
Edit `nodes/retrieve_enhanced.py`:
```python
chunks = vector_db.search(question, top_k=5)  # Number of chunks
```

#### Embedding Model
Edit `vector_db.py` (line 18):
```python
embedding_model: str = "all-MiniLM-L6-v2"  # Change model
```

## Data Storage

### Directory Structure
```
data/
├── sessions/              # MAIF session artifacts
│   ├── {session_id}.maif
│   └── {session_id}_manifest.json
├── kb/                    # Knowledge base artifacts
│   ├── doc_001.maif
│   ├── doc_001_manifest.json
│   └── ...
├── chroma_db/            # ChromaDB persistent storage
└── checkpoints_enhanced.db  # LangGraph checkpoints
```

### Session Artifact Structure

Each conversation creates a session MAIF file containing cryptographically-linked blocks:

```
Block 1: Session initialization (BDAT type)
Block 2: User message (TEXT type)
Block 3: Retrieval event (BDAT type)
  - Query text
  - Retrieved document IDs
  - Similarity scores
  - Method: chromadb_semantic_search
Block 4: Model response (TEXT type)
  - Full answer text
  - Model: gemini-2.0-flash
  - Iteration number
Block 5: Verification results (BDAT type)
  - Claims extracted
  - Verified claims
  - Confidence score
  - Method: gemini_llm
Block 6: Citations (BDAT type)
  - Claim to source mappings
  - Confidence per citation
```

All blocks linked via `previous_hash` field for tamper detection.

### Knowledge Base Artifacts

Each document is stored as a separate MAIF artifact:

```
Block 1: Document metadata (BDAT)
Block 2: Chunk 0 text (TEXT)
Block 3: Chunk 0 embedding (BDAT, 384 floats)
Block 4: Chunk 1 text (TEXT)
Block 5: Chunk 1 embedding (BDAT)
...
```

## Customization

### Adding Custom Documents

```python
from vector_db import get_vector_db
from maif_utils import KBManager

# Prepare document chunks
chunks = [
    {
        "text": "Your document content here...",
        "metadata": {"section": "introduction", "page": 1}
    },
    # ... more chunks
]

# Index in vector database
vdb = get_vector_db()
vdb.add_documents("your_doc_id", chunks, document_metadata={
    "title": "Your Document Title",
    "author": "Author Name",
    "source": "Document Source"
})

# Create MAIF artifact
kb_manager = KBManager()
kb_path = kb_manager.create_kb_artifact(
    doc_id="your_doc_id",
    chunks=chunks,
    document_metadata={"title": "Your Document Title"}
)

# Now searchable in the RAG system
```

### Changing the LLM

Replace Gemini with another provider by editing `nodes/synthesize.py`:

```python
# For OpenAI
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

# For Anthropic Claude
import anthropic
response = anthropic.Anthropic().messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": prompt}]
)
```

### Adding New Agent Nodes

1. Create node file in `nodes/`:
```python
def new_agent_node(state: RAGState) -> RAGState:
    # Perform agent-specific logic
    result = process_data(state)
    
    # Log to MAIF
    session_manager = SessionManager()
    block_id = session_manager.log_custom_event(
        state['session_artifact_path'],
        result,
        metadata={"node": "new_agent"}
    )
    
    state['current_turn_block_ids'].append(block_id)
    return state
```

2. Add to graph in `graph_enhanced.py`:
```python
graph.add_node("new_agent", new_agent_node)
graph.add_edge("previous_node", "new_agent")
```

## Testing

### Run Test Suite
```bash
python3 test_all_features.py
```

Tests validate:
- Vector database functionality
- Gemini API integration
- LLM fact-checking
- MAIF provenance logging
- Multi-agent pipeline
- Multi-turn conversations

### Manual Testing
```bash
# Test vector DB
python3 -c "
from vector_db import get_vector_db
vdb = get_vector_db()
print(vdb.get_stats())
"

# Test MAIF operations
python3 -c "
from maif_utils import SessionManager
sm = SessionManager()
path = sm.create_session('test')
print(f'Created: {path}')
"
```

## Performance Characteristics

Based on local testing (M1 Mac, 16GB RAM):

| Operation | Time | Notes |
|-----------|------|-------|
| Embedding generation | 200ms per text | Cached model, batch processing |
| Vector search | 50ms | ChromaDB with 14 chunks |
| Gemini API call | 2-3s | Network latency dependent |
| LLM fact-check | 2s per claim | Parallelized (2 concurrent) |
| MAIF block write | <10ms | Append-only operation |
| Hash verification | <1ms | Local computation |
| **Total per question** | 10-15s | Includes 1-2 iterations |

Scales to:
- 100K+ chunks in ChromaDB
- Multi-turn conversations (10+ turns tested)
- Concurrent sessions (thread-safe)

## Troubleshooting

### "GEMINI_API_KEY not found"
Create `.env` file in `examples/langgraph/`:
```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

### "Vector DB is empty"
Run KB creation:
```bash
python3 create_kb_enhanced.py
```

### "ModuleNotFoundError: langgraph"
Install dependencies:
```bash
pip install -r requirements_enhanced.txt
```

### "No module named 'examples.langgraph'"
Ensure you're running from project root or the examples/langgraph/ directory.

### Poor Search Results
The default KB contains climate change documents. Questions outside this domain will return low similarity scores. Add your own documents using the customization guide above.

### API Rate Limits
Gemini API has rate limits. If you hit them:
- Add delays between questions
- Reduce max_iterations
- Limit concurrent fact-checking

## Technical Details

### State Management
- **LangGraph State**: In-memory working data (fast access)
- **LangGraph Checkpoint**: SQLite persistence (resumability)
- **MAIF Artifacts**: Durable memory (cryptographic provenance)

This three-tier approach provides both performance and auditability.

### Fact-Checking Strategy
The system uses a two-phase verification:

1. **Extraction**: Parse answer into verifiable claims using regex
2. **Verification**: For each claim:
   - Construct verification prompt with claim and sources
   - Call Gemini API for semantic understanding
   - Parse JSON response (SUPPORTED/CONTRADICTED/UNVERIFIED)
   - Calculate confidence score
   - Aggregate results

If confidence < 75%, the synthesizer revises the answer with specific feedback about unverified claims.

### MAIF Block Types
- `TEXT`: User messages, model responses
- `BDAT`: Binary data (JSON metadata, verification results, citations)
- `EMBEDDING`: Vector embeddings (not used in sessions)

Each block has:
- `block_type`: 4-character identifier
- `data`: Raw bytes
- `metadata`: Dictionary with type, timestamp, etc.
- `previous_hash`: SHA-256 of previous block (chain integrity)

## File Reference

### Main Scripts
- `demo_enhanced.py` - Interactive demo with all features (primary entry point)
- `create_kb_enhanced.py` - Knowledge base creation with embeddings
- `demo_hybrid.py` - Version with web search fallback
- `test_all_features.py` - Comprehensive test suite

### Core Modules
- `state.py` - RAGState TypedDict definition
- `graph_enhanced.py` - LangGraph construction with conditional routing
- `graph_hybrid.py` - Hybrid graph with web search
- `maif_utils.py` - MAIF logging utilities (SessionManager, KBManager)
- `vector_db.py` - ChromaDB integration with sentence-transformers
- `enhanced_fact_check.py` - LLM-based verification logic
- `web_search_agent.py` - DuckDuckGo fallback search

### Agent Nodes (`nodes/`)
- `init_session.py` - Session artifact initialization
- `retrieve_enhanced.py` - ChromaDB semantic search
- `synthesize.py` - Gemini answer generation
- `fact_check_enhanced.py` - LLM claim verification
- `cite.py` - Citation formatting and attribution
- `retrieve_hybrid.py` - Hybrid local + web search

### Configuration
- `.env` - API keys and environment variables
- `requirements_enhanced.txt` - Production dependencies
- `.gitignore` - Excludes generated artifacts and databases

## Design Decisions

### Why LangGraph?
- Industry-standard orchestration framework
- Built-in checkpointing for resumability
- Conditional routing for complex workflows
- Easy to extend with new agents

### Why MAIF?
- Cryptographic provenance (tamper-evident audit trails)
- Multi-agent coordination via shared artifacts
- Append-only architecture (immutable history)
- Built-in security and privacy features

### Why ChromaDB?
- Persistent vector storage
- Fast similarity search
- Simple API
- No external server required

### Why Gemini?
- Competitive performance and cost
- Straightforward REST API
- Good balance of quality and speed
- Alternative: easily swap for OpenAI/Claude/etc.

## Known Limitations

1. **Local KB Required**: System searches local knowledge base, not the web
   - Workaround: Use `demo_hybrid.py` for web search fallback
   - Or add your own documents to KB

2. **Fact-Checking Accuracy**: LLM-based verification ~85-90% accurate
   - Depends on source quality and claim clarity
   - May require threshold tuning for specific domains

3. **API Dependency**: Requires Gemini API access
   - Rate limits apply
   - Network latency affects response time
   - Fallback mechanisms in place

4. **Embedding Model Size**: Sentence-transformers model ~90MB
   - First-time download required
   - Cached after initial load

## Extending the System

### Adding Compliance Agent

```python
def compliance_agent_node(state: RAGState) -> RAGState:
    """Check answer for compliance violations."""
    answer = state['answer']
    
    # Check for PII, sensitive data, etc.
    violations = check_compliance(answer)
    
    # Log to MAIF
    session_manager = SessionManager()
    block_id = session_manager.log_verification(
        state['session_artifact_path'],
        {"compliance_check": violations},
        metadata={"node": "compliance", "status": "pass" if not violations else "fail"}
    )
    
    state['current_turn_block_ids'].append(block_id)
    return state

# Add to graph
graph.add_node("compliance", compliance_agent_node)
graph.add_edge("cite", "compliance")
```

### Adding Retrieval from Multiple Sources

```python
def multi_source_retrieve(state: RAGState) -> RAGState:
    """Retrieve from multiple vector DBs."""
    question = state['question']
    
    # Search multiple collections
    vdb = get_vector_db()
    results_a = vdb.collection_a.query([question], n_results=3)
    results_b = vdb.collection_b.query([question], n_results=3)
    
    # Merge and re-rank
    all_chunks = merge_results(results_a, results_b)
    
    state['retrieved_chunks'] = all_chunks
    return state
```

## Production Deployment

### Containerization

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_enhanced.txt .
RUN pip install --no-cache-dir -r requirements_enhanced.txt

COPY . .
EXPOSE 8000

CMD ["python3", "demo_enhanced.py"]
```

### Scaling Considerations

- **Vector DB**: ChromaDB can handle 100K+ chunks efficiently
- **MAIF Artifacts**: Keep sessions under 10MB for optimal performance
- **Concurrent Access**: LangGraph checkpointer is thread-safe
- **API Limits**: Implement request queuing for high-traffic scenarios

### Monitoring

Add custom logging:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('rag_system.log')]
)
```

## Contributing

To extend this system:
1. Fork the repository
2. Create feature branch
3. Add your enhancements
4. Test with `test_all_features.py`
5. Submit pull request

## License

Part of the MAIF project. See main repository LICENSE file.

## Citation

If you use this system in your research:

```bibtex
@software{maif_langgraph_rag,
  title={LangGraph + MAIF Multi-Agent Research Assistant},
  author={MAIF Development Team},
  year={2024},
  url={https://github.com/vineethsai/maifscratch-1/tree/main/examples/langgraph}
}
```

## Support

- GitHub Issues: https://github.com/vineethsai/maifscratch-1/issues
- Documentation: See README files in parent directories
- Examples: All example files include inline documentation
