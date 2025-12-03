# MAIF Examples

## 🚀 Featured: LangGraph Multi-Agent RAG

**Location:** `langgraph/`

Production-ready multi-agent system with:
- Real ChromaDB vector search
- Gemini API integration
- LLM fact-checking
- MAIF cryptographic provenance

```bash
cd langgraph
python3 demo_enhanced.py
```

See `langgraph/README.md` for details.

---

## 📁 Example Categories

### `basic/` - Getting Started
- `basic_usage.py` - Simple MAIF operations
- `simple_api_demo.py` - High-level API

### `security/` - Privacy & Encryption
- `privacy_demo.py` - Privacy features
- `classified_api_simple_demo.py` - Classified data handling
- `classified_security_demo.py` - Security controls

### `aws/` - AWS Integration
- `bedrock_swarm_demo.py` - Multi-model swarm
- `s3_block_storage_demo.py` - S3 storage
- `kms_integration.py` - Key management
- [14 other AWS demos]

### `advanced/` - Advanced Features
- `multi_agent_consortium_demo.py` - Multi-agent collaboration
- `lifecycle_management_demo.py` - Artifact lifecycle
- `novel_algorithms_demo.py` - ACAM, HSC, CSB
- [7 other advanced demos]

---

## Quick Start

1. **Basic usage:**
   ```bash
   python3 basic/simple_api_demo.py
   ```

2. **Multi-agent RAG:**
   ```bash
   cd langgraph
   python3 create_kb_enhanced.py
   python3 demo_enhanced.py
   ```

3. **AWS integration:**
   ```bash
   python3 aws/bedrock_swarm_demo.py
   ```

