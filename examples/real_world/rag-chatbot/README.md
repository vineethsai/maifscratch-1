# RAG Chatbot with MAIF Provenance

A production-ready RAG (Retrieval-Augmented Generation) chatbot with full cryptographic audit trail.

## Features

- Document retrieval with citation tracking
- Multi-turn conversation with state persistence
- Cryptographic provenance (Ed25519 signatures)
- Tamper-evident audit trail
- Works with or without OpenAI API key

## Quick Start

```bash
# Install dependencies
pip install maif langgraph

# Optional: For real LLM responses
pip install langchain-openai
export OPENAI_API_KEY=your_key

# Run the chatbot
python main.py
```

## Usage

```
RAG Chatbot with MAIF Provenance
============================================================

Type 'quit' to exit, 'history' to see conversation
Type 'audit' to see provenance summary

You: What are your pricing plans?

[RETRIEVE] Found 3 relevant documents
[GENERATE] Response generated with 3 citations
