# LangChain Integration

This guide covers integrating MAIF with LangChain for cryptographic provenance tracking of LLM calls, chains, and memory.

## Overview

The MAIF LangChain integration provides:

- **MAIFCallbackHandler**: Track all LLM/chain/tool calls
- **MAIFVectorStore**: Store embeddings with provenance
- **MAIFChatMessageHistory**: Chat memory with audit trail

## Installation

```bash
pip install maif langchain-core

# For OpenAI
pip install langchain-openai
```

## Quick Start

### Callback Handler

Track all LLM interactions:

```python
from langchain_openai import ChatOpenAI
from maif.integrations.langchain import MAIFCallbackHandler

# Create handler
handler = MAIFCallbackHandler("llm_session.maif")

# Use with any LangChain component
llm = ChatOpenAI(callbacks=[handler])
response = llm.invoke("What is the capital of France?")

# Finalize when done
handler.finalize()
```

### Vector Store

Store embeddings with provenance:

```python
from langchain_openai import OpenAIEmbeddings
from maif.integrations.langchain import MAIFVectorStore

embeddings = OpenAIEmbeddings()
vectorstore = MAIFVectorStore(
    embedding=embeddings,
    artifact_path="vectors.maif"
)

# Add documents
vectorstore.add_texts([
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
])

# Search
results = vectorstore.similarity_search("What is France's capital?")
print(results[0].page_content)

vectorstore.finalize()
```

### Chat Memory

Persistent chat history with provenance:

```python
from maif.integrations.langchain import MAIFChatMessageHistory

history = MAIFChatMessageHistory(
    session_id="user-123",
    artifact_path="chat_history.maif"
)

history.add_user_message("Hello!")
history.add_ai_message("Hi! How can I help you today?")

# Access messages
for msg in history.messages:
    print(f"{msg.type}: {msg.content}")

history.finalize()
```

## API Reference

### MAIFCallbackHandler

```python
class MAIFCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        artifact_path: str,
        agent_id: str = "langchain_handler",
    ):
        """
        Initialize the callback handler.
        
        Args:
            artifact_path: Path to the MAIF artifact file
            agent_id: Optional identifier for this handler
        """
```

**Tracked Events:**
- `on_llm_start` / `on_llm_end` - LLM calls
- `on_chat_model_start` - Chat model invocations
- `on_chain_start` / `on_chain_end` - Chain executions
- `on_tool_start` / `on_tool_end` - Tool invocations
- `on_agent_action` / `on_agent_finish` - Agent actions
- Error events for all of the above

### MAIFVectorStore

```python
class MAIFVectorStore(VectorStore):
    def __init__(
        self,
        embedding: Embeddings,
        artifact_path: str,
        agent_id: str = "maif_vectorstore",
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding: Embeddings model to use
            artifact_path: Path to the MAIF artifact
            agent_id: Identifier for this store
        """
```

**Methods:**
- `add_texts(texts, metadatas)` - Add documents
- `similarity_search(query, k)` - Search by text
- `similarity_search_with_score(query, k)` - Search with scores
- `similarity_search_by_vector(embedding, k)` - Search by vector

### MAIFChatMessageHistory

```python
class MAIFChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        artifact_path: str,
        agent_id: str = "chat_history",
    ):
        """
        Initialize chat history.
        
        Args:
            session_id: Unique session identifier
            artifact_path: Path to MAIF artifact
            agent_id: Identifier for this history
        """
```

**Methods:**
- `add_message(message)` - Add any message
- `add_user_message(text)` - Add human message
- `add_ai_message(text)` - Add AI message
- `clear()` - Clear history (logged to audit trail)
- `messages` - Property to access all messages

## Usage with LCEL

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from maif.integrations.langchain import MAIFCallbackHandler

handler = MAIFCallbackHandler("chain_session.maif")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

chain = prompt | ChatOpenAI() | StrOutputParser()

# All chain operations are tracked
result = chain.invoke(
    {"input": "Tell me a joke"},
    config={"callbacks": [handler]}
)

handler.finalize()
```

## Usage with RunnableWithMessageHistory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from maif.integrations.langchain import MAIFChatMessageHistory

def get_history(session_id: str):
    return MAIFChatMessageHistory(session_id, "history.maif")

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Session history is automatically managed with provenance
response = chain_with_history.invoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "user-123"}}
)
```

## Verifying Provenance

After any session, verify the audit trail:

```python
from maif import MAIFDecoder

decoder = MAIFDecoder("llm_session.maif")
decoder.load()

# Verify integrity
is_valid, errors = decoder.verify_integrity()
print(f"Integrity: {'VERIFIED' if is_valid else 'FAILED'}")

# Inspect events
for block in decoder.blocks:
    print(f"{block.metadata.get('type')}: {block.metadata.get('timestamp')}")
```

## Related

- [LangGraph Integration](./langgraph.md)
- [MAIF Overview](../getting-started.md)
- [LangChain Documentation](https://python.langchain.com/docs/)

