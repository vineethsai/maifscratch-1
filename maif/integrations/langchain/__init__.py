"""
MAIF LangChain Integration

Provides MAIF-backed components for LangChain with cryptographic provenance:
- MAIFCallbackHandler: Track all LLM/chain calls
- MAIFVectorStore: Store embeddings with provenance
- MAIFChatMessageHistory: Chat memory with audit trail

Usage:
    from langchain_openai import ChatOpenAI
    from maif.integrations.langchain import MAIFCallbackHandler
    
    handler = MAIFCallbackHandler("session.maif")
    llm = ChatOpenAI(callbacks=[handler])
    llm.invoke("Hello!")
    handler.finalize()
"""

from maif.integrations.langchain.callback import MAIFCallbackHandler
from maif.integrations.langchain.vectorstore import MAIFVectorStore
from maif.integrations.langchain.memory import MAIFChatMessageHistory

__all__ = [
    "MAIFCallbackHandler",
    "MAIFVectorStore",
    "MAIFChatMessageHistory",
]
