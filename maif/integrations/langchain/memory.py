"""
MAIF Chat Message History for LangChain.

Provides persistent chat memory with cryptographic provenance.
"""

import time
import json
from typing import List, Optional, Sequence
from pathlib import Path

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
        FunctionMessage,
        ToolMessage,
        messages_from_dict,
        message_to_dict,
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatMessageHistory = object
    BaseMessage = dict

from maif.integrations._base import EventType, MAIFProvenanceTracker
from maif.integrations._utils import safe_serialize


class MAIFChatMessageHistory(BaseChatMessageHistory if LANGCHAIN_AVAILABLE else object):
    """MAIF-backed chat message history for LangChain.
    
    Stores chat messages in a MAIF artifact with cryptographic
    provenance tracking. Every message is signed and hash-chained.
    
    Usage:
        from maif.integrations.langchain import MAIFChatMessageHistory
        from langchain_core.messages import HumanMessage, AIMessage
        
        history = MAIFChatMessageHistory(
            session_id="user-123",
            artifact_path="chat_history.maif"
        )
        
        history.add_user_message("Hello!")
        history.add_ai_message("Hi there! How can I help?")
        
        # Get all messages
        messages = history.messages
        
        # Finalize when done
        history.finalize()
    
    With RunnableWithMessageHistory:
        from langchain_core.runnables.history import RunnableWithMessageHistory
        
        def get_history(session_id):
            return MAIFChatMessageHistory(session_id, "history.maif")
        
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    """
    
    def __init__(
        self,
        session_id: str,
        artifact_path: str,
        agent_id: str = "chat_history",
    ):
        """Initialize chat history.
        
        Args:
            session_id: Unique session identifier
            artifact_path: Path to MAIF artifact
            agent_id: Identifier for this history
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for MAIFChatMessageHistory. "
                "Install with: pip install langchain-core"
            )
        
        self.session_id = session_id
        self.artifact_path = Path(artifact_path)
        self._agent_id = agent_id
        
        # Initialize tracker
        self._tracker = MAIFProvenanceTracker(
            artifact_path=artifact_path,
            agent_id=agent_id,
            auto_finalize=False,
        )
        
        # In-memory message storage
        self._messages: List[BaseMessage] = []
        
        # Load existing messages
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing messages from MAIF artifact."""
        if not self.artifact_path.exists():
            return
        
        try:
            from maif import MAIFDecoder
            
            decoder = MAIFDecoder(str(self.artifact_path))
            decoder.load()
            
            for block in decoder.blocks:
                meta = block.metadata or {}
                if meta.get("type") in ["human_message", "ai_message", "system_message"]:
                    if meta.get("session_id") != self.session_id:
                        continue
                    
                    try:
                        data = block.data
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        event_data = json.loads(data).get("data", {})
                        
                        msg_type = meta.get("type")
                        content = event_data.get("content", "")
                        
                        if msg_type == "human_message":
                            self._messages.append(HumanMessage(content=content))
                        elif msg_type == "ai_message":
                            self._messages.append(AIMessage(content=content))
                        elif msg_type == "system_message":
                            self._messages.append(SystemMessage(content=content))
                    except:
                        pass
        except Exception:
            pass
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages in history."""
        return self._messages.copy()
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to history.
        
        Args:
            message: Message to add
        """
        self._messages.append(message)
        
        # Determine message type
        if isinstance(message, HumanMessage):
            msg_type = "human_message"
        elif isinstance(message, AIMessage):
            msg_type = "ai_message"
        elif isinstance(message, SystemMessage):
            msg_type = "system_message"
        elif isinstance(message, FunctionMessage):
            msg_type = "function_message"
        elif isinstance(message, ToolMessage):
            msg_type = "tool_message"
        else:
            msg_type = "unknown_message"
        
        # Log to MAIF
        self._tracker.log_event(
            event_type=EventType.STATE_CHECKPOINT,
            data={
                "content": str(message.content)[:2000],
                "additional_kwargs": safe_serialize(message.additional_kwargs),
            },
            metadata={
                "type": msg_type,
                "session_id": self.session_id,
                "message_index": len(self._messages) - 1,
                "timestamp": time.time(),
            },
        )
    
    def add_user_message(self, message: str) -> None:
        """Add a user message.
        
        Args:
            message: User message content
        """
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message.
        
        Args:
            message: AI message content
        """
        self.add_message(AIMessage(content=message))
    
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages.
        
        Args:
            messages: Messages to add
        """
        for message in messages:
            self.add_message(message)
    
    def clear(self) -> None:
        """Clear message history.
        
        Note: This clears the in-memory history but the MAIF artifact
        retains the full audit trail of all messages ever added.
        """
        # Log the clear action
        self._tracker.log_event(
            event_type=EventType.STATE_CHECKPOINT,
            data={
                "action": "clear",
                "messages_cleared": len(self._messages),
            },
            metadata={
                "type": "history_clear",
                "session_id": self.session_id,
                "timestamp": time.time(),
            },
        )
        
        self._messages.clear()
    
    def finalize(self) -> None:
        """Finalize the MAIF artifact."""
        self._tracker.finalize()
    
    def get_artifact_path(self) -> str:
        """Get the artifact path."""
        return str(self.artifact_path)
    
    def __len__(self) -> int:
        """Return number of messages."""
        return len(self._messages)
    
    def __enter__(self) -> "MAIFChatMessageHistory":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finalize()

