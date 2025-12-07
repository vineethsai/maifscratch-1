"""
MAIF Callback Handler for LangChain.

Tracks all LLM calls, chain runs, and tool invocations with
cryptographic provenance.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
    from langchain_core.agents import AgentAction, AgentFinish
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object

from maif.integrations._base import EventType, MAIFProvenanceTracker
from maif.integrations._utils import safe_serialize


class MAIFCallbackHandler(BaseCallbackHandler if LANGCHAIN_AVAILABLE else object):
    """MAIF-backed callback handler for LangChain.
    
    Tracks all LLM calls, chain executions, tool invocations, and
    agent actions with full cryptographic provenance.
    
    Usage:
        from langchain_openai import ChatOpenAI
        from maif.integrations.langchain import MAIFCallbackHandler
        
        handler = MAIFCallbackHandler("session.maif")
        llm = ChatOpenAI(callbacks=[handler])
        
        response = llm.invoke("Hello, how are you?")
        
        handler.finalize()
    
    All events are logged to a MAIF artifact with Ed25519 signatures
    for tamper-evident audit trails.
    """
    
    def __init__(
        self,
        artifact_path: str,
        agent_id: str = "langchain_handler",
    ):
        """Initialize the callback handler.
        
        Args:
            artifact_path: Path to the MAIF artifact file
            agent_id: Identifier for this handler
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for MAIFCallbackHandler. "
                "Install with: pip install langchain-core"
            )
        
        self.artifact_path = artifact_path
        self._agent_id = agent_id
        self._tracker = MAIFProvenanceTracker(
            artifact_path=artifact_path,
            agent_id=agent_id,
            auto_finalize=False,
        )
        
        # Track run metadata
        self._run_map: Dict[str, Dict[str, Any]] = {}
    
    # =========================================================================
    # LLM Callbacks
    # =========================================================================
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts."""
        run_id_str = str(run_id)
        
        self._run_map[run_id_str] = {
            "start_time": time.time(),
            "type": "llm",
        }
        
        self._tracker.log_event(
            event_type=EventType.NODE_START,
            data={
                "run_id": run_id_str,
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "llm_class": serialized.get("name", "unknown"),
                "prompts": prompts[:3],  # Limit for size
                "num_prompts": len(prompts),
                "tags": tags,
            },
            metadata={
                "type": "llm_start",
                "run_id": run_id_str,
                "timestamp": time.time(),
            },
        )
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends."""
        run_id_str = str(run_id)
        run_info = self._run_map.pop(run_id_str, {})
        duration = time.time() - run_info.get("start_time", time.time())
        
        # Extract response info
        generations = []
        for gen_list in response.generations:
            for gen in gen_list:
                generations.append({
                    "text": gen.text[:500] if gen.text else "",
                    "generation_info": safe_serialize(gen.generation_info),
                })
        
        self._tracker.log_event(
            event_type=EventType.NODE_END,
            data={
                "run_id": run_id_str,
                "generations": generations[:5],
                "llm_output": safe_serialize(response.llm_output),
                "duration_seconds": duration,
            },
            metadata={
                "type": "llm_end",
                "run_id": run_id_str,
                "duration": duration,
                "timestamp": time.time(),
            },
        )
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        run_id_str = str(run_id)
        self._run_map.pop(run_id_str, None)
        
        self._tracker.log_event(
            event_type=EventType.ERROR,
            data={
                "run_id": run_id_str,
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            },
            metadata={
                "type": "llm_error",
                "run_id": run_id_str,
                "timestamp": time.time(),
            },
        )
    
    # =========================================================================
    # Chat Model Callbacks
    # =========================================================================
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts."""
        run_id_str = str(run_id)
        
        self._run_map[run_id_str] = {
            "start_time": time.time(),
            "type": "chat",
        }
        
        # Serialize messages
        serialized_messages = []
        for msg_list in messages:
            for msg in msg_list:
                serialized_messages.append({
                    "type": msg.type,
                    "content": str(msg.content)[:200],
                })
        
        self._tracker.log_event(
            event_type=EventType.NODE_START,
            data={
                "run_id": run_id_str,
                "model_class": serialized.get("name", "unknown"),
                "messages": serialized_messages[:10],
                "num_messages": len(serialized_messages),
                "tags": tags,
            },
            metadata={
                "type": "chat_model_start",
                "run_id": run_id_str,
                "timestamp": time.time(),
            },
        )
    
    # =========================================================================
    # Chain Callbacks
    # =========================================================================
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts."""
        run_id_str = str(run_id)
        
        self._run_map[run_id_str] = {
            "start_time": time.time(),
            "type": "chain",
        }
        
        self._tracker.log_event(
            event_type=EventType.NODE_START,
            data={
                "run_id": run_id_str,
                "chain_class": serialized.get("name", "unknown"),
                "inputs": safe_serialize(inputs),
                "tags": tags,
            },
            metadata={
                "type": "chain_start",
                "run_id": run_id_str,
                "timestamp": time.time(),
            },
        )
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends."""
        run_id_str = str(run_id)
        run_info = self._run_map.pop(run_id_str, {})
        duration = time.time() - run_info.get("start_time", time.time())
        
        self._tracker.log_event(
            event_type=EventType.NODE_END,
            data={
                "run_id": run_id_str,
                "outputs": safe_serialize(outputs),
                "duration_seconds": duration,
            },
            metadata={
                "type": "chain_end",
                "run_id": run_id_str,
                "duration": duration,
                "timestamp": time.time(),
            },
        )
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        run_id_str = str(run_id)
        self._run_map.pop(run_id_str, None)
        
        self._tracker.log_event(
            event_type=EventType.ERROR,
            data={
                "run_id": run_id_str,
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            },
            metadata={
                "type": "chain_error",
                "run_id": run_id_str,
                "timestamp": time.time(),
            },
        )
    
    # =========================================================================
    # Tool Callbacks
    # =========================================================================
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts."""
        run_id_str = str(run_id)
        
        self._run_map[run_id_str] = {
            "start_time": time.time(),
            "type": "tool",
        }
        
        self._tracker.log_event(
            event_type=EventType.NODE_START,
            data={
                "run_id": run_id_str,
                "tool_name": serialized.get("name", "unknown"),
                "input": input_str[:500],
                "tags": tags,
            },
            metadata={
                "type": "tool_start",
                "run_id": run_id_str,
                "timestamp": time.time(),
            },
        )
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends."""
        run_id_str = str(run_id)
        run_info = self._run_map.pop(run_id_str, {})
        duration = time.time() - run_info.get("start_time", time.time())
        
        self._tracker.log_event(
            event_type=EventType.NODE_END,
            data={
                "run_id": run_id_str,
                "output": str(output)[:500],
                "duration_seconds": duration,
            },
            metadata={
                "type": "tool_end",
                "run_id": run_id_str,
                "duration": duration,
                "timestamp": time.time(),
            },
        )
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        run_id_str = str(run_id)
        self._run_map.pop(run_id_str, None)
        
        self._tracker.log_event(
            event_type=EventType.ERROR,
            data={
                "run_id": run_id_str,
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            },
            metadata={
                "type": "tool_error",
                "run_id": run_id_str,
                "timestamp": time.time(),
            },
        )
    
    # =========================================================================
    # Agent Callbacks
    # =========================================================================
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        self._tracker.log_event(
            event_type=EventType.NODE_START,
            data={
                "run_id": str(run_id),
                "tool": action.tool,
                "tool_input": safe_serialize(action.tool_input),
                "log": action.log[:500] if action.log else None,
            },
            metadata={
                "type": "agent_action",
                "run_id": str(run_id),
                "timestamp": time.time(),
            },
        )
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        self._tracker.log_event(
            event_type=EventType.NODE_END,
            data={
                "run_id": str(run_id),
                "return_values": safe_serialize(finish.return_values),
                "log": finish.log[:500] if finish.log else None,
            },
            metadata={
                "type": "agent_finish",
                "run_id": str(run_id),
                "timestamp": time.time(),
            },
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def finalize(self) -> None:
        """Finalize the MAIF artifact."""
        self._tracker.finalize()
    
    def get_artifact_path(self) -> str:
        """Get the artifact path."""
        return str(self.artifact_path)
    
    def __enter__(self) -> "MAIFCallbackHandler":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finalize()

