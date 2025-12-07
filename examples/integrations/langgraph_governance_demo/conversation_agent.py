"""
Multi-agent conversation system with access control and provenance.

This module implements a LangGraph-based conversation agent that demonstrates:
- Access control checks on queries
- Query routing to specialized agents
- Compliance checking on responses
- Full provenance tracking via MAIF
"""

import time
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from operator import add
from dataclasses import dataclass

# Terminal formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"


# Role-based permissions
ROLE_PERMISSIONS = {
    "analyst": {
        "financial_summary": True,
        "reports": True,
        "customer_pii": False,
        "raw_transactions": False,
        "admin_settings": False,
    },
    "manager": {
        "financial_summary": True,
        "reports": True,
        "customer_pii": True,
        "raw_transactions": True,
        "admin_settings": False,
    },
    "admin": {
        "financial_summary": True,
        "reports": True,
        "customer_pii": True,
        "raw_transactions": True,
        "admin_settings": True,
    },
    "auditor": {
        "financial_summary": False,
        "reports": True,
        "customer_pii": False,
        "raw_transactions": False,
        "admin_settings": False,
    },
}

# Query patterns for routing
QUERY_PATTERNS = {
    "financial": [
        r"revenue", r"profit", r"sales", r"quarter", r"q[1-4]",
        r"financial", r"earnings", r"budget", r"forecast",
    ],
    "customer": [
        r"customer", r"client", r"user", r"account",
        r"transaction history", r"purchase",
    ],
    "general": [
        r"what", r"how", r"why", r"when", r"where",
        r"explain", r"describe", r"tell me",
    ],
}

# Simulated knowledge base
KNOWLEDGE_BASE = {
    "financial": {
        "q3_revenue": "Q3 2024 revenue was $42.5M, representing a 15% YoY increase.",
        "q3_profit": "Q3 2024 net profit was $8.2M with a margin of 19.3%.",
        "forecast": "Q4 2024 forecast projects revenue of $48M based on current pipeline.",
        "budget": "Annual budget allocation: R&D 35%, Sales 25%, Operations 20%, Other 20%.",
    },
    "general": {
        "company": "We are a technology company focused on AI solutions.",
        "products": "Our main products include data analytics and AI governance tools.",
        "team": "The team consists of 150+ professionals across engineering and sales.",
    },
}


class ConversationState(TypedDict):
    """State for the conversation graph."""
    messages: Annotated[List[Dict[str, str]], add]
    query: str
    query_type: str
    access_granted: bool
    access_denied_reason: str
    retrieved_context: str
    response: str
    compliance_passed: bool
    compliance_issues: List[str]
    current_agent: str
    user_role: str


class ConversationAgent:
    """Multi-agent conversation system with provenance tracking."""
    
    def __init__(
        self,
        artifact_path: str,
        session_name: str,
        user_role: str = "analyst",
        resume: bool = False,
    ):
        """Initialize the conversation agent.
        
        Args:
            artifact_path: Path to MAIF artifact
            session_name: Name of this session
            user_role: Role of the current user
            resume: Whether to resume an existing session
        """
        self.artifact_path = Path(artifact_path)
        self.session_name = session_name
        self.user_role = user_role
        self.conversation_history: List[Dict[str, str]] = []
        
        # Initialize MAIF checkpointer
        self._init_checkpointer(resume)
        
        # Build the conversation graph
        self._build_graph()
    
    def _init_checkpointer(self, resume: bool):
        """Initialize the MAIF checkpointer."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
            
            self.checkpointer = MAIFCheckpointer(
                artifact_path=self.artifact_path,
                agent_id=f"governance_demo_{self.session_name}",
            )
            
            if resume and self.artifact_path.exists():
                # Load conversation history from existing artifact
                self._load_history()
        except ImportError:
            print(f"{RED}Error: LangGraph integration not available.{RESET}")
            print("Install with: pip install maif[integrations]")
            raise
    
    def _load_history(self):
        """Load conversation history from existing artifact."""
        try:
            from maif import MAIFDecoder
            
            decoder = MAIFDecoder(str(self.artifact_path))
            decoder.load()
            
            for block in decoder.blocks:
                meta = block.metadata or {}
                if meta.get("type") == "user_message":
                    data = json.loads(block.data.decode("utf-8"))
                    self.conversation_history.append({
                        "role": "user",
                        "content": data.get("query", ""),
                    })
                elif meta.get("type") == "assistant_response":
                    data = json.loads(block.data.decode("utf-8"))
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": data.get("response", ""),
                    })
        except Exception as e:
            print(f"{DIM}Could not load history: {e}{RESET}")
    
    def _build_graph(self):
        """Build the LangGraph conversation graph."""
        try:
            from langgraph.graph import StateGraph, START, END
        except ImportError as e:
            print(f"{RED}Error: LangGraph not available: {e}{RESET}")
            print("Install with: pip install langgraph langchain-core")
            print("You may need to update langchain-core: pip install -U langchain-core")
            raise ImportError(f"LangGraph import failed: {e}")
        
        # Create the graph
        graph = StateGraph(ConversationState)
        
        # Add nodes
        graph.add_node("access_control", self._access_control_node)
        graph.add_node("router", self._router_node)
        graph.add_node("financial_agent", self._financial_agent_node)
        graph.add_node("general_agent", self._general_agent_node)
        graph.add_node("synthesizer", self._synthesizer_node)
        graph.add_node("compliance_checker", self._compliance_checker_node)
        graph.add_node("access_denied_response", self._access_denied_node)
        
        # Add edges
        graph.add_edge(START, "access_control")
        
        # Conditional routing after access control
        graph.add_conditional_edges(
            "access_control",
            self._route_after_access_control,
            {
                "router": "router",
                "denied": "access_denied_response",
            }
        )
        
        # Conditional routing after router
        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "financial": "financial_agent",
                "general": "general_agent",
            }
        )
        
        # Both agents go to synthesizer
        graph.add_edge("financial_agent", "synthesizer")
        graph.add_edge("general_agent", "synthesizer")
        
        # Synthesizer goes to compliance
        graph.add_edge("synthesizer", "compliance_checker")
        
        # End states
        graph.add_edge("compliance_checker", END)
        graph.add_edge("access_denied_response", END)
        
        # Compile with checkpointer
        self.app = graph.compile(checkpointer=self.checkpointer)
    
    def _access_control_node(self, state: ConversationState) -> ConversationState:
        """Check access control for the query."""
        print(f"{DIM}[ACCESS CONTROL] Checking permissions for role: {state['user_role']}{RESET}")
        
        query = state["query"].lower()
        user_role = state["user_role"]
        permissions = ROLE_PERMISSIONS.get(user_role, {})
        
        # Check if query requires special permissions
        requires_pii = any(word in query for word in ["customer", "client", "user", "account"])
        requires_raw = "transaction" in query and "history" in query
        
        access_granted = True
        denied_reason = ""
        
        if requires_pii and not permissions.get("customer_pii", False):
            access_granted = False
            denied_reason = f"Role '{user_role}' does not have permission to access customer PII"
        elif requires_raw and not permissions.get("raw_transactions", False):
            access_granted = False
            denied_reason = f"Role '{user_role}' does not have permission to access raw transactions"
        
        status = f"{GREEN}GRANTED{RESET}" if access_granted else f"{RED}DENIED{RESET}"
        print(f"{DIM}[ACCESS CONTROL] Permission: {status}{RESET}")
        
        return {
            "access_granted": access_granted,
            "access_denied_reason": denied_reason,
            "current_agent": "access_control",
            "messages": [{
                "agent": "access_control",
                "action": "permission_check",
                "granted": access_granted,
                "reason": denied_reason if not access_granted else "all permissions satisfied",
            }],
        }
    
    def _route_after_access_control(self, state: ConversationState) -> str:
        """Route based on access control result."""
        if state.get("access_granted", False):
            return "router"
        return "denied"
    
    def _router_node(self, state: ConversationState) -> ConversationState:
        """Route the query to the appropriate agent."""
        query = state["query"].lower()
        
        # Determine query type based on patterns
        query_type = "general"
        confidence = 0.0
        
        for qtype, patterns in QUERY_PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, query, re.IGNORECASE))
            if matches > confidence:
                confidence = matches
                query_type = qtype
        
        print(f"{DIM}[ROUTER] Query classified as: {query_type} (confidence: {confidence:.2f}){RESET}")
        print(f"{DIM}[ROUTER] Routing to: {query_type}_agent{RESET}")
        
        return {
            "query_type": query_type,
            "current_agent": "router",
            "messages": [{
                "agent": "router",
                "action": "classify_query",
                "query_type": query_type,
                "confidence": confidence,
            }],
        }
    
    def _route_after_router(self, state: ConversationState) -> str:
        """Route to the appropriate agent."""
        query_type = state.get("query_type", "general")
        if query_type == "financial":
            return "financial"
        return "general"
    
    def _financial_agent_node(self, state: ConversationState) -> ConversationState:
        """Handle financial queries."""
        print(f"{DIM}[FINANCIAL AGENT] Processing financial query...{RESET}")
        
        query = state["query"].lower()
        context_parts = []
        
        # Search knowledge base
        for key, value in KNOWLEDGE_BASE["financial"].items():
            if any(word in query for word in key.split("_")):
                context_parts.append(value)
        
        if not context_parts:
            # Default financial context
            context_parts.append(KNOWLEDGE_BASE["financial"]["q3_revenue"])
        
        context = " ".join(context_parts)
        print(f"{DIM}[FINANCIAL AGENT] Retrieved {len(context_parts)} relevant facts{RESET}")
        
        return {
            "retrieved_context": context,
            "current_agent": "financial_agent",
            "messages": [{
                "agent": "financial_agent",
                "action": "retrieve",
                "num_facts": len(context_parts),
                "context_length": len(context),
            }],
        }
    
    def _general_agent_node(self, state: ConversationState) -> ConversationState:
        """Handle general queries."""
        print(f"{DIM}[GENERAL AGENT] Processing general query...{RESET}")
        
        query = state["query"].lower()
        context_parts = []
        
        # Search knowledge base
        for key, value in KNOWLEDGE_BASE["general"].items():
            if any(word in query for word in key.split("_")):
                context_parts.append(value)
        
        if not context_parts:
            context_parts.append("I can help you with questions about our company, products, and services.")
        
        context = " ".join(context_parts)
        print(f"{DIM}[GENERAL AGENT] Retrieved {len(context_parts)} relevant facts{RESET}")
        
        return {
            "retrieved_context": context,
            "current_agent": "general_agent",
            "messages": [{
                "agent": "general_agent",
                "action": "retrieve",
                "num_facts": len(context_parts),
            }],
        }
    
    def _synthesizer_node(self, state: ConversationState) -> ConversationState:
        """Synthesize a response from the retrieved context."""
        print(f"{DIM}[SYNTHESIZER] Generating response...{RESET}")
        
        context = state.get("retrieved_context", "")
        query = state.get("query", "")
        
        # Simple response synthesis (in production, this would use an LLM)
        if context:
            response = f"Based on available data: {context}"
        else:
            response = "I don't have specific information about that. Could you please rephrase your question?"
        
        return {
            "response": response,
            "current_agent": "synthesizer",
            "messages": [{
                "agent": "synthesizer",
                "action": "generate_response",
                "response_length": len(response),
            }],
        }
    
    def _compliance_checker_node(self, state: ConversationState) -> ConversationState:
        """Check response for compliance issues."""
        print(f"{DIM}[COMPLIANCE] Checking response for policy violations...{RESET}")
        
        response = state.get("response", "")
        issues = []
        
        # Check for PII patterns
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN detected"),
            (r"\b\d{16}\b", "Credit card number detected"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email detected"),
        ]
        
        for pattern, issue in pii_patterns:
            if re.search(pattern, response):
                issues.append(issue)
        
        passed = len(issues) == 0
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"{DIM}[COMPLIANCE] Check result: {status}{RESET}")
        
        if issues:
            for issue in issues:
                print(f"{DIM}[COMPLIANCE] Issue: {issue}{RESET}")
        
        return {
            "compliance_passed": passed,
            "compliance_issues": issues,
            "current_agent": "compliance_checker",
            "messages": [{
                "agent": "compliance_checker",
                "action": "compliance_check",
                "passed": passed,
                "issues": issues,
            }],
        }
    
    def _access_denied_node(self, state: ConversationState) -> ConversationState:
        """Handle access denied responses."""
        reason = state.get("access_denied_reason", "Access denied")
        
        response = f"ACCESS DENIED: {reason}. This access attempt has been logged to the audit trail."
        
        print(f"{RED}[ACCESS DENIED] {reason}{RESET}")
        
        return {
            "response": response,
            "compliance_passed": True,
            "compliance_issues": [],
            "current_agent": "access_denied_response",
            "messages": [{
                "agent": "access_denied_response",
                "action": "deny_access",
                "reason": reason,
            }],
        }
    
    def process_query(self, query: str) -> str:
        """Process a user query through the graph.
        
        Args:
            query: User's query string
            
        Returns:
            Response string
        """
        # Create initial state
        initial_state = {
            "messages": [],
            "query": query,
            "query_type": "",
            "access_granted": False,
            "access_denied_reason": "",
            "retrieved_context": "",
            "response": "",
            "compliance_passed": False,
            "compliance_issues": [],
            "current_agent": "",
            "user_role": self.user_role,
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": self.session_name}}
        result = self.app.invoke(initial_state, config=config)
        
        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": result["response"]})
        
        return result["response"]
    
    def get_conversation_summary(self) -> List[str]:
        """Get a summary of the conversation history.
        
        Returns:
            List of formatted conversation entries
        """
        summary = []
        for entry in self.conversation_history[-6:]:  # Last 3 exchanges
            role = "You" if entry["role"] == "user" else "Assistant"
            content = entry["content"][:100] + "..." if len(entry["content"]) > 100 else entry["content"]
            summary.append(f"{role}: {content}")
        return summary
    
    def show_full_history(self):
        """Display the full conversation history."""
        print()
        print(f"{BOLD}Conversation History{RESET}")
        print("-" * 60)
        for entry in self.conversation_history:
            role = f"{GREEN}You{RESET}" if entry["role"] == "user" else f"{BLUE}Assistant{RESET}"
            print(f"{role}: {entry['content']}")
            print()
        print("-" * 60)
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize the session and return summary.
        
        Returns:
            Session summary dict
        """
        self.checkpointer.finalize()
        
        # Get artifact stats
        try:
            from maif import MAIFDecoder
            decoder = MAIFDecoder(str(self.artifact_path))
            decoder.load()
            total_events = len(decoder.blocks)
        except Exception:
            total_events = len(self.conversation_history) * 2
        
        return {
            "session_name": self.session_name,
            "total_events": total_events,
            "conversation_turns": len(self.conversation_history) // 2,
            "artifact_path": str(self.artifact_path),
        }

