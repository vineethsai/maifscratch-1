#!/usr/bin/env python3
"""
Customer Support Multi-Agent System with MAIF Provenance

A production-ready customer support system demonstrating:
- Multiple specialized agents (triage, technical, billing, escalation)
- Intelligent routing based on query intent
- Full audit trail for compliance and QA
- Human-in-the-loop escalation

Usage:
    python main.py

Requirements:
    pip install maif langgraph
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, List, Annotated, Optional, Literal
from operator import add
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from langgraph.graph import StateGraph, START, END
from maif.integrations.langgraph import MAIFCheckpointer


# =============================================================================
# Types and State
# =============================================================================

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketCategory(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    ESCALATION = "escalation"


class Message(TypedDict):
    role: str
    content: str
    agent: Optional[str]
    timestamp: str


class SupportState(TypedDict):
    """State for customer support workflow."""
    messages: Annotated[List[Message], add]
    ticket_id: str
    customer_id: str
    category: Optional[str]
    priority: Optional[str]
    resolved: bool
    escalated: bool
    resolution_notes: str
    agent_actions: Annotated[List[dict], add]


# =============================================================================
# Knowledge Bases
# =============================================================================

TECHNICAL_KB = {
    "login": "For login issues: 1) Clear browser cache, 2) Reset password via /forgot-password, 3) Check if account is locked",
    "api": "API issues: Check your API key in Settings > API. Rate limit is 1000/min. Use exponential backoff for 429 errors.",
    "performance": "For slow performance: 1) Check system status at status.example.com, 2) Try a different browser, 3) Disable extensions",
    "integration": "Integration help: See docs at docs.example.com/integrations. Common issue: OAuth scopes need to include 'read' and 'write'",
}

BILLING_KB = {
    "refund": "Refund policy: Full refund within 30 days. Pro-rated refund after 30 days. Process takes 5-7 business days.",
    "upgrade": "To upgrade: Go to Settings > Billing > Change Plan. Changes apply immediately, pro-rated for current cycle.",
    "invoice": "Invoices available at Settings > Billing > Invoices. PDF download available. Contact billing@example.com for custom invoices.",
    "cancel": "To cancel: Settings > Billing > Cancel Subscription. Data retained for 30 days. Reactivate anytime.",
}


# =============================================================================
# Agent Functions
# =============================================================================

def triage_agent(state: SupportState) -> SupportState:
    """Analyze ticket and determine category/priority."""
    messages = state["messages"]
    last_message = messages[-1]["content"].lower() if messages else ""
    
    # Determine category
    category = TicketCategory.GENERAL.value
    if any(word in last_message for word in ["login", "error", "bug", "crash", "api", "slow"]):
        category = TicketCategory.TECHNICAL.value
    elif any(word in last_message for word in ["bill", "charge", "refund", "payment", "invoice", "cancel"]):
        category = TicketCategory.BILLING.value
    
    # Determine priority
    priority = TicketPriority.MEDIUM.value
    if any(word in last_message for word in ["urgent", "emergency", "critical", "down", "production"]):
        priority = TicketPriority.URGENT.value
    elif any(word in last_message for word in ["important", "asap", "deadline"]):
        priority = TicketPriority.HIGH.value
    elif any(word in last_message for word in ["question", "wondering", "curious"]):
        priority = TicketPriority.LOW.value
    
    print(f"[TRIAGE] Category: {category}, Priority: {priority}")
    
    return {
        "category": category,
        "priority": priority,
        "messages": [],
        "agent_actions": [{
            "agent": "triage",
            "action": "classify",
            "category": category,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
        }],
        "resolved": False,
        "escalated": False,
        "resolution_notes": "",
    }


def technical_agent(state: SupportState) -> SupportState:
    """Handle technical support issues."""
    messages = state["messages"]
    query = messages[-1]["content"].lower() if messages else ""
    
    # Find relevant knowledge
    response_parts = []
    for key, solution in TECHNICAL_KB.items():
        if key in query:
            response_parts.append(solution)
    
    if response_parts:
        response = "Here's how to resolve your issue:\n\n" + "\n\n".join(response_parts)
        resolved = True
    else:
        response = "I understand you're having a technical issue. Let me gather more information. Could you describe the exact error message or steps to reproduce?"
        resolved = False
    
    print(f"[TECHNICAL] Resolved: {resolved}")
    
    return {
        "messages": [{
            "role": "assistant",
            "content": response,
            "agent": "technical",
            "timestamp": datetime.now().isoformat(),
        }],
        "resolved": resolved,
        "agent_actions": [{
            "agent": "technical",
            "action": "respond",
            "resolved": resolved,
            "timestamp": datetime.now().isoformat(),
        }],
        "escalated": False,
        "resolution_notes": "",
        "category": None,
        "priority": None,
    }


def billing_agent(state: SupportState) -> SupportState:
    """Handle billing inquiries."""
    messages = state["messages"]
    query = messages[-1]["content"].lower() if messages else ""
    
    # Find relevant knowledge
    response_parts = []
    for key, info in BILLING_KB.items():
        if key in query:
            response_parts.append(info)
    
    if response_parts:
        response = "Here's the billing information you need:\n\n" + "\n\n".join(response_parts)
        resolved = True
    else:
        response = "I can help with billing questions. What specifically would you like to know about your account, invoices, or subscription?"
        resolved = False
    
    print(f"[BILLING] Resolved: {resolved}")
    
    return {
        "messages": [{
            "role": "assistant",
            "content": response,
            "agent": "billing",
            "timestamp": datetime.now().isoformat(),
        }],
        "resolved": resolved,
        "agent_actions": [{
            "agent": "billing",
            "action": "respond",
            "resolved": resolved,
            "timestamp": datetime.now().isoformat(),
        }],
        "escalated": False,
        "resolution_notes": "",
        "category": None,
        "priority": None,
    }


def general_agent(state: SupportState) -> SupportState:
    """Handle general inquiries."""
    response = """Thank you for contacting support! 

I can help you with:
- Technical issues (login, API, performance)
- Billing questions (invoices, refunds, subscriptions)
- General product questions

Could you provide more details about what you need help with?"""
    
    print("[GENERAL] Providing initial response")
    
    return {
        "messages": [{
            "role": "assistant",
            "content": response,
            "agent": "general",
            "timestamp": datetime.now().isoformat(),
        }],
        "resolved": False,
        "agent_actions": [{
            "agent": "general",
            "action": "initial_response",
            "timestamp": datetime.now().isoformat(),
        }],
        "escalated": False,
        "resolution_notes": "",
        "category": None,
        "priority": None,
    }


def escalation_agent(state: SupportState) -> SupportState:
    """Handle escalations to human agents."""
    priority = state.get("priority", "medium")
    
    response = f"""I've escalated your ticket to a senior support specialist.

Ticket ID: {state['ticket_id']}
Priority: {priority.upper()}
Estimated response time: {"15 minutes" if priority == "urgent" else "2 hours"}

A human agent will contact you shortly. Is there anything else I can note for them?"""
    
    print(f"[ESCALATION] Ticket escalated with priority {priority}")
    
    return {
        "messages": [{
            "role": "assistant",
            "content": response,
            "agent": "escalation",
            "timestamp": datetime.now().isoformat(),
        }],
        "escalated": True,
        "agent_actions": [{
            "agent": "escalation",
            "action": "escalate_to_human",
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
        }],
        "resolved": False,
        "resolution_notes": "",
        "category": None,
        "priority": None,
    }


# =============================================================================
# Router
# =============================================================================

def route_to_agent(state: SupportState) -> Literal["technical", "billing", "general", "escalation"]:
    """Route to appropriate agent based on category and priority."""
    category = state.get("category", "general")
    priority = state.get("priority", "medium")
    
    # Urgent tickets always escalate
    if priority == TicketPriority.URGENT.value:
        return "escalation"
    
    # Route by category
    if category == TicketCategory.TECHNICAL.value:
        return "technical"
    elif category == TicketCategory.BILLING.value:
        return "billing"
    else:
        return "general"


# =============================================================================
# Build Graph
# =============================================================================

def create_support_system(artifact_path: str = "support_system.maif"):
    """Create the customer support graph with MAIF provenance."""
    
    graph = StateGraph(SupportState)
    
    # Add nodes
    graph.add_node("triage", triage_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("billing", billing_agent)
    graph.add_node("general", general_agent)
    graph.add_node("escalation", escalation_agent)
    
    # Add edges
    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_to_agent,
        {
            "technical": "technical",
            "billing": "billing",
            "general": "general",
            "escalation": "escalation",
        }
    )
    
    # All agents end
    graph.add_edge("technical", END)
    graph.add_edge("billing", END)
    graph.add_edge("general", END)
    graph.add_edge("escalation", END)
    
    # Compile with MAIF
    checkpointer = MAIFCheckpointer(artifact_path, agent_id="support_system")
    app = graph.compile(checkpointer=checkpointer)
    
    return app, checkpointer


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Customer Support System with MAIF Provenance")
    print("=" * 60)
    print()
    print("Agents: Triage | Technical | Billing | General | Escalation")
    print("Type 'quit' to exit, 'audit' to see agent actions")
    print("-" * 60)
    
    app, checkpointer = create_support_system()
    
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = {"configurable": {"thread_id": ticket_id}}
    
    print(f"\nTicket ID: {ticket_id}")
    print()
    
    while True:
        try:
            user_input = input("\nCustomer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            break
        
        if user_input.lower() == "audit":
            from maif import MAIFDecoder
            try:
                decoder = MAIFDecoder(checkpointer.get_artifact_path())
                decoder.load()
                print(f"\n--- Agent Actions Audit ---")
                print(f"Total events: {len(decoder.blocks)}")
                print(f"Ticket: {ticket_id}")
                is_valid, _ = decoder.verify_integrity()
                print(f"Integrity: {'VERIFIED' if is_valid else 'FAILED'}")
                print("---")
            except:
                print("No audit data yet")
            continue
        
        # Process query
        result = app.invoke(
            {
                "messages": [{
                    "role": "user",
                    "content": user_input,
                    "agent": None,
                    "timestamp": datetime.now().isoformat(),
                }],
                "ticket_id": ticket_id,
                "customer_id": "customer-001",
                "category": None,
                "priority": None,
                "resolved": False,
                "escalated": False,
                "resolution_notes": "",
                "agent_actions": [],
            },
            config=config
        )
        
        # Show response
        for msg in result["messages"]:
            if msg["role"] == "assistant":
                agent = msg.get("agent", "unknown")
                print(f"\n[{agent.upper()}]: {msg['content']}")
        
        if result.get("resolved"):
            print("\n[TICKET RESOLVED]")
        if result.get("escalated"):
            print("\n[TICKET ESCALATED TO HUMAN]")
    
    # Finalize
    print("\n" + "=" * 60)
    checkpointer.finalize()
    
    from maif import MAIFDecoder
    decoder = MAIFDecoder(checkpointer.get_artifact_path())
    decoder.load()
    
    print(f"Session saved: {checkpointer.get_artifact_path()}")
    print(f"Total events: {len(decoder.blocks)}")
    is_valid, _ = decoder.verify_integrity()
    print(f"Integrity: {'VERIFIED' if is_valid else 'FAILED'}")


if __name__ == "__main__":
    main()

