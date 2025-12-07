# Agent Development

::: danger DEPRECATED
This page is deprecated. For the latest documentation, please visit **[DeepWiki](https://deepwiki.com/vineethsai/maif)**.
:::

This guide covers building AI agents that use MAIF for memory, state management, and provenance tracking.

## Overview

MAIF provides tools for building AI agents with:

- **Persistent Memory**: Store agent state in MAIF files
- **Provenance Tracking**: Track all agent actions
- **Security**: Encrypt sensitive data
- **Multi-modal Support**: Text, images, embeddings

## MAIF Agent Framework

### Basic Agent Structure

```python
from maif.agentic_framework import MAIFAgent

class MyAgent(MAIFAgent):
    """Custom AI agent using MAIF."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        
        # Initialize agent-specific resources
        self.initialize()
    
    def initialize(self):
        """Set up agent resources."""
        pass
    
    def process(self, input_data):
        """Process input and return response."""
        # Implement your agent logic
        pass
```

### Agent Components

```python
from maif.agentic_framework import (
    MAIFAgent,
    PerceptionSystem,
    ReasoningSystem,
    ExecutionSystem
)

# Perception: Process incoming data
perception = PerceptionSystem()

# Reasoning: Make decisions
reasoning = ReasoningSystem()

# Execution: Take actions
execution = ExecutionSystem()
```

## Building an Agent with MAIF Memory

### Simple Agent with Memory

```python
from maif_api import create_maif, load_maif
import os

class MemoryAgent:
    """Agent with MAIF-backed memory."""
    
    def __init__(self, agent_id: str, memory_path: str = None):
        self.agent_id = agent_id
        self.memory_path = memory_path or f"agents/{agent_id}/memory.maif"
        
        # Load or create memory
        if os.path.exists(self.memory_path):
            self.memory = load_maif(self.memory_path)
        else:
            self.memory = create_maif(agent_id)
    
    def remember(self, content: str, title: str = None):
        """Store something in memory."""
        self.memory.add_text(content, title=title)
    
    def recall(self, query: str, top_k: int = 5):
        """Search memory."""
        return self.memory.search(query, top_k=top_k)
    
    def save(self):
        """Persist memory to disk."""
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        self.memory.save(self.memory_path)

# Usage
agent = MemoryAgent("assistant")
agent.remember("User prefers brief responses", title="Preference")
agent.remember("Previous topic was about Python", title="Context")
agent.save()

# Later...
results = agent.recall("preferences")
```

### Conversational Agent

```python
from maif_api import create_maif, load_maif
from datetime import datetime

class ConversationalAgent:
    """Agent that maintains conversation history."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.conversation = create_maif(f"{agent_id}-conversation")
        self.turn_count = 0
    
    def process_message(self, user_message: str) -> str:
        """Process a user message and return response."""
        self.turn_count += 1
        
        # Log user message
        self.conversation.add_text(
            f"User: {user_message}",
            title=f"Turn {self.turn_count} - User"
        )
        
        # Generate response (implement your logic)
        response = self._generate_response(user_message)
        
        # Log agent response
        self.conversation.add_text(
            f"Agent: {response}",
            title=f"Turn {self.turn_count} - Agent"
        )
        
        return response
    
    def _generate_response(self, message: str) -> str:
        # Implement your response logic
        # Could use LLM, rule-based, etc.
        return f"I received: {message}"
    
    def get_history(self):
        """Get conversation history."""
        return self.conversation.get_content_list()
    
    def save_conversation(self, path: str):
        """Save conversation to file."""
        self.conversation.save(path)

# Usage
agent = ConversationalAgent("chat-bot")
response1 = agent.process_message("Hello!")
response2 = agent.process_message("What can you do?")
agent.save_conversation("conversations/chat_001.maif")
```

## Agent with Secure Memory

### Privacy-Enabled Agent

```python
from maif_api import create_maif
from maif.core import MAIFEncoder
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode

class SecureAgent:
    """Agent with encrypted memory."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.privacy_engine = PrivacyEngine()
        
        # Create secure memory
        self.memory = create_maif(agent_id, enable_privacy=True)
        
        # For more control, use encoder directly
        self.encoder = MAIFEncoder(
            agent_id=agent_id,
            enable_privacy=True,
            privacy_engine=self.privacy_engine
        )
    
    def store_public(self, content: str, title: str = None):
        """Store public (unencrypted) content."""
        self.memory.add_text(content, title=title)
    
    def store_confidential(self, content: str, title: str = None):
        """Store encrypted content."""
        self.memory.add_text(
            content,
            title=title,
            encrypt=True
        )
    
    def store_with_anonymization(self, content: str, title: str = None):
        """Store content with PII anonymization."""
        self.memory.add_text(
            content,
            title=title,
            anonymize=True
        )
    
    def save(self, path: str):
        self.memory.save(path)

# Usage
agent = SecureAgent("secure-assistant")
agent.store_public("General information")
agent.store_confidential("Secret data: API key is xyz123")
agent.store_with_anonymization("User John Smith called about order #12345")
agent.save("secure_memory.maif")
```

## Agent with Provenance Tracking

### Tracked Agent

```python
from maif.security import MAIFSigner
from maif_api import create_maif

class TrackedAgent:
    """Agent with full provenance tracking."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.signer = MAIFSigner(agent_id=agent_id)
        self.memory = create_maif(agent_id)
        self.action_count = 0
    
    def perform_action(self, action: str, data: str):
        """Perform and track an action."""
        self.action_count += 1
        action_id = f"action_{self.action_count}"
        
        # Record action in memory
        self.memory.add_text(
            f"Action: {action}\nData: {data}",
            title=f"Action: {action_id}"
        )
        
        # Add provenance entry
        self.signer.add_provenance_entry(action, action_id)
        
        return action_id
    
    def get_provenance(self):
        """Get provenance trail."""
        # Provenance is tracked by the signer
        return self.signer.get_provenance_chain() if hasattr(self.signer, 'get_provenance_chain') else []
    
    def save(self, path: str):
        self.memory.save(path)
```

## Multi-Modal Agent

### Agent with Embeddings

```python
from maif_api import create_maif

class EmbeddingAgent:
    """Agent that uses embeddings for semantic memory."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory = create_maif(agent_id)
        self.documents = []
    
    def add_document(self, content: str, embeddings: list, title: str = None):
        """Add document with embeddings."""
        # Store text
        self.memory.add_text(content, title=title)
        
        # Store embeddings
        self.memory.add_embeddings(
            [embeddings],
            model_name="custom"
        )
        
        self.documents.append({
            "content": content,
            "embeddings": embeddings,
            "title": title
        })
    
    def semantic_search(self, query: str, top_k: int = 3):
        """Search using semantic similarity."""
        return self.memory.search(query, top_k=top_k)
    
    def save(self, path: str):
        self.memory.save(path)
```

## Integration with LangChain/LangGraph

### MAIF as Memory Backend

```python
from maif_api import create_maif, load_maif
from maif.core import MAIFEncoder, MAIFDecoder

class MAIFMemoryBackend:
    """Use MAIF as a LangChain memory backend."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory_path = f"sessions/{session_id}.maif"
        self.memory = create_maif(f"langchain-{session_id}")
    
    def add_message(self, role: str, content: str):
        """Add a message to memory."""
        self.memory.add_text(
            f"{role}: {content}",
            title=f"Message from {role}"
        )
    
    def get_messages(self):
        """Retrieve all messages."""
        return self.memory.get_content_list()
    
    def clear(self):
        """Clear memory."""
        self.memory = create_maif(f"langchain-{self.session_id}")
    
    def save(self):
        """Persist memory."""
        self.memory.save(self.memory_path)
    
    @classmethod
    def load(cls, session_id: str):
        """Load existing session."""
        instance = cls(session_id)
        instance.memory = load_maif(instance.memory_path)
        return instance
```

## Complete Example: Customer Service Agent

```python
from maif_api import create_maif, load_maif
from maif.security import MAIFSigner
from datetime import datetime
import os

class CustomerServiceAgent:
    """Complete customer service agent with MAIF."""
    
    def __init__(self, agent_id: str = "customer-service"):
        self.agent_id = agent_id
        self.base_path = f"agents/{agent_id}"
        
        # Create directory
        os.makedirs(self.base_path, exist_ok=True)
        
        # Initialize components
        self.signer = MAIFSigner(agent_id=agent_id)
        self.knowledge = self._load_or_create("knowledge")
        self.conversations = self._load_or_create("conversations", enable_privacy=True)
        self.logs = self._load_or_create("logs")
    
    def _load_or_create(self, name: str, enable_privacy: bool = False):
        """Load existing or create new MAIF artifact."""
        path = f"{self.base_path}/{name}.maif"
        if os.path.exists(path):
            return load_maif(path)
        return create_maif(f"{self.agent_id}-{name}", enable_privacy=enable_privacy)
    
    def load_faq(self, faqs: list):
        """Load FAQ into knowledge base."""
        for faq in faqs:
            self.knowledge.add_text(
                f"Q: {faq['question']}\nA: {faq['answer']}",
                title=f"FAQ: {faq['question'][:50]}"
            )
        self.signer.add_provenance_entry("load_faq", f"loaded_{len(faqs)}_faqs")
    
    def handle_query(self, customer_id: str, query: str) -> str:
        """Handle a customer query."""
        # Log the query
        self.conversations.add_text(
            f"Customer {customer_id}: {query}",
            title=f"Query from {customer_id}"
        )
        
        # Search knowledge base
        results = self.knowledge.search(query, top_k=3)
        
        # Generate response (simplified)
        if results:
            response = f"Based on our FAQ: {results[0].get('text', 'No answer found')[:200]}"
        else:
            response = "I'll need to escalate this to a human agent."
        
        # Log response
        self.conversations.add_text(
            f"Agent: {response}",
            title=f"Response to {customer_id}"
        )
        
        # Track provenance
        self.signer.add_provenance_entry("handle_query", customer_id)
        
        # Audit log
        self.logs.add_text(
            f"[{datetime.now().isoformat()}] Query from {customer_id}: {query[:50]}...",
            title="Query Log"
        )
        
        return response
    
    def save_all(self):
        """Save all agent data."""
        self.knowledge.save(f"{self.base_path}/knowledge.maif")
        self.conversations.save(f"{self.base_path}/conversations.maif")
        self.logs.save(f"{self.base_path}/logs.maif")

# Usage
agent = CustomerServiceAgent()

# Load FAQs
agent.load_faq([
    {"question": "How do I return an item?", "answer": "Returns are accepted within 30 days..."},
    {"question": "What are your hours?", "answer": "We're open 9am-5pm Monday-Friday..."}
])

# Handle queries
response = agent.handle_query("cust_123", "How can I return my order?")
print(response)

# Save state
agent.save_all()
```

## Best Practices

1. **Use MAIF for persistent memory** - Survives restarts
2. **Enable privacy for sensitive data** - Protect customer information
3. **Track provenance** - Audit trail for compliance
4. **Separate concerns** - Different artifacts for different purposes
5. **Regular saves** - Persist state frequently

## Next Steps

- **[Agent Lifecycle →](/guide/agent-lifecycle)** - Managing agent lifecycle
- **[LangGraph Example →](/examples/langgraph-rag)** - RAG with MAIF
- **[API Reference →](/api/)** - Complete documentation
