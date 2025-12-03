# Hello World Agent

The simplest possible MAIF agent.

## Code

```python
from maif_api import create_maif

# Create agent with memory
maif = create_maif("hello-agent")

# Add content
maif.add_text("Hello, MAIF world!", title="Greeting")

# Save with cryptographic signing
maif.save("hello.maif", sign=True)

print("✅ Your first AI agent memory is ready!")
```

## Running the Example

```bash
# Navigate to repository
cd maifscratch-1

# Install MAIF
pip install -e .

# Run
python3 -c "
from maif_api import create_maif

maif = create_maif('hello-agent')
maif.add_text('Hello, MAIF world!', title='Greeting')
maif.save('hello.maif', sign=True)

print('✅ Your first AI agent memory is ready!')
"
```

Or run the basic example:

```bash
python3 examples/basic/simple_api_demo.py
```

## Explanation

1. **`create_maif("hello-agent")`**: Creates a new MAIF artifact with the agent ID "hello-agent"
2. **`add_text(...)`**: Adds text content with an optional title
3. **`save(..., sign=True)`**: Persists the artifact to disk with cryptographic signing

## What Happens Under the Hood

When you create and save a MAIF artifact:

1. **Block Creation**: Content is stored as a "block" with metadata
2. **Hash Chain**: Each block is linked to the previous via SHA-256 hash
3. **Digital Signature**: The entire artifact is signed for authenticity
4. **File Creation**: Binary `.maif` file is written to disk

## Loading the Artifact

```python
from maif_api import load_maif

# Load the artifact
loaded = load_maif("hello.maif")

# Verify integrity (tamper detection)
if loaded.verify_integrity():
    print("✅ Artifact is authentic and unmodified")

# Get content
content = loaded.get_content_list()
for item in content:
    print(f"Title: {item.get('title')}")
    print(f"Type: {item.get('type')}")
```

## Next Steps

- [Privacy Demo](./privacy-demo.md) - Add encryption and anonymization
- [Multi-Agent](./multi-agent.md) - Multiple agents sharing artifacts
- [LangGraph RAG](./langgraph-rag.md) - Production multi-agent system
