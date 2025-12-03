#!/bin/bash
# Quick setup script for LangGraph + MAIF demo

echo "🚀 Setting up LangGraph + MAIF Research Assistant..."
echo ""

# Check if we're in the right directory
if [ ! -f "demo_enhanced.py" ]; then
    echo "❌ Error: Please run this from examples/langgraph directory"
    echo "   cd examples/langgraph && ./RUN_ME_FIRST.sh"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install --user chromadb sentence-transformers langgraph \
             langgraph-checkpoint-sqlite google-generativeai tqdm requests

echo ""
echo "📚 Creating knowledge base with real embeddings..."
python3 create_kb_enhanced.py

echo ""
echo "="*80
echo "✅ Setup complete!"
echo "="*80
echo ""
echo "🎮 Run the interactive demo:"
echo "   python3 demo_enhanced.py"
echo ""
echo "📚 Read the docs:"
echo "   - START_HERE.md (quick overview)"
echo "   - QUICK_START_ENHANCED.md (3-step guide)"
echo ""
echo "🎉 Happy exploring!"

