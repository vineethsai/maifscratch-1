"""
MAIF Simple API Demo

This example shows how easy it is to use the new MAIF API for common tasks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maif_api import create_maif, load_maif, quick_text_maif, quick_multimodal_maif

def demo_basic_usage():
    """Demonstrate basic MAIF API usage."""
    print("🚀 MAIF Simple API Demo")
    print("=" * 40)
    
    # 1. Create a new MAIF
    print("\n1. Creating new MAIF...")
    maif = create_maif("demo_agent")
    
    # 2. Add different types of content
    print("2. Adding content...")
    
    # Add text
    text_id = maif.add_text(
        "This is a sample document about AI and machine learning.",
        title="AI Document"
    )
    print(f"   ✅ Added text: {text_id}")
    
    # Add multimodal content with ACAM processing
    multimodal_id = maif.add_multimodal({
        "text": "A beautiful mountain landscape at sunset",
        "image_description": "Scenic photography with warm colors",
        "location": "Rocky Mountains, Colorado"
    }, title="Mountain Sunset")
    print(f"   ✅ Added multimodal content: {multimodal_id}")
    
    # Add embeddings
    sample_embeddings = [
        [0.1, 0.2, 0.3, 0.4] * 96,  # 384-dimensional
        [0.5, 0.6, 0.7, 0.8] * 96   # 384-dimensional
    ]
    embedding_id = maif.add_embeddings(
        sample_embeddings,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        compress=True
    )
    print(f"   ✅ Added compressed embeddings: {embedding_id}")
    
    # 3. Save the MAIF
    print("\n3. Saving MAIF...")
    if maif.save("simple_demo.maif"):
        print("   ✅ MAIF saved successfully!")
    else:
        print("   ❌ Failed to save MAIF!")
        return
    
    # 4. Load and verify
    print("\n4. Loading and verifying MAIF...")
    loaded_maif = load_maif("simple_demo.maif")
    
    if loaded_maif.verify_integrity():
        print("   ✅ MAIF integrity verified!")
    else:
        print("   ❌ MAIF integrity check failed!")
        return
    
    # 5. Show content summary
    print("\n5. Content Summary:")
    content_list = loaded_maif.get_content_list()
    for i, content in enumerate(content_list, 1):
        title = content.get('title', 'Untitled')
        content_type = content.get('type', 'unknown')
        print(f"   {i}. {title} ({content_type})")
    
    print(f"\n📊 Total content blocks: {len(content_list)}")

def demo_privacy_features():
    """Demonstrate privacy and security features."""
    print("\n🔒 Privacy & Security Demo")
    print("=" * 40)
    
    # Create MAIF with privacy enabled
    secure_maif = create_maif("secure_agent", enable_privacy=True)
    
    # Add encrypted content
    secure_maif.add_text(
        "This is sensitive information that should be encrypted.",
        title="Confidential Document",
        encrypt=True,
        anonymize=True
    )
    
    # Save with signing
    if secure_maif.save("secure_demo.maif", sign=True):
        print("✅ Secure MAIF created with encryption and signing!")
        
        # Get privacy report
        privacy_report = secure_maif.get_privacy_report()
        print(f"📋 Privacy Report: {privacy_report}")
    else:
        print("❌ Failed to create secure MAIF!")

def demo_quick_functions():
    """Demonstrate quick convenience functions."""
    print("\n⚡ Quick Functions Demo")
    print("=" * 40)
    
    # Quick text MAIF
    if quick_text_maif(
        "This is a quick text document created with one function call!",
        "quick_text.maif",
        title="Quick Text Demo"
    ):
        print("✅ Quick text MAIF created!")
    
    # Quick multimodal MAIF
    if quick_multimodal_maif({
        "text": "Quick multimodal content",
        "description": "Created with a single function call",
        "category": "demo"
    }, "quick_multimodal.maif", title="Quick Multimodal Demo"):
        print("✅ Quick multimodal MAIF created!")

def demo_advanced_features():
    """Demonstrate advanced features like search."""
    print("\n🔍 Advanced Features Demo")
    print("=" * 40)
    
    # Create MAIF with searchable content
    search_maif = create_maif("search_agent")
    
    # Add multiple text documents
    documents = [
        ("Machine learning is a subset of artificial intelligence.", "ML Basics"),
        ("Deep learning uses neural networks with multiple layers.", "Deep Learning"),
        ("Natural language processing helps computers understand text.", "NLP Overview"),
        ("Computer vision enables machines to interpret visual information.", "Computer Vision")
    ]
    
    for text, title in documents:
        search_maif.add_text(text, title=title)
    
    # Save and load for searching
    if search_maif.save("searchable_demo.maif"):
        print("✅ Searchable MAIF created!")
        
        # Load for searching
        loaded_search_maif = load_maif("searchable_demo.maif")
        
        # Perform search
        try:
            results = loaded_search_maif.search("neural networks", top_k=2)
            print(f"🔍 Search results for 'neural networks': {len(results)} found")
            for result in results:
                print(f"   - {result}")
        except Exception as e:
            print(f"⚠️  Search not available: {e}")

if __name__ == "__main__":
    try:
        # Run all demos
        demo_basic_usage()
        demo_privacy_features()
        demo_quick_functions()
        demo_advanced_features()
        
        print("\n🎉 All demos completed successfully!")
        print("\nFiles created:")
        print("  - simple_demo.maif")
        print("  - secure_demo.maif") 
        print("  - quick_text.maif")
        print("  - quick_multimodal.maif")
        print("  - searchable_demo.maif")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()