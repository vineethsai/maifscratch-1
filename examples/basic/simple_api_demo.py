"""
MAIF Simple API Demo

This example shows how easy it is to use the new MAIF API for common tasks.
The MAIF class provides a high-level interface that handles all the complexity
of the self-contained secure format.
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from maif_api import (
    MAIF,
    create_maif,
    load_maif,
    quick_text_maif,
    quick_multimodal_maif,
)


def demo_basic_usage():
    """Demonstrate basic MAIF API usage."""
    print("🚀 MAIF Simple API Demo")
    print("=" * 40)

    # 1. Create a new MAIF using the MAIF class
    print("\n1. Creating new MAIF...")
    maif = MAIF("demo_agent")  # Use MAIF class, not create_maif

    # 2. Add different types of content
    print("2. Adding content...")

    # Add text - add_text returns self for chaining
    maif.add_text("This is a sample document about AI and machine learning.")
    print("   ✅ Added text block")

    # Add more text content
    maif.add_text("MAIF provides secure, verifiable storage for AI artifacts.")
    print("   ✅ Added second text block")

    # Add embeddings (sample 384-dimensional vectors)
    sample_embeddings = [
        [0.1, 0.2, 0.3, 0.4] * 96,  # 384-dimensional
        [0.5, 0.6, 0.7, 0.8] * 96,  # 384-dimensional
    ]
    maif.add_embeddings(sample_embeddings)
    print("   ✅ Added embeddings block")

    # 3. Save the MAIF (self-contained, no manifest needed)
    print("\n3. Saving MAIF...")
    output_path = "simple_demo.maif"
    if maif.save(output_path):
        print(f"   ✅ MAIF saved to {output_path}")
    else:
        print("   ❌ Failed to save MAIF!")
        return False

    # 4. Load and verify
    print("\n4. Loading and verifying MAIF...")
    loaded_maif = load_maif(output_path)

    if loaded_maif.verify():
        print("   ✅ MAIF integrity verified!")
    else:
        print("   ❌ MAIF integrity check failed!")
        return False

    # 5. Show content summary
    print("\n5. Content Summary:")
    print(f"   Agent: {loaded_maif.agent_id}")
    print(f"   Texts: {len(loaded_maif.texts)}")

    for i, text in enumerate(loaded_maif.texts, 1):
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"      [{i}] {preview}")

    return True


def demo_quick_functions():
    """Demonstrate quick convenience functions."""
    print("\n⚡ Quick Functions Demo")
    print("=" * 40)

    # Quick text MAIF - one-liner creation
    result = quick_text_maif(
        "This is a quick text document created with one function call!",
        "quick_text.maif",
    )
    if result:
        print(f"   ✅ Quick text MAIF created: {result}")

    # Quick multimodal MAIF - create with text and embeddings
    result = quick_multimodal_maif(
        texts=["Quick multimodal content", "Second text block"],
        embeddings=[[0.1] * 128, [0.2] * 128],  # 128-dimensional embeddings
        output_path="quick_multimodal.maif",
    )
    if result:
        print(f"   ✅ Quick multimodal MAIF created: {result}")

    return True


def demo_create_maif_helper():
    """Demonstrate the create_maif helper function."""
    print("\n📦 Create MAIF Helper Demo")
    print("=" * 40)

    # create_maif is a convenience function that creates and saves in one call
    result = create_maif(
        output_path="helper_demo.maif",
        texts=[
            "First document using the helper function.",
            "Second document with more content.",
        ],
        embeddings=[[0.5] * 64, [0.6] * 64],  # 64-dimensional embeddings
        agent_id="helper_agent",
    )

    if result:
        print(f"   ✅ Created MAIF using helper: {result}")

        # Verify it worked
        loaded = load_maif(result)
        print(f"   📋 Loaded {len(loaded.texts)} text blocks")

        return True

    return False


if __name__ == "__main__":
    try:
        success = True

        # Run all demos
        success = demo_basic_usage() and success
        success = demo_quick_functions() and success
        success = demo_create_maif_helper() and success

        if success:
            print("\n🎉 All demos completed successfully!")
        else:
            print("\n⚠️  Some demos had issues")

        print("\nFiles created:")
        print("  - simple_demo.maif")
        print("  - quick_text.maif")
        print("  - quick_multimodal.maif")
        print("  - helper_demo.maif")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
