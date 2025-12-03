"""
Enhanced KB Creation Script with REAL embeddings and vector DB.

This script:
1. Creates MAIF artifacts for documents
2. Generates REAL embeddings using sentence-transformers
3. Stores embeddings in MAIF artifacts
4. Indexes everything in ChromaDB for semantic search
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from examples.langgraph.maif_utils import KBManager
from examples.langgraph.vector_db import get_vector_db


def create_sample_documents():
    """Create sample documents about climate change for demo."""
    
    documents = {
        "doc_001": {
            "metadata": {
                "title": "Climate Change Causes",
                "author": "Climate Science Institute",
                "publication_date": "2024",
                "source": "Scientific Report",
                "domain": "environmental_science"
            },
            "chunks": [
                {
                    "text": "Climate change is primarily caused by greenhouse gas emissions from human activities. The burning of fossil fuels like coal, oil, and natural gas for energy and transportation releases large amounts of carbon dioxide (CO2) into the atmosphere. These emissions trap heat in the Earth's atmosphere through the greenhouse effect, leading to a gradual increase in global average temperatures.",
                    "metadata": {"topic": "fossil_fuels", "section": "causes", "confidence": "high"}
                },
                {
                    "text": "Deforestation is another major contributor to climate change. Trees absorb CO2 from the atmosphere through photosynthesis, and when forests are cleared or burned, this stored carbon is released back into the air as CO2. Additionally, deforestation reduces the planet's capacity to absorb future emissions, creating a double negative effect on climate stability.",
                    "metadata": {"topic": "deforestation", "section": "causes", "confidence": "high"}
                },
                {
                    "text": "Industrial processes, including manufacturing, cement production, and chemical production, contribute significantly to greenhouse gas emissions. These sectors are responsible for releasing not only CO2 but also other potent greenhouse gases like methane (CH4), nitrous oxide (N2O), and fluorinated gases. The industrial sector accounts for approximately 21% of global greenhouse gas emissions.",
                    "metadata": {"topic": "industry", "section": "causes", "confidence": "high"}
                },
                {
                    "text": "Agriculture and livestock farming produce substantial amounts of methane, a greenhouse gas that is 25-30 times more potent than CO2 in trapping heat over a 100-year period. Rice paddies, cattle ranching, and manure management are major sources of agricultural emissions. Nitrous oxide from fertilizers also contributes significantly to the problem.",
                    "metadata": {"topic": "agriculture", "section": "causes", "confidence": "high"}
                },
                {
                    "text": "Transportation, particularly cars, trucks, ships, and airplanes powered by fossil fuels, is one of the fastest-growing sources of greenhouse gas emissions globally. The transportation sector accounts for about 24% of direct CO2 emissions from fuel combustion. Electric vehicles and sustainable aviation fuels are emerging as potential solutions.",
                    "metadata": {"topic": "transportation", "section": "causes", "confidence": "high"}
                }
            ]
        },
        "doc_002": {
            "metadata": {
                "title": "Scientific Consensus on Climate Change",
                "author": "International Panel on Climate Change (IPCC)",
                "publication_date": "2024",
                "source": "IPCC Sixth Assessment Report",
                "domain": "climate_science"
            },
            "chunks": [
                {
                    "text": "The scientific consensus on climate change is overwhelming and unequivocal. Multiple independent studies and data from international bodies like the IPCC indicate that human activity is the dominant cause of observed warming since the mid-20th century. More than 97% of actively publishing climate scientists agree that climate-warming trends over the past century are extremely likely due to human activities.",
                    "metadata": {"topic": "consensus", "section": "evidence", "confidence": "very_high"}
                },
                {
                    "text": "Global average temperatures have increased by approximately 1.1°C (2°F) since pre-industrial times (1850-1900). This warming is unprecedented in at least the last 2,000 years and is directly correlated with the increase in atmospheric CO2 concentrations from about 280 parts per million (ppm) in pre-industrial times to over 420 ppm today.",
                    "metadata": {"topic": "temperature", "section": "evidence", "confidence": "very_high"}
                },
                {
                    "text": "The evidence for anthropogenic climate change comes from multiple independent lines of research including ice core samples showing historical CO2 levels and temperatures, satellite measurements of Earth's energy balance, ocean heat content data showing warming oceans, observations of melting ice sheets and glaciers, and biological indicators like changes in species migration patterns and flowering times.",
                    "metadata": {"topic": "evidence", "section": "research", "confidence": "very_high"}
                },
                {
                    "text": "Climate models, which are based on fundamental physical principles including thermodynamics and fluid dynamics, consistently project continued warming if greenhouse gas emissions continue at current rates. These models have been validated against historical data and have proven remarkably accurate in their predictions. Future projections suggest warming could reach 1.5°C above pre-industrial levels as early as 2030-2035.",
                    "metadata": {"topic": "models", "section": "projections", "confidence": "high"}
                }
            ]
        },
        "doc_003": {
            "metadata": {
                "title": "Climate Change Mitigation Strategies",
                "author": "Global Environment Foundation",
                "publication_date": "2024",
                "source": "Policy and Solutions Report",
                "domain": "environmental_policy"
            },
            "chunks": [
                {
                    "text": "Transitioning to renewable energy sources is one of the most effective strategies for mitigating climate change. Solar photovoltaic, wind, hydroelectric, and geothermal energy can replace fossil fuels without producing greenhouse gas emissions. The cost of renewable energy has decreased dramatically, with solar and wind now being the cheapest sources of electricity in many regions.",
                    "metadata": {"topic": "renewable_energy", "section": "solutions", "confidence": "high"}
                },
                {
                    "text": "Improving energy efficiency in buildings, transportation, and industry can significantly reduce emissions while also lowering costs. Technologies like LED lighting, smart thermostats, improved insulation, heat pumps, and electric vehicles offer immediate opportunities for emission reductions. Energy efficiency improvements could reduce global energy demand by 40% by 2050.",
                    "metadata": {"topic": "efficiency", "section": "solutions", "confidence": "high"}
                },
                {
                    "text": "Protecting and restoring forests is crucial for climate mitigation. Forests act as carbon sinks, absorbing approximately 7.6 billion metric tons of CO2 annually. Reforestation and afforestation programs, along with preventing deforestation through sustainable land management, can have immediate and long-term benefits. Natural climate solutions could provide up to 37% of the emissions reductions needed by 2030.",
                    "metadata": {"topic": "forests", "section": "solutions", "confidence": "high"}
                },
                {
                    "text": "Changing agricultural practices can reduce emissions while maintaining or increasing food production. Conservation agriculture, precision farming, improved livestock management, and reducing food waste can significantly lower the sector's carbon footprint. Practices like cover cropping and reduced tillage also increase soil carbon sequestration.",
                    "metadata": {"topic": "agriculture", "section": "solutions", "confidence": "medium"}
                },
                {
                    "text": "Carbon capture and storage (CCS) technologies can remove CO2 from industrial processes and potentially from the atmosphere through direct air capture. While these technologies are still developing and face economic and technical challenges, they could play an important role in achieving net-zero emissions targets, particularly for hard-to-decarbonize sectors like cement and steel production.",
                    "metadata": {"topic": "carbon_capture", "section": "solutions", "confidence": "medium"}
                }
            ]
        }
    }
    
    return documents


def main():
    """Create KB artifacts with REAL embeddings and ChromaDB indexing."""
    print("\n" + "="*80)
    print("🚀 ENHANCED Knowledge Base Creation")
    print("   - Real embeddings (sentence-transformers)")
    print("   - ChromaDB vector database")
    print("   - MAIF artifact storage")
    print("="*80 + "\n")
    
    # Initialize managers
    kb_manager = KBManager(kb_dir="examples/langgraph/data/kb")
    
    print("🧠 Initializing Vector Database...")
    vector_db = get_vector_db()
    
    # Clear existing data (optional - comment out to preserve)
    print("🗑️  Clearing existing vector DB...")
    vector_db.clear()
    
    # Get documents
    documents = create_sample_documents()
    
    created_paths = {}
    
    for doc_id, doc_data in documents.items():
        print("\n" + "-"*80)
        print(f"📄 Processing: {doc_id}")
        print(f"   Title: {doc_data['metadata']['title']}")
        print(f"   Chunks: {len(doc_data['chunks'])}")
        print("-"*80)
        
        try:
            # Add to vector database (with real embeddings!)
            vector_db.add_documents(
                doc_id=doc_id,
                chunks=doc_data['chunks'],
                document_metadata=doc_data['metadata']
            )
            
            # Get embeddings from vector DB for MAIF storage
            print(f"   📦 Retrieving embeddings for MAIF storage...")
            embeddings = vector_db.generate_embeddings(
                [chunk['text'] for chunk in doc_data['chunks']],
                show_progress=False
            )
            
            # Add embeddings to chunks
            chunks_with_embeddings = []
            for i, chunk in enumerate(doc_data['chunks']):
                chunk_copy = chunk.copy()
                chunk_copy['embedding'] = embeddings[i].tolist()
                chunks_with_embeddings.append(chunk_copy)
            
            # Create MAIF artifact with embeddings
            print(f"   💾 Creating MAIF artifact with embeddings...")
            kb_path = kb_manager.create_kb_artifact(
                doc_id=doc_id,
                chunks=chunks_with_embeddings,
                document_metadata=doc_data['metadata']
            )
            
            created_paths[doc_id] = kb_path
            print(f"   ✅ MAIF artifact created: {kb_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ ENHANCED Knowledge Base Creation Complete!")
    print("="*80 + "\n")
    
    print("📊 Vector Database Stats:")
    stats = vector_db.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n📋 Created KB Artifacts (with embeddings):")
    for doc_id, path in created_paths.items():
        print(f"   {doc_id}: {path}")
        # Show file size
        if Path(path).exists():
            size = Path(path).stat().st_size
            print(f"      Size: {size:,} bytes")
    
    print(f"\n🎉 Success! Your KB now has:")
    print(f"   ✅ Real semantic embeddings")
    print(f"   ✅ ChromaDB vector database")
    print(f"   ✅ MAIF provenance artifacts")
    
    print(f"\n💡 Next step:")
    print(f"   Run: python3 demo_enhanced.py")


if __name__ == "__main__":
    main()

