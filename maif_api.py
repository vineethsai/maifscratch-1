"""
MAIF Simple API - Easy-to-use interface for the existing MAIF library

This API provides a simplified interface to the existing MAIF implementation,
making it easy to create, manipulate, and work with MAIF files.
"""

import os
import json
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

# Import existing MAIF components
from maif.core import MAIFEncoder, MAIFDecoder
from maif.security import MAIFSigner, MAIFVerifier
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode
from maif.semantic_optimized import AdaptiveCrossModalAttention, HierarchicalSemanticCompression


class MAIF:
    """
    Simple MAIF API - One class to rule them all.
    
    Usage:
        # Create a new MAIF
        maif = MAIF("my_agent")
        
        # Add content
        maif.add_text("Hello world!")
        maif.add_image("photo.jpg")
        
        # Save
        maif.save("my_artifact.maif")
        
        # Load existing
        maif = MAIF.load("my_artifact.maif")
    """
    
    def __init__(self, agent_id: str = "default_agent", enable_privacy: bool = False):
        """Initialize MAIF with simple configuration."""
        self.agent_id = agent_id
        self.enable_privacy = enable_privacy
        self._name = f"{agent_id}_artifact"  # Default name
        
        # Initialize components
        self.privacy_engine = PrivacyEngine() if enable_privacy else None
        self.encoder = MAIFEncoder(
            agent_id=agent_id,
            enable_privacy=enable_privacy,
            privacy_engine=self.privacy_engine
        )
        self.signer = MAIFSigner(agent_id=agent_id)
        
        # Track added content
        self.content_blocks = []
    
    @property
    def name(self) -> str:
        """Get the name of this MAIF artifact."""
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set the name of this MAIF artifact."""
        self._name = value
        
    @classmethod
    def load(cls, filepath: str) -> 'MAIF':
        """Load existing MAIF file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MAIF file not found: {filepath}")
            
        # Create instance and load decoder
        instance = cls()
        manifest_path = filepath.replace('.maif', '_manifest.json')
        if os.path.exists(manifest_path):
            instance.decoder = MAIFDecoder(filepath, manifest_path)
            # Populate content_blocks from decoder
            instance._load_content_from_decoder()
        else:
            # Try loading without manifest
            try:
                instance.decoder = MAIFDecoder(filepath, filepath + ".manifest.json")
                instance._load_content_from_decoder()
            except FileNotFoundError:
                # No manifest found, create minimal decoder
                pass
        return instance
    
    def _load_content_from_decoder(self):
        """Load content blocks from decoder into content_blocks list."""
        if not hasattr(self, 'decoder') or self.decoder is None:
            return
            
        for block in self.decoder.blocks:
            # Extract block data
            block_data = self.decoder.get_block_data(block.block_id)
            content_entry = {
                'block_id': block.block_id,
                'block_type': block.block_type,
                'metadata': block.metadata or {},
                'data': block_data
            }
            self.content_blocks.append(content_entry)
    
    def add_text(self, text: str, title: str = None, language: str = "en", 
                 encrypt: bool = False, anonymize: bool = False) -> str:
        """
        Add text content to MAIF.
        
        Args:
            text: Text content to add
            title: Optional title for the text
            language: Language code (default: "en")
            encrypt: Whether to encrypt the text
            anonymize: Whether to anonymize sensitive data
            
        Returns:
            Block ID of the added text
        """
        metadata = {"language": language}
        if title:
            metadata["title"] = title
            
        if encrypt or anonymize:
            block_id = self.encoder.add_text_block(
                text,
                anonymize=anonymize,
                privacy_level=PrivacyLevel.CONFIDENTIAL if encrypt else PrivacyLevel.PUBLIC,
                encryption_mode=EncryptionMode.AES_GCM if encrypt else None,
                metadata=metadata
            )
        else:
            block_id = self.encoder.add_text_block(text, metadata=metadata)
            
        self.content_blocks.append({"type": "text", "id": block_id, "title": title})
        return block_id
    
    def add_image(self, image_path: str, title: str = None, 
                  extract_metadata: bool = True) -> str:
        """
        Add image to MAIF.
        
        Args:
            image_path: Path to image file
            title: Optional title for the image
            extract_metadata: Whether to extract image metadata
            
        Returns:
            Block ID of the added image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, 'rb') as f:
            image_data = f.read()
            
        metadata = {"source_file": os.path.basename(image_path)}
        if title:
            metadata["title"] = title
            
        block_id = self.encoder.add_image_block(
            image_data,
            extract_metadata=extract_metadata,
            metadata=metadata
        )
        
        self.content_blocks.append({"type": "image", "id": block_id, "title": title})
        return block_id
    
    def add_video(self, video_path: str, title: str = None,
                  extract_metadata: bool = True) -> str:
        """
        Add video to MAIF.
        
        Args:
            video_path: Path to video file
            title: Optional title for the video
            extract_metadata: Whether to extract video metadata
            
        Returns:
            Block ID of the added video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        with open(video_path, 'rb') as f:
            video_data = f.read()
            
        metadata = {"source_file": os.path.basename(video_path)}
        if title:
            metadata["title"] = title
            
        block_id = self.encoder.add_video_block(
            video_data,
            extract_metadata=extract_metadata,
            metadata=metadata
        )
        
        self.content_blocks.append({"type": "video", "id": block_id, "title": title})
        return block_id
    
    def add_multimodal(self, content: Dict[str, Any], title: str = None,
                       use_acam: bool = True) -> str:
        """
        Add multimodal content using ACAM processing.
        
        Args:
            content: Dictionary with different modality data
            title: Optional title for the content
            use_acam: Whether to use Adaptive Cross-Modal Attention
            
        Returns:
            Block ID of the added multimodal content
        """
        metadata = {}
        if title:
            metadata["title"] = title
            
        block_id = self.encoder.add_cross_modal_block(
            content,
            use_enhanced_acam=use_acam,
            metadata=metadata
        )
        
        self.content_blocks.append({"type": "multimodal", "id": block_id, "title": title})
        return block_id
    
    def add_embeddings(self, embeddings: List[List[float]], model_name: str = "custom",
                       compress: bool = True) -> str:
        """
        Add embeddings with optional compression.
        
        Args:
            embeddings: List of embedding vectors
            model_name: Name of the model that generated embeddings
            compress: Whether to use HSC compression
            
        Returns:
            Block ID of the added embeddings
        """
        if compress:
            block_id = self.encoder.add_compressed_embeddings_block(
                embeddings,
                metadata={"model": model_name, "compressed": True}
            )
        else:
            block_id = self.encoder.add_embeddings_block(
                embeddings,
                metadata={"model": model_name, "dimensions": len(embeddings[0]) if embeddings else 0}
            )
            
        self.content_blocks.append({"type": "embeddings", "id": block_id, "model": model_name})
        return block_id
    
    def save(self, filepath: str, sign: bool = True) -> bool:
        """
        Save MAIF to file.
        
        Args:
            filepath: Output file path
            sign: Whether to cryptographically sign the file
            
        Returns:
            True if successful
        """
        try:
            # Generate manifest path
            manifest_path = filepath.replace('.maif', '_manifest.json')
            
            # Build MAIF file
            self.encoder.build_maif(filepath, manifest_path)
            
            # Sign if requested
            if sign:
                for block in self.content_blocks:
                    self.signer.add_provenance_entry("create", block["id"])
                    
            return True
            
        except Exception as e:
            print(f"Error saving MAIF: {e}")
            return False
    
    def get_content_list(self) -> List[Dict[str, Any]]:
        """Get list of all content blocks."""
        return self.content_blocks.copy()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search content using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching content with similarity scores
        """
        if not hasattr(self, 'decoder'):
            raise RuntimeError("Cannot search - MAIF not loaded from file")
            
        # Check if decoder has search functionality
        if hasattr(self.decoder, 'search_semantic'):
            results = self.decoder.search_semantic(query, top_k=top_k)
            return results
        else:
            # Fallback: Simple text search in text blocks
            text_blocks = self.decoder.get_text_blocks()
            results = []
            query_lower = query.lower()
            
            for i, text in enumerate(text_blocks):
                if query_lower in text.lower():
                    # Calculate simple relevance score based on frequency
                    score = text.lower().count(query_lower) / len(text.split())
                    results.append({
                        'block_index': i,
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'score': score
                    })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
    
    def verify_integrity(self) -> bool:
        """Verify MAIF file integrity."""
        if not hasattr(self, 'decoder'):
            raise RuntimeError("Cannot verify - MAIF not loaded from file")
            
        return self.decoder.verify_integrity()
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Get privacy report for the MAIF."""
        if not self.enable_privacy:
            return {"privacy_enabled": False}
            
        return self.encoder.get_privacy_report()


# Convenience functions for quick operations
def create_maif(agent_id: str = "default_agent", enable_privacy: bool = False) -> MAIF:
    """Create a new MAIF instance."""
    return MAIF(agent_id=agent_id, enable_privacy=enable_privacy)

def load_maif(filepath: str) -> MAIF:
    """Load existing MAIF file."""
    return MAIF.load(filepath)

def quick_text_maif(text: str, output_path: str, title: str = None) -> bool:
    """Quickly create a MAIF with just text content."""
    maif = create_maif()
    maif.add_text(text, title=title)
    return maif.save(output_path)

def quick_multimodal_maif(content: Dict[str, Any], output_path: str, title: str = None) -> bool:
    """Quickly create a MAIF with multimodal content."""
    maif = create_maif()
    maif.add_multimodal(content, title=title)
    return maif.save(output_path)


# Example usage
if __name__ == "__main__":
    # Simple example
    print("Creating simple MAIF...")
    
    # Create new MAIF
    maif = create_maif("demo_agent")
    
    # Add some content
    maif.add_text("Hello, this is a test document!", title="Test Document")
    maif.add_multimodal({
        "text": "A beautiful sunset",
        "description": "Nature photography"
    }, title="Sunset Scene")
    
    # Save
    if maif.save("demo.maif"):
        print("✅ MAIF saved successfully!")
        
        # Load and verify
        loaded_maif = load_maif("demo.maif")
        if loaded_maif.verify_integrity():
            print("✅ MAIF integrity verified!")
            print(f"Content blocks: {len(loaded_maif.get_content_list())}")
        else:
            print("❌ MAIF integrity check failed!")
    else:
        print("❌ Failed to save MAIF!")