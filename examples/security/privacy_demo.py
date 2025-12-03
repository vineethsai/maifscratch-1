#!/usr/bin/env python3
"""
Privacy-by-Design MAIF Demo
Demonstrates comprehensive privacy features including encryption, access controls, and anonymization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maif.core import MAIFEncoder, MAIFDecoder
from maif.privacy import (
    PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode, 
    AccessRule, DifferentialPrivacy, SecureMultipartyComputation, ZeroKnowledgeProof
)
import time
import json

def demo_basic_privacy():
    """Demonstrate basic privacy-by-design features."""
    print("=== Basic Privacy-by-Design Demo ===")
    
    # Create encoder with privacy enabled
    encoder = MAIFEncoder(agent_id="privacy_demo_agent", enable_privacy=True)
    
    # Set up different privacy policies
    public_policy = PrivacyPolicy(
        privacy_level=PrivacyLevel.PUBLIC,
        encryption_mode=EncryptionMode.NONE,
        anonymization_required=False
    )
    
    confidential_policy = PrivacyPolicy(
        privacy_level=PrivacyLevel.CONFIDENTIAL,
        encryption_mode=EncryptionMode.AES_GCM,
        anonymization_required=True,
        retention_period=365  # 1 year
    )
    
    secret_policy = PrivacyPolicy(
        privacy_level=PrivacyLevel.SECRET,
        encryption_mode=EncryptionMode.CHACHA20_POLY1305,
        anonymization_required=True,
        audit_required=True
    )
    
    # Add blocks with different privacy levels
    print("Adding public data...")
    public_text = "This is public information that everyone can see."
    encoder.add_text_block(public_text, {"description": "Public announcement"}, 
                          privacy_policy=public_policy)
    
    print("Adding confidential data with anonymization...")
    confidential_text = "Employee John Smith from john.smith@company.com worked on project Alpha."
    encoder.add_text_block(confidential_text, {"description": "HR record"}, 
                          privacy_policy=confidential_policy, anonymize=True)
    
    print("Adding secret data with strong encryption...")
    secret_text = "The launch codes are: ALPHA-BRAVO-CHARLIE-123456"
    encoder.add_text_block(secret_text, {"description": "Classified information"}, 
                          privacy_policy=secret_policy)
    
    # Add some embeddings with privacy
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    encoder.add_embeddings_block(embeddings, {"model": "secure-bert"}, 
                                privacy_policy=confidential_policy)
    
    # Set up access control rules
    print("\nSetting up access control rules...")
    
    # Public access for everyone
    encoder.add_access_rule("*", "*", ["read"], conditions={"privacy_level": "public"})
    
    # Confidential access for authorized users
    encoder.add_access_rule("authorized_user", "*", ["read", "write"], 
                           conditions={"privacy_level": "confidential"})
    
    # Secret access only for admin
    encoder.add_access_rule("admin", "*", ["read", "write", "delete"], 
                           conditions={"privacy_level": "secret"})
    
    # Time-limited access
    expiry_time = time.time() + 3600  # 1 hour from now
    encoder.add_access_rule("temp_user", "*", ["read"], expiry=expiry_time)
    
    # Save the MAIF file
    encoder.save("privacy_demo.maif", "privacy_demo_manifest.json")
    
    # Generate privacy report
    privacy_report = encoder.get_privacy_report()
    print(f"\nPrivacy Report:")
    print(json.dumps(privacy_report, indent=2))
    
    return encoder.privacy_engine

def demo_access_control():
    """Demonstrate access control with different user roles."""
    print("\n=== Access Control Demo ===")
    
    # Test access as different users
    users = [
        ("anonymous", "Anonymous user"),
        ("authorized_user", "Authorized employee"),
        ("admin", "System administrator"),
        ("temp_user", "Temporary contractor")
    ]
    
    for user_id, description in users:
        print(f"\n--- Testing access as {description} ({user_id}) ---")
        
        # Create decoder with specific user context
        privacy_engine = PrivacyEngine()  # In real scenario, this would be shared
        decoder = MAIFDecoder("privacy_demo.maif", "privacy_demo_manifest.json", 
                             privacy_engine=privacy_engine, requesting_agent=user_id)
        
        # Try to access different types of data
        accessible_blocks = decoder.get_accessible_blocks("read")
        print(f"Accessible blocks: {len(accessible_blocks)}")
        
        # Try to get text blocks
        try:
            texts = decoder.get_text_blocks()
            print(f"Retrieved {len(texts)} text blocks")
            for i, text in enumerate(texts):
                print(f"  Text {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")
        except Exception as e:
            print(f"Error accessing text blocks: {e}")
        
        # Get privacy summary
        privacy_summary = decoder.get_privacy_summary()
        # Calculate access ratio from available data
        total_blocks = privacy_summary.get('total_blocks', 0)
        access_controlled = privacy_summary.get('access_controlled_blocks', 0)
        if total_blocks > 0:
            access_ratio = access_controlled / total_blocks
            print(f"Access ratio: {access_ratio:.2%}")
        else:
            print("Access ratio: N/A (no blocks)")

def demo_advanced_privacy():
    """Demonstrate advanced privacy features."""
    print("\n=== Advanced Privacy Features Demo ===")
    
    # Differential Privacy
    print("--- Differential Privacy ---")
    dp = DifferentialPrivacy(epsilon=1.0)
    
    original_value = 100.0
    noisy_value = dp.add_noise(original_value)
    print(f"Original value: {original_value}")
    print(f"Noisy value: {noisy_value:.2f}")
    
    # Vector with noise
    original_vector = [1.0, 2.0, 3.0, 4.0, 5.0]
    noisy_vector = dp.add_noise_to_vector(original_vector)
    print(f"Original vector: {original_vector}")
    print(f"Noisy vector: {[f'{v:.2f}' for v in noisy_vector]}")
    
    # Secure Multiparty Computation
    print("\n--- Secure Multiparty Computation ---")
    smc = SecureMultipartyComputation()
    
    secret_value = 42
    shares = smc.secret_share(secret_value, num_parties=3)
    print(f"Secret value: {secret_value}")
    print(f"Secret shares: {shares}")
    
    reconstructed = smc.reconstruct_secret(shares)
    print(f"Reconstructed value: {reconstructed}")
    
    # Zero-Knowledge Proofs
    print("\n--- Zero-Knowledge Proofs ---")
    zkp = ZeroKnowledgeProof()
    
    secret_data = b"This is my secret"
    commitment = zkp.commit(secret_data)
    print(f"Secret data: {secret_data}")
    print(f"Commitment: {commitment.hex()[:32]}...")
    
    # Verify commitment (in real scenario, verifier wouldn't have the secret)
    nonce = zkp.commitments[list(zkp.commitments.keys())[0]]
    is_valid = zkp.verify_commitment(commitment, secret_data, nonce)
    print(f"Commitment verification: {'Valid' if is_valid else 'Invalid'}")

def demo_data_lifecycle():
    """Demonstrate privacy-aware data lifecycle management."""
    print("\n=== Data Lifecycle Management Demo ===")
    
    # Create encoder with privacy
    encoder = MAIFEncoder(agent_id="lifecycle_agent", enable_privacy=True)
    
    # Add data with retention policies
    short_term_policy = PrivacyPolicy(
        privacy_level=PrivacyLevel.INTERNAL,
        encryption_mode=EncryptionMode.AES_GCM,
        retention_period=1,  # 1 day
        audit_required=True
    )
    
    long_term_policy = PrivacyPolicy(
        privacy_level=PrivacyLevel.CONFIDENTIAL,
        encryption_mode=EncryptionMode.CHACHA20_POLY1305,
        retention_period=365,  # 1 year
        audit_required=True
    )
    
    # Add temporary data
    encoder.add_text_block("Temporary session data", 
                          {"type": "session", "created": time.time()},
                          privacy_policy=short_term_policy)
    
    # Add persistent data
    encoder.add_text_block("Important business record", 
                          {"type": "business", "created": time.time()},
                          privacy_policy=long_term_policy)
    
    # Check retention policy enforcement
    expired_blocks = encoder.privacy_engine.enforce_retention_policy()
    print(f"Blocks marked for deletion due to retention policy: {len(expired_blocks)}")
    
    # Generate comprehensive privacy report
    report = encoder.get_privacy_report()
    print("\nComprehensive Privacy Report:")
    print(json.dumps(report, indent=2))

def demo_anonymization():
    """Demonstrate data anonymization capabilities."""
    print("\n=== Data Anonymization Demo ===")
    
    privacy_engine = PrivacyEngine()
    
    # Test data with PII
    test_data = [
        "John Smith works at john.smith@company.com and his phone is 555-123-4567",
        "Contact Sarah Johnson at sarah.j@email.com for project updates",
        "The meeting with Michael Brown (michael.brown@corp.com) is at 2 PM",
        "Invoice #12345 for $1000 sent to client@business.org"
    ]
    
    print("Original data:")
    for i, data in enumerate(test_data, 1):
        print(f"  {i}. {data}")
    
    print("\nAnonymized data:")
    for i, data in enumerate(test_data, 1):
        anonymized = privacy_engine.anonymize_data(data, "demo_context")
        print(f"  {i}. {anonymized}")
    
    # Show anonymization mapping
    print(f"\nAnonymization mapping for 'demo_context':")
    mapping = privacy_engine.anonymization_maps.get("demo_context", {})
    for original, anonymized in mapping.items():
        print(f"  '{original}' -> '{anonymized}'")

def main():
    """Run all privacy demos."""
    print("MAIF Privacy-by-Design Comprehensive Demo")
    print("=" * 50)
    
    try:
        # Run all demos
        privacy_engine = demo_basic_privacy()
        demo_access_control()
        demo_advanced_privacy()
        demo_data_lifecycle()
        demo_anonymization()
        
        print("\n" + "=" * 50)
        print("Privacy-by-Design Demo Complete!")
        print("\nKey Features Demonstrated:")
        print("✓ Multi-level encryption (AES-GCM, ChaCha20-Poly1305)")
        print("✓ Granular access control with role-based permissions")
        print("✓ Automatic data anonymization")
        print("✓ Differential privacy for statistical queries")
        print("✓ Secure multiparty computation primitives")
        print("✓ Zero-knowledge proof commitments")
        print("✓ Data retention policy enforcement")
        print("✓ Comprehensive privacy reporting")
        print("✓ Privacy-aware data lifecycle management")
        
        print(f"\nMAIF is now a true privacy-by-design data container!")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()