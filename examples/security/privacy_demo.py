#!/usr/bin/env python3
"""
Privacy-by-Design MAIF Demo

Demonstrates privacy features using the secure MAIF format with:
- Ed25519 signatures (64 bytes per block)
- Self-contained files (no external manifest)
- Embedded provenance chain
- Privacy primitives (differential privacy, ZKP, SMC)
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from maif import MAIFEncoder, MAIFDecoder, MAIFParser
from maif.privacy import (
    PrivacyEngine,
    PrivacyPolicy,
    PrivacyLevel,
    EncryptionMode,
    DifferentialPrivacy,
    SecureMultipartyComputation,
    ZeroKnowledgeProof,
)
import time
import json


def demo_privacy_engine():
    """Demonstrate the PrivacyEngine features."""
    print("=== Privacy Engine Demo ===")

    # Create a privacy engine with custom settings
    privacy_engine = PrivacyEngine()

    # Test data with PII
    test_data = [
        "John Smith works at john.smith@company.com and his phone is 555-123-4567",
        "Contact Sarah Johnson at sarah.j@email.com for project updates",
        "The meeting with Michael Brown (michael.brown@corp.com) is at 2 PM",
    ]

    print("\n📋 Original data:")
    for i, data in enumerate(test_data, 1):
        print(f"  {i}. {data}")

    print("\n🔒 Anonymized data:")
    for i, data in enumerate(test_data, 1):
        anonymized = privacy_engine.anonymize_data(data, "demo_context")
        print(f"  {i}. {anonymized}")

    # Show anonymization mapping
    print(f"\n📊 Anonymization mapping:")
    mapping = privacy_engine.anonymization_maps.get("demo_context", {})
    for original, anonymized in list(mapping.items())[:5]:  # Show first 5
        print(f"  '{original}' -> '{anonymized}'")

    return privacy_engine


def demo_secure_maif_with_privacy():
    """Demonstrate creating MAIF files with privacy-aware content."""
    print("\n=== Secure MAIF with Privacy Demo ===")

    # Create a secure MAIF file (uses Ed25519)
    encoder = MAIFEncoder("privacy_secure.maif", agent_id="privacy_agent")

    # Add blocks with different sensitivity levels
    print("\n📝 Adding blocks with different sensitivity levels...")

    # Public data
    public_text = "This is public information that everyone can see."
    encoder.add_text_block(public_text, {"sensitivity": "public"})
    print("  ✓ Added public text block")

    # Confidential data - would normally be encrypted
    confidential_text = "Employee performance data: meeting all targets."
    encoder.add_text_block(confidential_text, {"sensitivity": "confidential"})
    print("  ✓ Added confidential text block")

    # Add some embeddings
    embeddings = [[0.1, 0.2, 0.3, 0.4] * 32 for _ in range(3)]  # 128-dim
    encoder.add_embeddings_block(embeddings, {"model": "privacy-aware-bert"})
    print("  ✓ Added embeddings block")

    # Finalize (signs with Ed25519)
    encoder.finalize()
    print(f"\n✓ Created secure MAIF: privacy_secure.maif")
    print("  (Self-contained with Ed25519 signatures)")

    # Verify the file
    decoder = MAIFDecoder("privacy_secure.maif")
    is_valid, errors = decoder.verify_integrity()
    print(f"  Integrity: {'✓ Valid' if is_valid else '✗ Invalid'}")

    return "privacy_secure.maif"


def demo_differential_privacy():
    """Demonstrate differential privacy features."""
    print("\n=== Differential Privacy Demo ===")

    # Create DP engine with different epsilon values
    dp_low = DifferentialPrivacy(epsilon=0.1)  # High privacy
    dp_med = DifferentialPrivacy(epsilon=1.0)  # Medium privacy
    dp_high = DifferentialPrivacy(epsilon=10.0)  # Low privacy

    original_value = 100.0

    print(f"\n📊 Original value: {original_value}")
    print("\nNoisy values with different epsilon (privacy levels):")

    for name, dp in [
        ("ε=0.1 (high privacy)", dp_low),
        ("ε=1.0 (medium)", dp_med),
        ("ε=10.0 (low privacy)", dp_high),
    ]:
        noisy = dp.add_noise(original_value)
        print(f"  {name}: {noisy:.2f}")

    # Vector noise demonstration
    print("\n📈 Vector noise (ε=1.0):")
    original_vector = [1.0, 2.0, 3.0, 4.0, 5.0]
    noisy_vector = dp_med.add_noise_to_vector(original_vector)
    print(f"  Original: {original_vector}")
    print(f"  Noisy:    {[f'{v:.2f}' for v in noisy_vector]}")


def demo_secure_multiparty():
    """Demonstrate secure multiparty computation."""
    print("\n=== Secure Multiparty Computation Demo ===")

    smc = SecureMultipartyComputation()

    # Secret sharing
    secret_value = 42
    shares = smc.secret_share(secret_value, num_parties=3)

    print(f"\n🔐 Secret value: {secret_value}")
    print(f"📤 Secret shares distributed to 3 parties:")
    for i, share in enumerate(shares, 1):
        print(f"  Party {i}: {share}")

    # Reconstruction
    reconstructed = smc.reconstruct_secret(shares)
    print(f"\n📥 Reconstructed value: {reconstructed}")
    print(f"  Match: {'✓' if reconstructed == secret_value else '✗'}")


def demo_zero_knowledge():
    """Demonstrate zero-knowledge proofs."""
    print("\n=== Zero-Knowledge Proofs Demo ===")

    zkp = ZeroKnowledgeProof()

    # Commit to a secret
    secret_data = b"This is my secret password"
    commitment = zkp.commit(secret_data)

    print(f"\n🔒 Secret: {secret_data.decode()}")
    print(f"📝 Commitment: {commitment.hex()[:32]}...")

    # Verify commitment (prover proves knowledge without revealing)
    nonce = zkp.commitments[list(zkp.commitments.keys())[0]]
    is_valid = zkp.verify_commitment(commitment, secret_data, nonce)
    print(f"\n✓ Commitment verification: {'Valid' if is_valid else 'Invalid'}")

    # Try with wrong secret
    wrong_secret = b"Wrong password"
    is_invalid = zkp.verify_commitment(commitment, wrong_secret, nonce)
    print(
        f"✗ Wrong secret verification: {'Valid (BAD!)' if is_invalid else 'Invalid (correct)'}"
    )


def demo_privacy_policies():
    """Demonstrate privacy policy usage."""
    print("\n=== Privacy Policies Demo ===")

    # Define different privacy policies
    policies = {
        "Public": PrivacyPolicy(
            privacy_level=PrivacyLevel.PUBLIC,
            encryption_mode=EncryptionMode.NONE,
            anonymization_required=False,
        ),
        "Internal": PrivacyPolicy(
            privacy_level=PrivacyLevel.INTERNAL,
            encryption_mode=EncryptionMode.AES_GCM,
            anonymization_required=False,
            retention_period=90,
        ),
        "Confidential": PrivacyPolicy(
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            encryption_mode=EncryptionMode.AES_GCM,
            anonymization_required=True,
            audit_required=True,
            retention_period=365,
        ),
        "Secret": PrivacyPolicy(
            privacy_level=PrivacyLevel.SECRET,
            encryption_mode=EncryptionMode.CHACHA20_POLY1305,
            anonymization_required=True,
            audit_required=True,
        ),
    }

    print("\n📋 Privacy Policy Configurations:")
    for name, policy in policies.items():
        print(f"\n  {name}:")
        print(f"    Level: {policy.privacy_level.name}")
        print(f"    Encryption: {policy.encryption_mode.name}")
        print(f"    Anonymize: {policy.anonymization_required}")
        print(f"    Audit: {policy.audit_required}")
        if hasattr(policy, "retention_period") and policy.retention_period:
            print(f"    Retention: {policy.retention_period} days")


def main():
    """Run all privacy demos."""
    print("=" * 60)
    print("MAIF Privacy-by-Design Comprehensive Demo")
    print("=" * 60)

    try:
        # Run all demos
        demo_privacy_engine()
        demo_secure_maif_with_privacy()
        demo_differential_privacy()
        demo_secure_multiparty()
        demo_zero_knowledge()
        demo_privacy_policies()

        print("\n" + "=" * 60)
        print("Privacy Demo Complete!")
        print("=" * 60)

        print("\nKey Features Demonstrated:")
        print("  ✓ Ed25519 signatures for secure MAIF files")
        print("  ✓ Automatic data anonymization (PII detection)")
        print("  ✓ Differential privacy for statistical queries")
        print("  ✓ Secure multiparty computation (secret sharing)")
        print("  ✓ Zero-knowledge proof commitments")
        print("  ✓ Privacy policy configurations")

        print("\nFiles created:")
        print("  - privacy_secure.maif")

    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
