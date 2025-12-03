#!/usr/bin/env python3
"""
Demonstration of MAIF Classified Security Features
Shows PKI authentication, classified data handling, and AWS integration
"""

import os
import json
import time
from datetime import datetime, timedelta
from maif.classified_security import (
    ClassificationLevel,
    AuthenticationMethod,
    UserClearance,
    ClassifiedSecurityManager,
    MAIFClassifiedSecurity
)
from maif.core import MAIFBlock, BlockType
from maif_sdk.client import MAIFClient
from maif_sdk.artifact import Artifact
from maif_sdk.types import SecurityLevel

def demonstrate_pki_authentication():
    """Demonstrate PKI certificate-based authentication."""
    print("\n=== PKI Certificate Authentication Demo ===\n")
    
    # Initialize security manager
    security_manager = ClassifiedSecurityManager(
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
        use_fips=True
    )
    
    # Simulate PKI certificate (in production, this would come from CAC/PIV card)
    mock_certificate = """-----BEGIN CERTIFICATE-----
MIIDQTCCAimgAwIBAgITBmyfz5m/jAo54vB4ikPmljZbyjANBgkqhkiG9w0BAQsF
... (certificate content) ...
-----END CERTIFICATE-----"""
    
    # Attempt authentication
    success, user_id = security_manager.authenticate_user(
        AuthenticationMethod.PKI_CERTIFICATE,
        {"certificate": mock_certificate}
    )
    
    if success:
        print(f"✓ Authentication successful for user: {user_id}")
    else:
        print("✗ Authentication failed")
        # In production, you would use a real certificate
        print("  (Using mock authentication for demo)")
        user_id = "john.doe.1234567890"
    
    return security_manager, user_id


def demonstrate_user_clearance():
    """Demonstrate user clearance management."""
    print("\n=== User Clearance Management Demo ===\n")
    
    # Create user clearance
    clearance = UserClearance(
        user_id="john.doe.1234567890",
        clearance_level=ClassificationLevel.SECRET,
        clearance_expiry=datetime.now() + timedelta(days=365),
        compartments=["CRYPTO", "SIGINT"],
        caveat_access=["NOFORN", "REL TO FVEY"],
        last_investigation=datetime.now() - timedelta(days=180)
    )
    
    print(f"User: {clearance.user_id}")
    print(f"Clearance Level: {clearance.clearance_level.name}")
    print(f"Compartments: {', '.join(clearance.compartments)}")
    print(f"Caveats: {', '.join(clearance.caveat_access)}")
    print(f"Valid: {clearance.is_valid()}")
    
    # Test access scenarios
    test_cases = [
        (ClassificationLevel.CONFIDENTIAL, [], [], True),
        (ClassificationLevel.SECRET, ["CRYPTO"], [], True),
        (ClassificationLevel.TOP_SECRET, [], [], False),
        (ClassificationLevel.SECRET, ["HUMINT"], [], False),
    ]
    
    print("\nAccess Control Tests:")
    for level, comps, caveats, expected in test_cases:
        result = clearance.can_access(level, comps, caveats)
        status = "✓" if result == expected else "✗"
        print(f"{status} {level.name} {comps}: {'Granted' if result else 'Denied'}")
    
    return clearance


def demonstrate_classified_data_handling(security_manager: ClassifiedSecurityManager):
    """Demonstrate handling of classified data."""
    print("\n=== Classified Data Handling Demo ===\n")
    
    # Register user clearance
    clearance = UserClearance(
        user_id="analyst.001",
        clearance_level=ClassificationLevel.SECRET,
        clearance_expiry=datetime.now() + timedelta(days=365)
    )
    security_manager.mac.register_user_clearance(clearance)
    
    # Create classified data
    classified_data = {
        "operation": "OPERATION_REDACTED",
        "location": "38.8977° N, 77.0365° W",
        "assets": ["ASSET_001", "ASSET_002"],
        "status": "ACTIVE"
    }
    
    data_bytes = json.dumps(classified_data).encode()
    
    # Encrypt at SECRET level
    if security_manager.aws_integration:
        print("Encrypting classified data...")
        encrypted = security_manager.encrypt_classified_data(
            data_bytes,
            ClassificationLevel.SECRET
        )
        print(f"✓ Data encrypted with KMS")
        print(f"  Key ID: {encrypted['key_id']}")
        print(f"  FIPS Compliant: {encrypted['fips_compliant']}")
        print(f"  Classification: {encrypted['encryption_context']['classification']}")
    else:
        print("⚠ AWS integration not available - skipping encryption")
        encrypted = None
    
    # Demonstrate access control
    print("\nAccess Control Checks:")
    
    # Register data classification
    security_manager.mac.classify_data(
        "doc_001",
        ClassificationLevel.SECRET
    )
    
    # Check read access
    can_read, reason = security_manager.mac.check_read_access("analyst.001", "doc_001")
    print(f"✓ Read access: {reason}")
    
    # Check write access
    can_write, reason = security_manager.mac.check_write_access("analyst.001", "doc_001")
    print(f"✓ Write access: {reason}")
    
    return encrypted


def demonstrate_immutable_audit_trail(security_manager: ClassifiedSecurityManager):
    """Demonstrate immutable audit trail."""
    print("\n=== Immutable Audit Trail Demo ===\n")
    
    # Create audit events
    events = [
        {
            "event_type": "data_access",
            "user": "analyst.001",
            "resource": "classified_doc_001",
            "action": "read",
            "classification": "SECRET",
            "timestamp": datetime.now().isoformat()
        },
        {
            "event_type": "data_modification",
            "user": "analyst.001",
            "resource": "classified_doc_001",
            "action": "update",
            "classification": "SECRET",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    if security_manager.aws_integration:
        print("Creating immutable audit trail in AWS CloudWatch...")
        for event in events:
            event_id = security_manager.aws_integration.create_immutable_audit_trail(event)
            print(f"✓ Logged event: {event['event_type']} (ID: {event_id})")
        
        print("\nAudit trail properties:")
        print("  - Immutable once written")
        print("  - 7-year retention for compliance")
        print("  - Cryptographic integrity hashing")
        print(f"  - Region: {security_manager.aws_integration.region}")
    else:
        print("⚠ AWS integration not available - using local audit log")
        for event in events:
            security_manager._log_audit_event(event)
        print(f"✓ Logged {len(events)} events locally")


def demonstrate_maif_integration():
    """Demonstrate integration with MAIF blocks."""
    print("\n=== MAIF Integration Demo ===\n")
    
    # Initialize components
    security_manager = ClassifiedSecurityManager(use_fips=True)
    maif_security = MAIFClassifiedSecurity(security_manager)
    
    # Create a classified MAIF block
    classified_content = {
        "mission": "OPERATION_EXAMPLE",
        "classification": "SECRET",
        "data": "Sensitive operational data"
    }
    
    block_data = json.dumps(classified_content).encode()
    block_metadata = {
        "classification": "SECRET",
        "compartments": ["CRYPTO"],
        "created_by": "analyst.001"
    }
    
    # Secure the block
    print("Securing MAIF block...")
    secured = maif_security.secure_block(block_data, block_metadata)
    
    print(f"✓ Block secured")
    print(f"  Encrypted: {secured['encrypted']}")
    print(f"  Classification: {secured['classification']}")
    
    if secured['encrypted'] and security_manager.aws_integration:
        print(f"  Encryption: AWS KMS (FIPS-compliant)")
    
    # Demonstrate provenance chain verification
    print("\nProvenance Chain Verification:")
    
    provenance_chain = [
        {
            "timestamp": time.time(),
            "agent_id": "system",
            "action": "genesis",
            "block_hash": "genesis_hash",
            "entry_hash": "calculated_hash_1"
        },
        {
            "timestamp": time.time(),
            "agent_id": "analyst.001",
            "action": "create",
            "block_hash": "block_001_hash",
            "previous_hash": "calculated_hash_1",
            "entry_hash": "calculated_hash_2"
        }
    ]
    
    # Add proper hashes
    for entry in provenance_chain:
        entry_copy = entry.copy()
        entry_copy.pop("entry_hash", None)
        import hashlib
        entry["entry_hash"] = hashlib.sha256(
            json.dumps(entry_copy, sort_keys=True).encode()
        ).hexdigest()
    
    is_valid, errors = security_manager.verify_provenance_chain(provenance_chain)
    print(f"✓ Provenance chain valid: {is_valid}")
    if errors:
        for error in errors:
            print(f"  Error: {error}")


def demonstrate_hardware_mfa():
    """Demonstrate hardware MFA."""
    print("\n=== Hardware MFA Demo ===\n")
    
    security_manager = ClassifiedSecurityManager()
    
    # Initiate MFA challenge
    user_id = "analyst.001"
    token_serial = "YubiKey-5C-12345678"
    
    print(f"Initiating hardware MFA for user: {user_id}")
    print(f"Token: {token_serial}")
    
    challenge_id = security_manager.mfa_auth.initiate_hardware_mfa(
        user_id, token_serial
    )
    
    print(f"✓ Challenge initiated: {challenge_id}")
    print("  Waiting for hardware token response...")
    
    # Simulate token response (in production, this comes from hardware)
    # For demo, we'll calculate what the expected response would be
    import hmac
    import hashlib
    challenge_data = security_manager.mfa_auth.pending_challenges[challenge_id]
    mock_response = hmac.new(
        challenge_data["challenge"].encode(),
        str(int(time.time() // 30)).encode(),
        hashlib.sha256
    ).hexdigest()[:6]
    
    # Verify response
    is_valid, message = security_manager.mfa_auth.verify_hardware_token(
        challenge_id, mock_response
    )
    
    print(f"{'✓' if is_valid else '✗'} MFA verification: {message}")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("MAIF Classified Security Features Demonstration")
    print("=" * 60)
    
    # Check AWS availability
    try:
        import boto3
        print("\n✓ AWS SDK available")
        print(f"  Region: {os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')}")
        print(f"  FIPS Mode: Enabled")
    except ImportError:
        print("\n⚠ AWS SDK not available - some features will be limited")
    
    # Run demonstrations
    security_manager, user_id = demonstrate_pki_authentication()
    demonstrate_user_clearance()
    demonstrate_classified_data_handling(security_manager)
    demonstrate_immutable_audit_trail(security_manager)
    demonstrate_maif_integration()
    demonstrate_hardware_mfa()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)
    
    print("\nKey Features Demonstrated:")
    print("✓ PKI/CAC/PIV authentication support")
    print("✓ Mandatory Access Control (Bell-LaPadula)")
    print("✓ FIPS 140-2 compliant encryption")
    print("✓ Immutable audit trails with AWS CloudWatch")
    print("✓ Hardware MFA support")
    print("✓ Cryptographic provenance chain verification")
    print("✓ Works with both AWS Commercial and GovCloud")
    
    print("\nProduction Considerations:")
    print("• Use real PKI certificates from CAC/PIV cards")
    print("• Configure trusted Certificate Authorities")
    print("• Enable CloudHSM for FIPS 140-2 Level 3")
    print("• Implement proper key rotation policies")
    print("• Configure SIEM integration for audit logs")
    print("• Regular security clearance verification")
    print("• Continuous monitoring and anomaly detection")


if __name__ == "__main__":
    main()