#!/usr/bin/env python3
"""
Simple demonstration of the MAIF Classified Security API
Shows how easy it is to work with classified data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from maif.security.classified_api import SecureMAIF, quick_secure_storage


def simple_example():
    """Simplest possible example."""
    print("=== Simple Classified Data Storage ===\n")

    # 1. Create secure MAIF instance
    maif = SecureMAIF(classification="SECRET")

    # 2. Grant user clearance
    maif.grant_clearance(user_id="analyst.001", level="SECRET", compartments=["CRYPTO"])

    # 3. Store classified data
    doc_id = maif.store_classified_data(
        data={"operation": "EAGLE_EYE", "location": "CLASSIFIED", "status": "ACTIVE"},
        classification="SECRET",
    )

    print(f"Stored classified document: {doc_id}")

    # 4. Check access
    if maif.can_access("analyst.001", doc_id):
        print("User has access to document")

        # 5. Retrieve data
        data = maif.retrieve_classified_data(doc_id, "analyst.001")
        print("Retrieved classified data")

        # 6. Log the access
        maif.log_access("analyst.001", doc_id, "read")
        print("Access logged to audit trail")

    # 7. Check status
    status = maif.get_status()
    print(f"\nSystem Status:")
    print(f"FIPS Mode: {status['fips_mode']}")
    print(f"Documents: {status['documents_stored']}")
    print(f"Audit Events: {status['audit_events']}")


def quick_start_example():
    """Even simpler with quick start functions."""
    print("\n=== Quick Start Example ===\n")

    # One line to securely store classified data!
    maif, user, doc = quick_secure_storage(
        data={"mission": "OPERATION_PHOENIX"}, classification="SECRET"
    )

    print(f"Created secure storage")
    print(f"User: {user}")
    print(f"Document: {doc}")
    print(f"Can access: {maif.can_access(user, doc)}")


def mfa_example():
    """Example with MFA."""
    print("\n=== MFA Example ===\n")

    maif = SecureMAIF(classification="TOP_SECRET")

    # 1. Start MFA authentication
    challenge_id = maif.authenticate_with_mfa(
        user_id="admin.001", token_serial="YubiKey-12345"
    )

    print(f"MFA Challenge: {challenge_id}")
    print("(In production, user enters code from hardware token)")

    # 2. Verify MFA (simulated)
    # In real usage, this code comes from the hardware token
    if maif.verify_mfa(challenge_id, "123456"):
        print("MFA Verified")


def access_control_example():
    """Example showing access control."""
    print("\n=== Access Control Example ===\n")

    maif = SecureMAIF()

    # Set up users with different clearances
    users = [
        ("intern.001", "UNCLASSIFIED", []),
        ("analyst.001", "SECRET", ["SIGINT"]),
        ("director.001", "TOP_SECRET", ["SIGINT", "CRYPTO"]),
    ]

    for user_id, level, compartments in users:
        maif.grant_clearance(user_id, level, compartments=compartments)
        print(f"{user_id}: {level} clearance")

    # Create documents at different levels
    docs = [
        ("Public report", "UNCLASSIFIED", []),
        ("Intel brief", "SECRET", ["SIGINT"]),
        ("Op plan", "TOP_SECRET", ["CRYPTO"]),
    ]

    doc_ids = []
    for title, classification, compartments in docs:
        doc_id = maif.store_classified_data(
            data={"title": title},
            classification=classification,
            compartments=compartments,
        )
        doc_ids.append((doc_id, classification))
        print(f"Created {classification} document: {title}")

    # Check access
    print("\nAccess Matrix:")
    print("User          | UNCLASS | SECRET  | TOP_SECRET")
    print("-" * 50)

    for user_id, _, _ in users:
        access = []
        for doc_id, _ in doc_ids:
            can_read = maif.can_access(user_id, doc_id)
            access.append("" if can_read else "")

        print(f"{user_id:<13} | {access[0]:<7} | {access[1]:<7} | {access[2]}")


def audit_trail_example():
    """Example showing audit trail."""
    print("\n=== Audit Trail Example ===\n")

    maif = SecureMAIF()

    # Create some activity
    maif.grant_clearance("user.001", "SECRET")
    doc_id = maif.store_classified_data({"data": "test"}, "SECRET")

    # Various access attempts
    events = [
        ("user.001", doc_id, "read", True),
        ("user.002", doc_id, "read", False),  # No clearance
        ("user.001", doc_id, "write", True),
        ("user.001", doc_id, "delete", False),  # Not implemented
    ]

    for user, doc, action, success in events:
        maif.log_access(user, doc, action, success)

    # Get audit trail
    audit_events = maif.get_audit_trail(limit=5)

    print("Recent Audit Events:")
    for event in audit_events:
        status = "" if event.get("success", False) else ""
        print(f"{status} {event['user_id']}: {event['action']} on {event['resource']}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("MAIF Classified Security - Simple API Demo")
    print("=" * 60)

    simple_example()
    quick_start_example()
    mfa_example()
    access_control_example()
    audit_trail_example()

    print("\n" + "=" * 60)
    print("API Summary:")
    print("=" * 60)

    print("""
Core API Methods:
    
1. Initialize:
   maif = SecureMAIF(classification="SECRET")
   
2. User Management:
   maif.grant_clearance(user_id, level, compartments=[])
   maif.check_clearance(user_id)
   
3. Data Storage:
   doc_id = maif.store_classified_data(data, classification)
   data = maif.retrieve_classified_data(doc_id)
   
4. Access Control:
   maif.can_access(user_id, doc_id, operation="read")
   maif.require_mfa_for_access(doc_id)
   
5. Authentication:
   user_id = maif.authenticate_with_pki(certificate)
   challenge = maif.authenticate_with_mfa(user_id, token_serial)
   success = maif.verify_mfa(challenge, code)
   
6. Audit:
   maif.log_access(user_id, doc_id, action)
   events = maif.get_audit_trail()
   
7. Status:
   status = maif.get_status()
   
Quick Start:
   # One-liner for secure storage
   maif, user, doc = quick_secure_storage(data, classification)
""")


if __name__ == "__main__":
    main()
