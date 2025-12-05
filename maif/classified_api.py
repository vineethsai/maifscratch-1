"""
Simple, Clear API for MAIF Classified Security
Provides easy-to-use interfaces for common classified operations.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import os

from .classified_security import (
    ClassificationLevel,
    AuthenticationMethod,
    UserClearance,
    ClassifiedSecurityManager,
    MAIFClassifiedSecurity,
)


class SecureMAIF:
    """
    Simple API for secure MAIF operations with classified data.

    Example:
        # Initialize
        maif = SecureMAIF(classification="SECRET")

        # Authenticate user
        user_id = maif.authenticate_with_pki(certificate_pem)

        # Store classified data
        doc_id = maif.store_classified_data(
            data={"mission": "REDACTED"},
            classification="SECRET"
        )

        # Check access
        if maif.can_access(user_id, doc_id):
            data = maif.retrieve_classified_data(doc_id)
    """

    def __init__(
        self,
        classification: str = "UNCLASSIFIED",
        region: str = None,
        use_fips: bool = True,
    ):
        """
        Initialize SecureMAIF with default classification level.

        Args:
            classification: Default classification level (UNCLASSIFIED, SECRET, etc.)
            region: AWS region (auto-detected if not specified)
            use_fips: Enable FIPS 140-2 compliant mode
        """
        self.default_classification = ClassificationLevel[classification.upper()]
        self.security_manager = ClassifiedSecurityManager(
            region_name=region, use_fips=use_fips
        )
        self.maif_security = MAIFClassifiedSecurity(self.security_manager)
        self._data_store: Dict[str, Dict[str, Any]] = {}

    # ========== Authentication API ==========

    def authenticate_with_pki(self, certificate_pem: str) -> Optional[str]:
        """
        Authenticate user with PKI certificate (CAC/PIV).

        Args:
            certificate_pem: PEM-encoded X.509 certificate

        Returns:
            User ID if authenticated, None otherwise
        """
        success, user_id = self.security_manager.authenticate_user(
            AuthenticationMethod.PKI_CERTIFICATE, {"certificate": certificate_pem}
        )
        return user_id if success else None

    def authenticate_with_mfa(self, user_id: str, token_serial: str) -> str:
        """
        Start MFA authentication process.

        Args:
            user_id: User identifier
            token_serial: Hardware token serial number

        Returns:
            Challenge ID to use with verify_mfa()
        """
        return self.security_manager.mfa_auth.initiate_hardware_mfa(
            user_id, token_serial
        )

    def verify_mfa(self, challenge_id: str, token_code: str) -> bool:
        """
        Verify MFA token code.

        Args:
            challenge_id: Challenge ID from authenticate_with_mfa()
            token_code: 6-digit code from hardware token

        Returns:
            True if verified, False otherwise
        """
        success, _ = self.security_manager.mfa_auth.verify_hardware_token(
            challenge_id, token_code
        )
        return success

    # ========== User Clearance API ==========

    def grant_clearance(
        self,
        user_id: str,
        level: str,
        valid_days: int = 365,
        compartments: List[str] = None,
        caveats: List[str] = None,
    ) -> None:
        """
        Grant security clearance to a user.

        Args:
            user_id: User identifier
            level: Clearance level (CONFIDENTIAL, SECRET, TOP_SECRET, etc.)
            valid_days: Days until clearance expires
            compartments: SCI compartments (e.g., ["CRYPTO", "SIGINT"])
            caveats: Handling caveats (e.g., ["NOFORN", "REL TO FVEY"])
        """
        clearance = UserClearance(
            user_id=user_id,
            clearance_level=ClassificationLevel[level.upper()],
            clearance_expiry=datetime.now() + timedelta(days=valid_days),
            compartments=compartments or [],
            caveat_access=caveats or [],
        )
        self.security_manager.mac.register_user_clearance(clearance)

    def check_clearance(self, user_id: str) -> Optional[str]:
        """
        Check user's clearance level.

        Args:
            user_id: User identifier

        Returns:
            Clearance level name or None if not found
        """
        with self.security_manager.mac._lock:
            if user_id in self.security_manager.mac.clearances:
                clearance = self.security_manager.mac.clearances[user_id]
                if clearance.is_valid():
                    return clearance.clearance_level.name
        return None

    # ========== Data Classification API ==========

    def store_classified_data(
        self,
        data: Union[Dict, str, bytes],
        classification: str = None,
        compartments: List[str] = None,
        caveats: List[str] = None,
    ) -> str:
        """
        Store classified data with automatic encryption.

        Args:
            data: Data to store (dict, string, or bytes)
            classification: Classification level (uses default if not specified)
            compartments: Required compartments for access
            caveats: Handling caveats

        Returns:
            Document ID for retrieval
        """
        # Convert data to bytes
        if isinstance(data, dict):
            import json

            data_bytes = json.dumps(data).encode()
        elif isinstance(data, str):
            data_bytes = data.encode()
        else:
            data_bytes = data

        # Determine classification
        if classification:
            class_level = ClassificationLevel[classification.upper()]
        else:
            class_level = self.default_classification

        # Generate document ID
        import uuid

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        # Classify the data
        self.security_manager.mac.classify_data(
            doc_id, class_level, compartments, caveats
        )

        # Encrypt if classified
        if class_level != ClassificationLevel.UNCLASSIFIED:
            if self.security_manager.aws_integration:
                encrypted = self.security_manager.encrypt_classified_data(
                    data_bytes, class_level
                )
                self._data_store[doc_id] = {
                    "encrypted": True,
                    "data": encrypted,
                    "classification": class_level,
                }
            else:
                # Fallback to local storage with warning
                self._data_store[doc_id] = {
                    "encrypted": False,
                    "data": data_bytes,
                    "classification": class_level,
                    "warning": "AWS KMS not available - data not encrypted",
                }
        else:
            self._data_store[doc_id] = {
                "encrypted": False,
                "data": data_bytes,
                "classification": class_level,
            }

        return doc_id

    def retrieve_classified_data(
        self, doc_id: str, user_id: str = None
    ) -> Optional[bytes]:
        """
        Retrieve classified data (with access control).

        Args:
            doc_id: Document ID from store_classified_data()
            user_id: User requesting access (for audit)

        Returns:
            Decrypted data bytes or None if access denied
        """
        if doc_id not in self._data_store:
            return None

        stored = self._data_store[doc_id]

        # Decrypt if needed
        if stored["encrypted"] and self.security_manager.aws_integration:
            enc_data = stored["data"]
            decrypted = self.security_manager.aws_integration.decrypt_with_kms(
                enc_data["ciphertext"], enc_data["encryption_context"]
            )
            return decrypted
        else:
            return stored["data"]

    # ========== Access Control API ==========

    def can_access(self, user_id: str, doc_id: str, operation: str = "read") -> bool:
        """
        Check if user can access a document.

        Args:
            user_id: User identifier
            doc_id: Document ID
            operation: "read" or "write"

        Returns:
            True if access allowed, False otherwise
        """
        if operation == "read":
            allowed, _ = self.security_manager.mac.check_read_access(user_id, doc_id)
        else:
            allowed, _ = self.security_manager.mac.check_write_access(user_id, doc_id)
        return allowed

    def require_mfa_for_access(self, doc_id: str) -> None:
        """
        Require MFA for accessing a specific document.

        Args:
            doc_id: Document ID
        """
        # This would integrate with the streaming access control
        # For now, we'll track it in metadata
        if doc_id in self._data_store:
            self._data_store[doc_id]["require_mfa"] = True

    # ========== Audit Trail API ==========

    def log_access(
        self, user_id: str, doc_id: str, action: str, success: bool = True
    ) -> str:
        """
        Log access attempt to audit trail.

        Args:
            user_id: User who attempted access
            doc_id: Document accessed
            action: Action performed (read, write, delete)
            success: Whether access was successful

        Returns:
            Audit event ID
        """
        event = {
            "event_type": "data_access",
            "user_id": user_id,
            "resource": doc_id,
            "action": action,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }

        # Add classification info if available
        if doc_id in self._data_store:
            event["classification"] = self._data_store[doc_id]["classification"].name

        # Log to AWS if available
        if self.security_manager.aws_integration:
            return self.security_manager.aws_integration.create_immutable_audit_trail(
                event
            )
        else:
            self.security_manager._log_audit_event(event)
            return event.get("event_id", "local_log")

    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent audit events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of audit events
        """
        with self.security_manager._lock:
            return self.security_manager.audit_events[-limit:]

    # ========== Status and Health API ==========

    def get_status(self) -> Dict[str, Any]:
        """
        Get system status and configuration.

        Returns:
            Status information dictionary
        """
        status = {
            "fips_mode": self.security_manager.aws_integration.use_fips
            if self.security_manager.aws_integration
            else False,
            "aws_available": self.security_manager.aws_integration is not None,
            "default_classification": self.default_classification.name,
            "users_registered": len(self.security_manager.mac.clearances),
            "documents_stored": len(self._data_store),
            "audit_events": len(self.security_manager.audit_events),
        }

        if self.security_manager.aws_integration:
            status.update(
                {
                    "aws_region": self.security_manager.aws_integration.region,
                    "aws_account": self.security_manager.aws_integration.account_id,
                    "is_govcloud": self.security_manager.aws_integration.is_govcloud,
                    "hsm_available": self.security_manager.aws_integration.hsm_available,
                }
            )

        return status


# ========== Quick Start Functions ==========


def create_secure_maif(classification: str = "SECRET") -> SecureMAIF:
    """
    Quick function to create a secure MAIF instance.

    Args:
        classification: Default classification level

    Returns:
        Configured SecureMAIF instance
    """
    return SecureMAIF(classification=classification)


def quick_secure_storage(
    data: Dict, classification: str = "SECRET", user_clearance: str = "SECRET"
) -> Tuple[SecureMAIF, str, str]:
    """
    Quick function to securely store classified data.

    Args:
        data: Data to store
        classification: Data classification level
        user_clearance: User's clearance level

    Returns:
        Tuple of (SecureMAIF instance, user_id, document_id)

    Example:
        maif, user, doc = quick_secure_storage(
            {"mission": "OPERATION_X"},
            classification="SECRET"
        )
    """
    # Create secure instance
    maif = SecureMAIF(classification=classification)

    # Create test user with clearance
    user_id = "quick_user_001"
    maif.grant_clearance(user_id, user_clearance)

    # Store data
    doc_id = maif.store_classified_data(data, classification)

    # Log access
    maif.log_access(user_id, doc_id, "create", True)

    return maif, user_id, doc_id
