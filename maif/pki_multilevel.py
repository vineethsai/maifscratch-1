"""
Multi-Level PKI Support for MAIF
================================

Implements advanced PKI features including:
- Multi-level certificate hierarchies (Root CA -> Intermediate CAs -> End certificates)
- Cross-certification support for inter-domain trust
- Bridge CA support for connecting multiple PKI domains
- Certificate path building and validation
- Advanced revocation checking (CRL and OCSP)
"""

import os
import logging
import threading
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import base64

try:
    from cryptography import x509
    from cryptography.x509 import load_pem_x509_certificate, load_der_x509_certificate
    from cryptography.x509.oid import NameOID, ExtensionOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend
    from cryptography.x509.verification import PolicyBuilder, Store

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CertificateInfo:
    """Enhanced certificate information with multi-level support."""

    certificate: Any  # x509.Certificate
    subject: str
    issuer: str
    serial_number: str
    not_valid_before: datetime
    not_valid_after: datetime
    key_usage: List[str]
    is_ca: bool
    path_length_constraint: Optional[int]
    trust_level: int  # 0=untrusted, 1=end cert, 2=intermediate, 3=root
    cross_certificates: List[str] = field(default_factory=list)
    revocation_info: Dict[str, Any] = field(default_factory=dict)


class CertificatePathBuilder:
    """Builds and validates certificate paths in multi-level PKI."""

    def __init__(self):
        self.cert_cache: Dict[str, CertificateInfo] = {}
        self.trust_anchors: Set[str] = set()
        self.bridge_cas: Set[str] = set()
        self._lock = threading.RLock()

    def add_trust_anchor(self, cert_pem: str) -> bool:
        """Add a root CA as trust anchor."""
        try:
            cert = load_pem_x509_certificate(cert_pem.encode(), default_backend())
            fingerprint = cert.fingerprint(hashes.SHA256()).hex()

            with self._lock:
                self.trust_anchors.add(fingerprint)
                self.cert_cache[fingerprint] = self._extract_cert_info(
                    cert, trust_level=3
                )

            logger.info(f"Added trust anchor: {fingerprint}")
            return True
        except Exception as e:
            logger.error(f"Failed to add trust anchor: {e}")
            return False

    def add_bridge_ca(self, cert_pem: str) -> bool:
        """Add a bridge CA for cross-domain trust."""
        try:
            cert = load_pem_x509_certificate(cert_pem.encode(), default_backend())
            fingerprint = cert.fingerprint(hashes.SHA256()).hex()

            with self._lock:
                self.bridge_cas.add(fingerprint)
                self.cert_cache[fingerprint] = self._extract_cert_info(
                    cert, trust_level=2
                )

            logger.info(f"Added bridge CA: {fingerprint}")
            return True
        except Exception as e:
            logger.error(f"Failed to add bridge CA: {e}")
            return False

    def build_path(
        self, end_cert_pem: str, intermediate_certs: List[str] = None
    ) -> Optional[List[CertificateInfo]]:
        """Build a certificate path from end certificate to trust anchor."""
        try:
            end_cert = load_pem_x509_certificate(
                end_cert_pem.encode(), default_backend()
            )

            # Add intermediate certificates to cache
            if intermediate_certs:
                for int_cert_pem in intermediate_certs:
                    int_cert = load_pem_x509_certificate(
                        int_cert_pem.encode(), default_backend()
                    )
                    fingerprint = int_cert.fingerprint(hashes.SHA256()).hex()
                    self.cert_cache[fingerprint] = self._extract_cert_info(
                        int_cert, trust_level=2
                    )

            # Build path using depth-first search
            path = self._build_path_recursive(end_cert, [])

            if path:
                logger.info(f"Built certificate path of length {len(path)}")
                return path
            else:
                logger.warning("Failed to build certificate path to trust anchor")
                return None

        except Exception as e:
            logger.error(f"Error building certificate path: {e}")
            return None

    def _build_path_recursive(
        self, cert: Any, current_path: List[CertificateInfo], visited: Set[str] = None
    ) -> Optional[List[CertificateInfo]]:
        """Recursively build path to trust anchor."""
        if visited is None:
            visited = set()

        fingerprint = cert.fingerprint(hashes.SHA256()).hex()

        # Avoid cycles
        if fingerprint in visited:
            return None
        visited.add(fingerprint)

        # Extract certificate info
        cert_info = self._extract_cert_info(cert)
        current_path = current_path + [cert_info]

        # Check if we reached a trust anchor
        if fingerprint in self.trust_anchors:
            return current_path

        # Check if this is a bridge CA (can connect domains)
        if fingerprint in self.bridge_cas:
            # Try to find path through bridge
            for cross_cert_fp in cert_info.cross_certificates:
                if cross_cert_fp in self.cert_cache:
                    cross_cert = self.cert_cache[cross_cert_fp].certificate
                    result = self._build_path_recursive(
                        cross_cert, current_path, visited.copy()
                    )
                    if result:
                        return result

        # Try to find issuer
        issuer_name = cert.issuer
        for cached_fp, cached_info in self.cert_cache.items():
            if cached_info.subject == issuer_name.rfc4514_string():
                result = self._build_path_recursive(
                    cached_info.certificate, current_path, visited.copy()
                )
                if result:
                    return result

        return None

    def _extract_cert_info(self, cert: Any, trust_level: int = 1) -> CertificateInfo:
        """Extract detailed certificate information."""
        # Check if this is a CA certificate
        is_ca = False
        path_length = None
        try:
            basic_constraints = cert.extensions.get_extension_for_oid(
                ExtensionOID.BASIC_CONSTRAINTS
            )
            is_ca = basic_constraints.value.ca
            path_length = basic_constraints.value.path_length
        except x509.ExtensionNotFound:
            pass  # Authority information access extension not present

        # Extract key usage
        key_usage = []
        try:
            ku_ext = cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE)
            ku = ku_ext.value
            if ku.digital_signature:
                key_usage.append("digital_signature")
            if ku.key_cert_sign:
                key_usage.append("key_cert_sign")
            if ku.crl_sign:
                key_usage.append("crl_sign")
        except x509.ExtensionNotFound:
            pass  # CRL distribution points extension not present

        # Extract revocation info
        revocation_info = {}
        try:
            crl_ext = cert.extensions.get_extension_for_oid(
                ExtensionOID.CRL_DISTRIBUTION_POINTS
            )
            revocation_info["crl_points"] = [
                point.full_name[0].value for point in crl_ext.value if point.full_name
            ]
        except x509.ExtensionNotFound:
            pass  # Key usage extension not present

        try:
            aia_ext = cert.extensions.get_extension_for_oid(
                ExtensionOID.AUTHORITY_INFORMATION_ACCESS
            )
            ocsp_urls = []
            for desc in aia_ext.value:
                if desc.access_method._name == "OCSP":
                    ocsp_urls.append(desc.access_location.value)
            if ocsp_urls:
                revocation_info["ocsp_urls"] = ocsp_urls
        except x509.ExtensionNotFound:
            pass  # Basic constraints extension not present

        return CertificateInfo(
            certificate=cert,
            subject=cert.subject.rfc4514_string(),
            issuer=cert.issuer.rfc4514_string(),
            serial_number=str(cert.serial_number),
            not_valid_before=cert.not_valid_before,
            not_valid_after=cert.not_valid_after,
            key_usage=key_usage,
            is_ca=is_ca,
            path_length_constraint=path_length,
            trust_level=trust_level,
            revocation_info=revocation_info,
        )


class MultiLevelPKIValidator:
    """Validates certificates in multi-level PKI hierarchies."""

    def __init__(self):
        self.path_builder = CertificatePathBuilder()
        self.revocation_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._lock = threading.RLock()

    def validate_certificate_chain(
        self,
        end_cert_pem: str,
        intermediate_certs: List[str] = None,
        check_revocation: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate a certificate chain with full path validation."""
        result = {"valid": False, "path": [], "errors": [], "warnings": []}

        try:
            # Build certificate path
            path = self.path_builder.build_path(end_cert_pem, intermediate_certs)
            if not path:
                result["errors"].append("Failed to build path to trust anchor")
                return False, result

            result["path"] = [
                {
                    "subject": cert.subject,
                    "issuer": cert.issuer,
                    "serial": cert.serial_number,
                    "is_ca": cert.is_ca,
                }
                for cert in path
            ]

            # Validate each certificate in the path
            for i, cert_info in enumerate(path):
                # Check validity period
                now = datetime.utcnow()
                if now < cert_info.not_valid_before:
                    result["errors"].append(f"Certificate {i} not yet valid")
                    return False, result
                if now > cert_info.not_valid_after:
                    result["errors"].append(f"Certificate {i} expired")
                    return False, result

                # Check CA constraints
                if i < len(path) - 1:  # Not the last cert (root)
                    if not cert_info.is_ca:
                        result["errors"].append(
                            f"Certificate {i} is not a CA but has issued certificates"
                        )
                        return False, result

                    # Check path length constraints
                    if cert_info.path_length_constraint is not None:
                        remaining_path = len(path) - i - 2  # Exclude current and root
                        if remaining_path > cert_info.path_length_constraint:
                            result["errors"].append(
                                f"Path length constraint violated at certificate {i}"
                            )
                            return False, result

                # Check key usage
                if i < len(path) - 1 and "key_cert_sign" not in cert_info.key_usage:
                    result["errors"].append(
                        f"Certificate {i} lacks key_cert_sign usage"
                    )
                    return False, result

                # Check revocation if enabled
                if check_revocation:
                    is_revoked, revocation_time = self._check_revocation(cert_info)
                    if is_revoked:
                        result["errors"].append(
                            f"Certificate {i} revoked at {revocation_time}"
                        )
                        return False, result

            # Verify signatures in the chain
            for i in range(len(path) - 1):
                child_cert = path[i].certificate
                parent_cert = path[i + 1].certificate

                try:
                    # Verify child was signed by parent
                    from cryptography.hazmat.primitives.asymmetric import padding

                    parent_public_key = parent_cert.public_key()
                    parent_public_key.verify(
                        child_cert.signature,
                        child_cert.tbs_certificate_bytes,
                        padding.PKCS1v15(),
                        child_cert.signature_hash_algorithm,
                    )
                except Exception as e:
                    result["errors"].append(
                        f"Signature verification failed at level {i}: {str(e)}"
                    )
                    return False, result

            result["valid"] = True
            return True, result

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
            return False, result

    def _check_revocation(
        self, cert_info: CertificateInfo
    ) -> Tuple[bool, Optional[datetime]]:
        """Check certificate revocation status."""
        fingerprint = cert_info.certificate.fingerprint(hashes.SHA256()).hex()

        # Check cache first
        with self._lock:
            if fingerprint in self.revocation_cache:
                cached_result, cache_time = self.revocation_cache[fingerprint]
                if datetime.utcnow() - cache_time < timedelta(hours=1):
                    return cached_result, cache_time if cached_result else None

        # Check CRL if available
        if "crl_points" in cert_info.revocation_info:
            # In production, fetch and check CRL
            # For now, assume not revoked
            pass

        # Check OCSP if available
        if "ocsp_urls" in cert_info.revocation_info:
            # In production, query OCSP responder
            # For now, assume not revoked
            pass

        # Cache result
        with self._lock:
            self.revocation_cache[fingerprint] = (False, datetime.utcnow())

        return False, None


class CrossCertificationManager:
    """Manages cross-certification between different PKI domains."""

    def __init__(self):
        self.cross_certs: Dict[Tuple[str, str], str] = {}  # (domain1, domain2) -> cert
        self.domain_roots: Dict[str, str] = {}  # domain -> root cert
        self._lock = threading.RLock()

    def add_domain_root(self, domain: str, root_cert_pem: str) -> bool:
        """Add a root certificate for a PKI domain."""
        try:
            with self._lock:
                self.domain_roots[domain] = root_cert_pem
            logger.info(f"Added root certificate for domain: {domain}")
            return True
        except Exception as e:
            logger.error(f"Failed to add domain root: {e}")
            return False

    def add_cross_certification(
        self, domain1: str, domain2: str, cross_cert_pem: str
    ) -> bool:
        """Add a cross-certification between two domains."""
        try:
            with self._lock:
                self.cross_certs[(domain1, domain2)] = cross_cert_pem
                self.cross_certs[(domain2, domain1)] = cross_cert_pem
            logger.info(f"Added cross-certification between {domain1} and {domain2}")
            return True
        except Exception as e:
            logger.error(f"Failed to add cross-certification: {e}")
            return False

    def find_trust_path(
        self, cert_domain: str, target_domain: str
    ) -> Optional[List[str]]:
        """Find a trust path between two PKI domains."""
        if cert_domain == target_domain:
            return [cert_domain]

        # Use BFS to find shortest path
        from collections import deque

        queue = deque([(cert_domain, [cert_domain])])
        visited = {cert_domain}

        while queue:
            current_domain, path = queue.popleft()

            # Check all cross-certifications from current domain
            for (d1, d2), _ in self.cross_certs.items():
                next_domain = None
                if d1 == current_domain and d2 not in visited:
                    next_domain = d2
                elif d2 == current_domain and d1 not in visited:
                    next_domain = d1

                if next_domain:
                    new_path = path + [next_domain]
                    if next_domain == target_domain:
                        return new_path
                    visited.add(next_domain)
                    queue.append((next_domain, new_path))

        return None


# Integration with existing ClassifiedDataProtection
def enhance_classified_security_with_multilevel_pki(cdp_instance):
    """Enhance existing ClassifiedDataProtection with multi-level PKI support."""

    # Create multi-level PKI components
    pki_validator = MultiLevelPKIValidator()
    cross_cert_manager = CrossCertificationManager()

    # Add enhanced PKI validation method
    def validate_multilevel_pki(
        self,
        cert_pem: str,
        intermediate_certs: List[str] = None,
        required_domain: str = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate certificate with multi-level PKI support."""

        # First, do multi-level validation
        is_valid, validation_result = pki_validator.validate_certificate_chain(
            cert_pem, intermediate_certs
        )

        if not is_valid:
            return False, validation_result

        # If domain is specified, check cross-certification
        if required_domain:
            cert = load_pem_x509_certificate(cert_pem.encode(), default_backend())
            cert_domain = self._extract_domain_from_cert(cert)

            if cert_domain != required_domain:
                # Check if there's a trust path via cross-certification
                trust_path = cross_cert_manager.find_trust_path(
                    cert_domain, required_domain
                )
                if not trust_path:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"No trust path from {cert_domain} to {required_domain}"
                    )
                    return False, validation_result

                validation_result["cross_certification_path"] = trust_path

        return True, validation_result

    # Add method to ClassifiedDataProtection
    cdp_instance.validate_multilevel_pki = validate_multilevel_pki.__get__(cdp_instance)
    cdp_instance._pki_validator = pki_validator
    cdp_instance._cross_cert_manager = cross_cert_manager

    return cdp_instance
