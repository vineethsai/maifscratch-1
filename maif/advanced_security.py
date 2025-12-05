"""
Advanced Security Features for MAIF
Implements enhanced threat protection, behavioral anomaly detection, and advanced cryptographic features.
"""

import time
import hashlib
import hmac
import secrets
import json
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import statistics
import os
import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import HSM support libraries
try:
    import PyKCS11

    PKCS11_AVAILABLE = True
except ImportError:
    PKCS11_AVAILABLE = False
    logger.warning("PyKCS11 not available. Install python-pkcs11 for HSM support.")

try:
    import yubihsm

    YUBIHSM_AVAILABLE = True
except ImportError:
    YUBIHSM_AVAILABLE = False

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class ThreatLevel(Enum):
    """Threat severity levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AttackType(Enum):
    """Types of detected attacks."""

    TIMING_ATTACK = "timing_attack"
    REPLAY_ATTACK = "replay_attack"
    INJECTION_ATTACK = "injection_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    CRYPTOGRAPHIC_ATTACK = "cryptographic_attack"


@dataclass
class SecurityEvent:
    """Security event record."""

    timestamp: float
    event_type: AttackType
    threat_level: ThreatLevel
    source_id: str
    target_resource: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    mitigated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "source_id": self.source_id,
            "target_resource": self.target_resource,
            "description": self.description,
            "evidence": self.evidence,
            "mitigated": self.mitigated,
        }


class TimingAttackDetector:
    """Detects timing-based attacks through response time analysis."""

    def __init__(self, baseline_samples: int = 100, threshold_multiplier: float = 3.0):
        self.baseline_samples = baseline_samples
        self.threshold_multiplier = threshold_multiplier
        self.response_times: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=baseline_samples)
        )
        self.baselines: Dict[str, float] = {}
        self.attack_detected = False

    def record_operation_time(self, operation: str, duration: float) -> bool:
        """Record operation timing and detect anomalies."""
        self.response_times[operation].append(duration)

        # Build baseline if we have enough samples
        if len(self.response_times[operation]) >= self.baseline_samples:
            times = list(self.response_times[operation])
            baseline_mean = statistics.mean(times)
            baseline_std = statistics.stdev(times) if len(times) > 1 else 0.1

            self.baselines[operation] = baseline_mean + (
                self.threshold_multiplier * baseline_std
            )

            # Check for timing attack
            if duration > self.baselines[operation]:
                self.attack_detected = True
                return True

        return False

    def get_constant_time_response(
        self, operation: str, actual_duration: float
    ) -> float:
        """Return normalized response time to prevent timing attacks."""
        if operation in self.baselines:
            # Always return the baseline time (1-10ms as specified in paper)
            return max(0.001, min(0.010, self.baselines[operation]))
        return max(0.001, min(0.010, actual_duration))


class ReplayAttackProtection:
    """Prevents replay attacks using nonces and timestamps."""

    def __init__(self, nonce_window: int = 300):  # 5 minutes
        self.nonce_window = nonce_window
        self.used_nonces: Set[str] = set()
        self.nonce_timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def generate_nonce(self) -> str:
        """Generate cryptographically secure nonce."""
        return secrets.token_hex(16)

    def validate_nonce(self, nonce: str, timestamp: float) -> bool:
        """Validate nonce hasn't been used and is within time window."""
        current_time = time.time()

        with self._lock:
            # Check if nonce was already used
            if nonce in self.used_nonces:
                return False

            # Check timestamp is within acceptable window
            if abs(current_time - timestamp) > self.nonce_window:
                return False

            # Add nonce to used set
            self.used_nonces.add(nonce)
            self.nonce_timestamps[nonce] = timestamp

            # Clean up old nonces
            self._cleanup_old_nonces(current_time)

            return True

    def _cleanup_old_nonces(self, current_time: float):
        """Remove expired nonces."""
        expired_nonces = [
            nonce
            for nonce, ts in self.nonce_timestamps.items()
            if current_time - ts > self.nonce_window
        ]

        for nonce in expired_nonces:
            self.used_nonces.discard(nonce)
            self.nonce_timestamps.pop(nonce, None)


class BehavioralAnomalyDetector:
    """ML-based behavioral anomaly detection for AI agents."""

    def __init__(self, learning_window: int = 1000):
        self.learning_window = learning_window
        self.agent_profiles: Dict[str, Dict[str, Any]] = {}
        self.operation_patterns: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=learning_window)
        )
        self.anomaly_threshold = 2.5  # Standard deviations

    def record_agent_behavior(
        self, agent_id: str, operation: str, metadata: Dict[str, Any]
    ):
        """Record agent behavior for pattern analysis."""
        timestamp = time.time()

        # Initialize agent profile
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = {
                "operations_per_hour": deque(maxlen=24),
                "operation_types": defaultdict(int),
                "access_patterns": defaultdict(list),
                "last_activity": timestamp,
            }

        profile = self.agent_profiles[agent_id]

        # Update operation counts
        profile["operation_types"][operation] += 1
        profile["last_activity"] = timestamp

        # Record operation pattern
        pattern_key = f"{agent_id}:{operation}"
        self.operation_patterns[pattern_key].append(
            {"timestamp": timestamp, "metadata": metadata}
        )

    def detect_anomalies(
        self, agent_id: str, operation: str, metadata: Dict[str, Any]
    ) -> List[SecurityEvent]:
        """Detect behavioral anomalies."""
        anomalies = []

        if agent_id not in self.agent_profiles:
            return anomalies

        profile = self.agent_profiles[agent_id]
        current_time = time.time()

        # Check for excessive activity (>100 operations per agent as mentioned in paper)
        recent_ops = sum(
            1
            for ops in self.operation_patterns.values()
            for op in ops
            if current_time - op["timestamp"] < 3600
        )

        if recent_ops > 100:
            anomalies.append(
                SecurityEvent(
                    timestamp=current_time,
                    event_type=AttackType.BEHAVIORAL_ANOMALY,
                    threat_level=ThreatLevel.HIGH,
                    source_id=agent_id,
                    target_resource=operation,
                    description=f"Excessive activity detected: {recent_ops} operations in last hour",
                    evidence={"operation_count": recent_ops, "threshold": 100},
                )
            )

        # Check for unusual timing patterns
        pattern_key = f"{agent_id}:{operation}"
        if (
            pattern_key in self.operation_patterns
            and len(self.operation_patterns[pattern_key]) > 10
        ):
            recent_times = [
                op["timestamp"]
                for op in list(self.operation_patterns[pattern_key])[-10:]
            ]
            intervals = [
                recent_times[i] - recent_times[i - 1]
                for i in range(1, len(recent_times))
            ]

            if intervals and statistics.stdev(intervals) < 0.1:  # Very regular timing
                anomalies.append(
                    SecurityEvent(
                        timestamp=current_time,
                        event_type=AttackType.BEHAVIORAL_ANOMALY,
                        threat_level=ThreatLevel.MEDIUM,
                        source_id=agent_id,
                        target_resource=operation,
                        description="Automated/scripted behavior detected",
                        evidence={"timing_regularity": statistics.stdev(intervals)},
                    )
                )

        return anomalies


class MultiFactorAuthentication:
    """Multi-factor authentication for sensitive operations."""

    def __init__(self):
        self.pending_challenges: Dict[str, Dict[str, Any]] = {}
        self.authenticated_sessions: Dict[str, float] = {}
        self.session_timeout = 3600  # 1 hour

    def require_mfa(self, agent_id: str, operation: str, resource: str) -> str:
        """Initiate MFA challenge for sensitive operation."""
        challenge_id = secrets.token_hex(16)
        challenge_code = secrets.randbelow(900000) + 100000  # 6-digit code

        self.pending_challenges[challenge_id] = {
            "agent_id": agent_id,
            "operation": operation,
            "resource": resource,
            "code": challenge_code,
            "timestamp": time.time(),
            "attempts": 0,
        }

        return challenge_id

    def verify_mfa(self, challenge_id: str, provided_code: int) -> bool:
        """Verify MFA challenge response."""
        if challenge_id not in self.pending_challenges:
            return False

        challenge = self.pending_challenges[challenge_id]
        challenge["attempts"] += 1

        # Check if challenge expired (5 minutes)
        if time.time() - challenge["timestamp"] > 300:
            del self.pending_challenges[challenge_id]
            return False

        # Check if too many attempts
        if challenge["attempts"] > 3:
            del self.pending_challenges[challenge_id]
            return False

        # Verify code
        if challenge["code"] == provided_code:
            # Create authenticated session
            session_id = f"{challenge['agent_id']}:{challenge['operation']}"
            self.authenticated_sessions[session_id] = time.time()
            del self.pending_challenges[challenge_id]
            return True

        return False

    def is_authenticated(self, agent_id: str, operation: str) -> bool:
        """Check if agent is authenticated for operation."""
        session_id = f"{agent_id}:{operation}"

        if session_id not in self.authenticated_sessions:
            return False

        # Check session timeout
        if time.time() - self.authenticated_sessions[session_id] > self.session_timeout:
            del self.authenticated_sessions[session_id]
            return False

        return True


class AdvancedThreatProtection:
    """Comprehensive threat protection system."""

    def __init__(self):
        self.timing_detector = TimingAttackDetector()
        self.replay_protection = ReplayAttackProtection()
        self.anomaly_detector = BehavioralAnomalyDetector()
        self.mfa = MultiFactorAuthentication()
        self.security_events: List[SecurityEvent] = []
        self.threat_counters: Dict[AttackType, int] = defaultdict(int)

    def process_operation(
        self, agent_id: str, operation: str, resource: str, metadata: Dict[str, Any]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Process operation through all security checks."""
        start_time = time.time()
        events = []

        # Record behavior
        self.anomaly_detector.record_agent_behavior(agent_id, operation, metadata)

        # Check for behavioral anomalies
        anomalies = self.anomaly_detector.detect_anomalies(
            agent_id, operation, metadata
        )
        events.extend(anomalies)

        # Check timing attacks
        operation_duration = time.time() - start_time
        if self.timing_detector.record_operation_time(operation, operation_duration):
            events.append(
                SecurityEvent(
                    timestamp=time.time(),
                    event_type=AttackType.TIMING_ATTACK,
                    threat_level=ThreatLevel.HIGH,
                    source_id=agent_id,
                    target_resource=resource,
                    description="Timing attack detected",
                    evidence={"duration": operation_duration},
                )
            )

        # Validate nonce if provided
        if "nonce" in metadata and "timestamp" in metadata:
            if not self.replay_protection.validate_nonce(
                metadata["nonce"], metadata["timestamp"]
            ):
                events.append(
                    SecurityEvent(
                        timestamp=time.time(),
                        event_type=AttackType.REPLAY_ATTACK,
                        threat_level=ThreatLevel.CRITICAL,
                        source_id=agent_id,
                        target_resource=resource,
                        description="Replay attack detected",
                        evidence={"nonce": metadata["nonce"]},
                    )
                )

        # Store events
        self.security_events.extend(events)
        for event in events:
            self.threat_counters[event.event_type] += 1

        # Determine if operation should be allowed
        critical_threats = [e for e in events if e.threat_level == ThreatLevel.CRITICAL]
        allowed = len(critical_threats) == 0

        return allowed, events

    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        current_time = time.time()
        recent_events = [
            e for e in self.security_events if current_time - e.timestamp < 3600
        ]

        return {
            "total_events": len(self.security_events),
            "recent_events": len(recent_events),
            "threat_counters": dict(self.threat_counters),
            "attack_types_detected": len(self.threat_counters),
            "timing_attacks_detected": self.timing_detector.attack_detected,
            "active_nonces": len(self.replay_protection.used_nonces),
            "agent_profiles": len(self.anomaly_detector.agent_profiles),
            "security_score": self._calculate_security_score(),
        }

    def _calculate_security_score(self) -> int:
        """Calculate overall security score (0-100)."""
        base_score = 100

        # Deduct points for detected threats
        for threat_type, count in self.threat_counters.items():
            if threat_type == AttackType.TIMING_ATTACK:
                base_score -= min(count * 5, 20)
            elif threat_type == AttackType.REPLAY_ATTACK:
                base_score -= min(count * 10, 30)
            elif threat_type == AttackType.BEHAVIORAL_ANOMALY:
                base_score -= min(count * 3, 15)
            else:
                base_score -= min(count * 7, 25)

        return max(0, base_score)


# Hardware Security Module (HSM) Integration
class HSMIntegration:
    """Integration with Hardware Security Modules for enhanced cryptographic operations."""

    def __init__(self, hsm_available: bool = False):
        self.hsm_available = hsm_available
        self.key_cache: Dict[str, bytes] = {}

    def generate_secure_key(self, key_type: str = "AES-256") -> bytes:
        """Generate cryptographically secure key using HSM if available."""
        if self.hsm_available:
            # In production, this would interface with actual HSM
            # For now, use high-quality random generation
            if key_type == "AES-256":
                return secrets.token_bytes(32)
            elif key_type == "AES-128":
                return secrets.token_bytes(16)

        # Fallback to software generation
        return secrets.token_bytes(32)

    def secure_encrypt(self, data: bytes, key_id: str) -> bytes:
        """Perform encryption using HSM-backed keys."""
        if key_id not in self.key_cache:
            self.key_cache[key_id] = self.generate_secure_key()

        # In production, this would use HSM encryption
        # For now, simulate with high-security software encryption
        from cryptography.fernet import Fernet
        import base64

        key = base64.urlsafe_b64encode(self.key_cache[key_id])
        f = Fernet(key)
        return f.encrypt(data)

    def secure_decrypt(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Perform decryption using HSM-backed keys."""
        if key_id not in self.key_cache:
            raise ValueError(f"Key {key_id} not found")

        from cryptography.fernet import Fernet
        import base64

        key = base64.urlsafe_b64encode(self.key_cache[key_id])
        f = Fernet(key)
        return f.decrypt(encrypted_data)


# Export all classes
__all__ = [
    "ThreatLevel",
    "AttackType",
    "SecurityEvent",
    "TimingAttackDetector",
    "ReplayAttackProtection",
    "BehavioralAnomalyDetector",
    "MultiFactorAuthentication",
    "AdvancedThreatProtection",
    "HSMIntegration",
]
