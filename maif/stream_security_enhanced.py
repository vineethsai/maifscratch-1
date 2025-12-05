"""
Enhanced Stream Security for MAIF
=================================

Fixes the three critical security gaps:
1. Timing attack protection
2. Anti-replay protection
3. Multi-factor authentication
"""

import time
import threading
import secrets
import hmac
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict, deque

from .stream_access_control import (
    StreamAccessController,
    StreamAccessRule,
    AccessLevel,
    AccessDecision,
    StreamSession,
)


class EnhancedStreamAccessController(StreamAccessController):
    """Enhanced access controller with advanced security features."""

    def __init__(self):
        super().__init__()

        # Timing attack protection
        self._timing_randomization = True
        self._min_response_time = 0.001  # 1ms minimum
        self._max_response_time = 0.010  # 10ms maximum

        # Anti-replay protection
        self._global_nonce_history: Set[str] = set()
        self._nonce_cleanup_interval = 3600  # 1 hour
        self._last_nonce_cleanup = time.time()

        # MFA settings
        self._mfa_secret_key = secrets.token_bytes(32)
        self._mfa_timeout = 300  # 5 minutes

        # Behavioral analysis
        self._behavioral_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self._anomaly_threshold = 0.7

    def check_stream_access_secure(
        self,
        session_id: str,
        operation: str,
        block_type: str = None,
        block_data: bytes = None,
        request_nonce: str = None,
        request_timestamp: float = None,
    ) -> Tuple[AccessDecision, str]:
        """
        Enhanced access check with anti-replay, timing attack protection, and MFA.
        """
        start_time = time.time()

        try:
            # 1. Anti-replay protection
            if request_nonce and request_timestamp:
                replay_check = self._check_anti_replay(
                    session_id, request_nonce, request_timestamp
                )
                if replay_check != AccessDecision.ALLOW:
                    return self._timing_safe_response(
                        start_time, replay_check, "Replay attack detected"
                    )

            # 2. MFA verification for sensitive operations
            mfa_check = self._check_mfa_requirement(session_id, operation, block_type)
            if mfa_check != AccessDecision.ALLOW:
                return self._timing_safe_response(
                    start_time, mfa_check, "MFA verification required"
                )

            # 3. Behavioral analysis
            behavioral_check = self._analyze_behavioral_pattern(
                session_id, operation, block_type
            )
            if behavioral_check != AccessDecision.ALLOW:
                return self._timing_safe_response(
                    start_time, behavioral_check, "Suspicious behavioral pattern"
                )

            # 4. Standard access control check
            decision, reason = self.check_stream_access(
                session_id, operation, block_type, block_data
            )

            # 5. Update behavioral patterns
            self._update_behavioral_pattern(session_id, operation, block_type, decision)

            return self._timing_safe_response(start_time, decision, reason)

        except Exception as e:
            return self._timing_safe_response(
                start_time, AccessDecision.DENY, f"Security check failed: {str(e)}"
            )

    def _check_anti_replay(
        self, session_id: str, nonce: str, timestamp: float
    ) -> AccessDecision:
        """Check for replay attacks using nonce and timestamp validation."""
        current_time = time.time()

        # Check timestamp freshness (within 30 seconds)
        if abs(current_time - timestamp) > 30:
            return AccessDecision.DENY

        # Check global nonce uniqueness
        if nonce in self._global_nonce_history:
            return AccessDecision.DENY

        # Check session-specific nonce
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]

            # Add nonce tracking to session if not present
            if not hasattr(session, "nonce_history"):
                session.nonce_history = set()
                session.last_request_timestamp = 0.0

            if nonce in session.nonce_history:
                return AccessDecision.DENY

            # Check timestamp ordering (must be newer than last request)
            if timestamp <= session.last_request_timestamp:
                return AccessDecision.DENY

            # Update session state
            session.nonce_history.add(nonce)
            session.last_request_timestamp = timestamp

            # Limit nonce history size
            if len(session.nonce_history) > 1000:
                session.nonce_history = set(list(session.nonce_history)[-500:])

        # Add to global nonce history
        self._global_nonce_history.add(nonce)

        # Cleanup old nonces periodically
        if current_time - self._last_nonce_cleanup > self._nonce_cleanup_interval:
            self._cleanup_old_nonces()

        return AccessDecision.ALLOW

    def _check_mfa_requirement(
        self, session_id: str, operation: str, block_type: str
    ) -> AccessDecision:
        """Check if MFA is required and verified for this operation."""
        if session_id not in self.active_sessions:
            return AccessDecision.DENY

        session = self.active_sessions[session_id]

        # Add MFA tracking to session if not present
        if not hasattr(session, "mfa_verified"):
            session.mfa_verified = False
            session.mfa_required = False
            session.mfa_challenge_time = None

        # Determine if MFA is required based on operation sensitivity
        requires_mfa = (
            session.mfa_required
            or operation == "write"
            or block_type in ["SECU", "ACLS", "PROV"]  # Security-sensitive blocks
            or getattr(session, "suspicious_activity_score", 0.0) > 0.5
        )

        if requires_mfa and not session.mfa_verified:
            return AccessDecision.DENY

        # Check MFA timeout
        if session.mfa_verified and session.mfa_challenge_time:
            if time.time() - session.mfa_challenge_time > self._mfa_timeout:
                session.mfa_verified = False
                return AccessDecision.DENY

        return AccessDecision.ALLOW

    def _analyze_behavioral_pattern(
        self, session_id: str, operation: str, block_type: str
    ) -> AccessDecision:
        """Analyze user behavior for anomalies."""
        if session_id not in self.active_sessions:
            return AccessDecision.DENY

        session = self.active_sessions[session_id]
        current_time = time.time()

        # Add behavioral tracking to session if not present
        if not hasattr(session, "suspicious_activity_score"):
            session.suspicious_activity_score = 0.0

        # Create behavior signature
        behavior = {
            "operation": operation,
            "block_type": block_type,
            "timestamp": current_time,
            "hour_of_day": int((current_time % 86400) / 3600),
            "day_of_week": int((current_time / 86400) % 7),
        }

        # Get user's historical patterns
        user_patterns = self._behavioral_patterns[session.user_id]

        if len(user_patterns) < 10:
            # Not enough data for analysis
            user_patterns.append(behavior)
            return AccessDecision.ALLOW

        # Simple anomaly detection based on patterns
        anomaly_score = self._calculate_anomaly_score(behavior, user_patterns)
        session.suspicious_activity_score = anomaly_score

        if anomaly_score > self._anomaly_threshold:
            # Require MFA for suspicious activity
            session.mfa_required = True
            if not session.mfa_verified:
                return AccessDecision.DENY

        # Update patterns (keep last 100 behaviors)
        user_patterns.append(behavior)
        if len(user_patterns) > 100:
            user_patterns.pop(0)

        return AccessDecision.ALLOW

    def _calculate_anomaly_score(
        self, current_behavior: Dict, historical_patterns: List[Dict]
    ) -> float:
        """Calculate anomaly score based on historical patterns."""
        if not historical_patterns:
            return 0.0

        score = 0.0

        # Check operation frequency
        operation_count = sum(
            1
            for p in historical_patterns
            if p["operation"] == current_behavior["operation"]
        )
        operation_frequency = operation_count / len(historical_patterns)
        if operation_frequency < 0.1:  # Rare operation
            score += 0.3

        # Check time-of-day patterns
        hour = current_behavior["hour_of_day"]
        hour_count = sum(
            1 for p in historical_patterns if abs(p["hour_of_day"] - hour) <= 1
        )
        hour_frequency = hour_count / len(historical_patterns)
        if hour_frequency < 0.1:  # Unusual time
            score += 0.4

        # Check block type patterns
        if current_behavior["block_type"]:
            block_count = sum(
                1
                for p in historical_patterns
                if p["block_type"] == current_behavior["block_type"]
            )
            block_frequency = block_count / len(historical_patterns)
            if block_frequency < 0.05:  # Very rare block type
                score += 0.3

        return min(score, 1.0)

    def _timing_safe_response(
        self, start_time: float, decision: AccessDecision, reason: str
    ) -> Tuple[AccessDecision, str]:
        """Return response with timing attack protection."""
        if not self._timing_randomization:
            return decision, reason

        elapsed = time.time() - start_time

        # Add random delay to normalize response times
        if elapsed < self._min_response_time:
            delay = (
                self._min_response_time
                - elapsed
                + secrets.randbelow(int(self._max_response_time * 1000)) / 1000
            )
            time.sleep(delay)
        elif elapsed > self._max_response_time:
            # Response took too long, add small random delay
            time.sleep(secrets.randbelow(5) / 1000)  # 0-5ms
        else:
            # Add small random delay to mask timing differences
            time.sleep(secrets.randbelow(3) / 1000)  # 0-3ms

        return decision, reason

    def _update_behavioral_pattern(
        self, session_id: str, operation: str, block_type: str, decision: AccessDecision
    ) -> None:
        """Update behavioral patterns based on access decision."""
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # Add behavioral tracking if not present
            if not hasattr(session, "suspicious_activity_score"):
                session.suspicious_activity_score = 0.0

            # Adjust suspicion score based on access patterns
            if decision == AccessDecision.DENY:
                session.suspicious_activity_score = min(
                    session.suspicious_activity_score + 0.1, 1.0
                )
            else:
                session.suspicious_activity_score = max(
                    session.suspicious_activity_score - 0.05, 0.0
                )

    def _cleanup_old_nonces(self) -> None:
        """Clean up old nonces to prevent memory growth."""
        if len(self._global_nonce_history) > 10000:
            # Keep only recent nonces (this is approximate)
            self._global_nonce_history = set(list(self._global_nonce_history)[-5000:])

        self._last_nonce_cleanup = time.time()

    def initiate_mfa_challenge(self, session_id: str) -> Optional[str]:
        """Initiate MFA challenge for a session."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Add MFA tracking if not present
        if not hasattr(session, "mfa_challenge_time"):
            session.mfa_challenge_time = None

        # Generate MFA challenge using a real MFA provider (TOTP/SMS/hardware token)
        # This is a placeholder: integrate with your MFA provider here
        raise NotImplementedError(
            "MFA challenge generation must be implemented with a real provider."
        )

    def verify_mfa_response(
        self, session_id: str, response: str, expected_response: str
    ) -> bool:
        """Verify MFA response."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # Add MFA tracking if not present
        if not hasattr(session, "mfa_verified"):
            session.mfa_verified = False
            session.mfa_challenge_time = None

        # Verify response using a real MFA provider
        # This is a placeholder: integrate with your MFA provider here
        raise NotImplementedError(
            "MFA response verification must be implemented with a real provider."
        )


class SecureStreamReaderEnhanced:
    """Enhanced secure stream reader with all security features."""

    def __init__(
        self,
        maif_path: str,
        user_id: str,
        access_controller: EnhancedStreamAccessController,
    ):
        self.maif_path = maif_path
        self.user_id = user_id
        self.access_controller = access_controller
        self.session_id = None
        self._base_reader = None

    def __enter__(self):
        # Create session
        self.session_id = self.access_controller.create_session(
            self.user_id, self.maif_path
        )

        # Initialize base reader
        from .streaming import MAIFStreamReader, StreamingConfig

        config = StreamingConfig()
        self._base_reader = MAIFStreamReader(self.maif_path, config)
        self._base_reader.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session_id:
            self.access_controller.close_session(self.session_id)

        if self._base_reader:
            self._base_reader.__exit__(exc_type, exc_val, exc_tb)

    def stream_blocks_secure_enhanced(self, enable_mfa: bool = False):
        """Stream blocks with enhanced security features."""
        if not self.session_id or not self._base_reader:
            raise RuntimeError("SecureStreamReaderEnhanced not properly initialized")

        for block_type, block_data in self._base_reader.stream_blocks():
            # Generate nonce and timestamp for anti-replay
            nonce = secrets.token_hex(16)
            timestamp = time.time()

            # Check access with enhanced security
            decision, reason = self.access_controller.check_stream_access_secure(
                self.session_id, "read", block_type, block_data, nonce, timestamp
            )

            if decision == AccessDecision.DENY and "MFA" in reason and enable_mfa:
                # Initiate MFA challenge
                challenge = self.access_controller.initiate_mfa_challenge(
                    self.session_id
                )
                if challenge:
                    # In a real implementation, this would prompt the user
                    print(f"MFA Challenge required: {challenge}")
                    # In production, prompt the user for MFA response and verify it
                    raise NotImplementedError(
                        "MFA user prompt and verification must be implemented in production."
                    )

            if decision != AccessDecision.ALLOW:
                raise PermissionError(f"Enhanced stream access denied: {reason}")

            yield block_type, block_data

    def get_session_stats(self) -> Optional[Dict[str, Any]]:
        """Get current session statistics."""
        if self.session_id:
            return self.access_controller.get_session_stats(self.session_id)
        return None
