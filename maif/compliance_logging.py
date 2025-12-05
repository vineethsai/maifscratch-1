"""
Enhanced Compliance Logging for MAIF with STIG/FIPS Validation and SIEM Integration
==================================================================================

Implements comprehensive compliance logging with:
- STIG/FIPS-140-2 validation
- SIEM integration (CloudWatch, Splunk, Elastic)
- Tamper-evident audit trails
- Compliance framework validation (HIPAA, FISMA, etc.)
"""

import os
import time
import json
import sqlite3
import hashlib
import uuid
import tempfile
import shutil
import threading
import logging
import requests
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from enum import Enum
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import base64

# AWS imports for CloudWatch integration
try:
    import boto3

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# Import MAIF modules
from .block_storage import BlockStorage, BlockType
from .signature_verification import create_default_verifier, sign_block_data


class LogLevel(Enum):
    """Log levels for compliance logging."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogCategory(Enum):
    """Log categories for compliance events."""

    ACCESS = "access"
    DATA = "data"
    SECURITY = "security"
    ADMIN = "admin"
    SYSTEM = "system"
    COMPLIANCE = "compliance"


class ComplianceLevel(Enum):
    """Compliance levels for different security requirements."""

    BASIC = "basic"
    FIPS_140_2 = "fips_140_2"
    STIG = "stig"
    FISMA_LOW = "fisma_low"
    FISMA_MODERATE = "fisma_moderate"
    FISMA_HIGH = "fisma_high"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    HIPAA = "hipaa"
    FISMA = "fisma"
    DISA_STIG = "disa_stig"
    PCI_DSS = "pci_dss"
    NIST_800_53 = "nist_800_53"
    ISO_27001 = "iso_27001"


class AuditEventType(Enum):
    """Types of audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    ERROR = "error"
    COMPLIANCE = "compliance"


@dataclass
class AuditEvent:
    """Enhanced audit event with compliance metadata."""

    event_type: AuditEventType
    action: str
    user_id: str
    resource_id: Optional[str] = None
    classification: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    event_id: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "action": self.action,
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "classification": self.classification,
            "success": self.success,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "session_id": self.session_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def to_cef(self) -> str:
        """Convert to Common Event Format for SIEM."""
        severity = 3 if self.success else 6
        return (
            f"CEF:0|MAIF|ComplianceLogger|1.0|{self.event_type.value}|"
            f"{self.action}|{severity}|"
            f"act={self.action} suser={self.user_id} "
            f"cs1={self.resource_id} cs1Label=ResourceID "
            f"outcome={'success' if self.success else 'failure'}"
        )

    def to_syslog(self, facility: int = 16, severity: int = 6) -> str:
        """Convert to syslog format."""
        priority = facility * 8 + severity
        timestamp = self.timestamp.strftime("%b %d %H:%M:%S")
        hostname = os.uname().nodename
        return (
            f"<{priority}>{timestamp} {hostname} MAIF[{os.getpid()}]: {self.to_json()}"
        )


@dataclass
class AuditLogEntry(AuditEvent):
    """Extended audit log entry with additional metadata."""

    user_agent: Optional[str] = None
    integrity_hash: Optional[str] = None

    def calculate_integrity_hash(self) -> str:
        """Calculate integrity hash for tamper detection."""
        data = json.dumps(self.to_dict(), sort_keys=True).encode()
        self.integrity_hash = hashlib.sha256(data).hexdigest()
        return self.integrity_hash


@dataclass
class LogEntry:
    """Compliance log entry."""

    timestamp: float
    level: LogLevel
    category: LogCategory
    user_id: str
    action: str
    resource_id: str
    details: Dict[str, Any]
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    previous_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "category": self.category.value,
            "user_id": self.user_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            level=LogLevel(data.get("level", LogLevel.INFO.value)),
            category=LogCategory(data.get("category", LogCategory.SYSTEM.value)),
            user_id=data.get("user_id", ""),
            action=data.get("action", ""),
            resource_id=data.get("resource_id", ""),
            details=data.get("details", {}),
            previous_hash=data.get("previous_hash"),
        )

    def calculate_hash(self) -> str:
        """Calculate cryptographic hash of log entry."""
        # Create canonical representation
        canonical = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "category": self.category.value,
            "user_id": self.user_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "details": json.dumps(self.details, sort_keys=True),
            "previous_hash": self.previous_hash,
        }

        # Convert to JSON string
        canonical_json = json.dumps(canonical, sort_keys=True)

        # Calculate hash
        return hashlib.sha256(canonical_json.encode()).hexdigest()


class SIEMIntegration:
    """SIEM integration for sending audit logs to external systems."""

    def __init__(self, provider: str, config: Dict[str, Any]):
        """
        Initialize SIEM integration.

        Args:
            provider: SIEM provider (cloudwatch, splunk, elastic)
            config: Provider-specific configuration
        """
        self.provider = provider.lower()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._event_buffer: List[AuditEvent] = []
        self._buffer_lock = threading.RLock()

        # Initialize provider-specific clients
        if self.provider == "cloudwatch" and AWS_AVAILABLE:
            self.client = boto3.client(
                "logs", region_name=config.get("region", "us-east-1")
            )
            self.log_group = config.get("log_group", "/aws/maif/compliance")
            self.log_stream = config.get("log_stream", "audit-logs")
            self._ensure_cloudwatch_setup()

    def _ensure_cloudwatch_setup(self):
        """Ensure CloudWatch log group and stream exist."""
        try:
            self.client.create_log_group(logGroupName=self.log_group)
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass

        try:
            self.client.create_log_stream(
                logGroupName=self.log_group, logStreamName=self.log_stream
            )
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass

    def send_event(self, event: AuditEvent):
        """Send audit event to SIEM."""
        batch_size = self.config.get("batch_size", 1)

        if batch_size > 1:
            with self._buffer_lock:
                self._event_buffer.append(event)
                if len(self._event_buffer) >= batch_size:
                    self._flush_batch()
        else:
            self._send_with_retry(event)

    def _send_with_retry(self, event: AuditEvent):
        """Send event with retry logic."""
        retry_count = self.config.get("retry_count", 0)

        for attempt in range(retry_count + 1):
            try:
                if self.provider == "cloudwatch":
                    self._send_to_cloudwatch(event)
                elif self.provider == "splunk":
                    self._send_to_splunk(event)
                elif self.provider == "elastic":
                    self._send_to_elastic(event)
                else:
                    self.logger.warning(f"Unknown SIEM provider: {self.provider}")
                return
            except Exception as e:
                if attempt == retry_count:
                    self.logger.error(
                        f"Failed to send to SIEM after {retry_count} retries: {e}"
                    )
                    self._handle_send_failure(event)
                else:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff

    def _send_to_cloudwatch(self, event: Union[AuditEvent, List[AuditEvent]]):
        """Send event(s) to AWS CloudWatch."""
        if not AWS_AVAILABLE:
            self.logger.error("boto3 not available for CloudWatch integration")
            return

        events = event if isinstance(event, list) else [event]
        log_events = [
            {"timestamp": int(e.timestamp.timestamp() * 1000), "message": e.to_json()}
            for e in events
        ]

        try:
            self.client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=log_events,
            )
        except Exception as e:
            # Re-raise to trigger retry logic
            raise e

    def _send_to_splunk(self, event: AuditEvent):
        """Send event to Splunk via HTTP Event Collector."""
        hec_url = self.config.get("hec_url")
        hec_token = self.config.get("hec_token")

        if not hec_url or not hec_token:
            self.logger.error("Splunk HEC URL and token required")
            return

        headers = {
            "Authorization": f"Splunk {hec_token}",
            "Content-Type": "application/json",
        }

        data = {
            "time": event.timestamp.timestamp(),
            "host": os.uname().nodename,
            "source": "maif-compliance",
            "sourcetype": "_json",
            "index": self.config.get("index", "main"),
            "event": event.to_dict(),
        }

        try:
            response = requests.post(
                hec_url,
                headers=headers,
                data=json.dumps(data),
                verify=self.config.get("verify_ssl", True),
                timeout=self.config.get("timeout", 30),
            )
            response.raise_for_status()
        except Exception as e:
            # Re-raise to trigger retry logic
            raise e

    def _send_to_elastic(self, event: AuditEvent):
        """Send event to Elasticsearch."""
        elastic_url = self.config.get("url")
        index = self.config.get("index", "maif-compliance")

        if not elastic_url:
            self.logger.error("Elasticsearch URL required")
            return

        url = f"{elastic_url}/{index}/_doc/{event.event_id}"

        headers = {"Content-Type": "application/json"}

        # Add authentication if configured
        if self.config.get("api_key"):
            headers["Authorization"] = f"ApiKey {self.config['api_key']}"
        elif self.config.get("username") and self.config.get("password"):
            auth_str = f"{self.config['username']}:{self.config['password']}"
            auth_bytes = auth_str.encode("ascii")
            auth_b64 = base64.b64encode(auth_bytes).decode("ascii")
            headers["Authorization"] = f"Basic {auth_b64}"

        try:
            response = requests.put(
                url,
                headers=headers,
                data=event.to_json(),
                verify=self.config.get("verify_ssl", True),
                timeout=self.config.get("timeout", 30),
            )
            response.raise_for_status()
        except Exception as e:
            # Re-raise to trigger retry logic
            raise e

    def _handle_send_failure(self, event: AuditEvent):
        """Handle SIEM send failures with fallback."""
        # Add to buffer for retry
        with self._buffer_lock:
            self._event_buffer.append(event)

        # Write to fallback file if configured
        fallback_file = self.config.get("fallback_file")
        if fallback_file:
            try:
                with open(fallback_file, "a") as f:
                    f.write(event.to_json() + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write to fallback file: {e}")

    def _flush_batch(self):
        """Flush the current batch of events."""
        with self._buffer_lock:
            if not self._event_buffer:
                return
            events = self._event_buffer[:]
            self._event_buffer.clear()

        if self.provider == "cloudwatch":
            # CloudWatch supports batching natively
            try:
                self._send_to_cloudwatch(events)
            except Exception as e:
                self.logger.error(f"Failed to send batch to CloudWatch: {e}")
                for event in events:
                    self._handle_send_failure(event)
        else:
            # Other providers: send individually for now
            for event in events:
                self._send_with_retry(event)

    def flush(self):
        """Flush any buffered events."""
        self._flush_batch()


class EnhancedComplianceLogger:
    """Enhanced compliance logger with STIG/FIPS validation and SIEM integration."""

    def __init__(
        self,
        compliance_level: ComplianceLevel = ComplianceLevel.BASIC,
        frameworks: List[ComplianceFramework] = None,
        siem_config: Optional[Dict[str, Any]] = None,
        retention_days: int = 365,
        enable_tamper_detection: bool = True,
    ):
        """
        Initialize compliance logger.

        Args:
            compliance_level: Required compliance level
            frameworks: List of compliance frameworks to validate against
            siem_config: SIEM integration configuration
            retention_days: Audit log retention period in days
            enable_tamper_detection: Enable integrity checking
        """
        self.compliance_level = compliance_level
        self.frameworks = frameworks or []
        self.retention_days = retention_days
        self.enable_tamper_detection = enable_tamper_detection

        self.audit_events: List[AuditEvent] = []
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        # Initialize SIEM integration if configured
        self.siem: Optional[SIEMIntegration] = None
        if siem_config:
            self.siem = SIEMIntegration(
                provider=siem_config.get("provider"), config=siem_config
            )

        # FIPS-approved algorithms
        self.fips_algorithms = {
            "AES-128-GCM",
            "AES-192-GCM",
            "AES-256-GCM",
            "AES-128-CBC",
            "AES-192-CBC",
            "AES-256-CBC",
            "SHA-256",
            "SHA-384",
            "SHA-512",
            "RSA-2048",
            "RSA-3072",
            "RSA-4096",
            "ECDSA-P256",
            "ECDSA-P384",
            "ECDSA-P521",
        }

    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        user_id: str,
        resource_id: Optional[str] = None,
        classification: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log a compliance event."""
        event = AuditLogEntry(
            event_type=event_type,
            action=action,
            user_id=user_id,
            resource_id=resource_id,
            classification=classification,
            success=success,
            details=details or {},
            **kwargs,
        )

        # Calculate integrity hash if enabled
        if self.enable_tamper_detection and isinstance(event, AuditLogEntry):
            event.calculate_integrity_hash()

        # Store event
        with self._lock:
            self.audit_events.append(event)

        # Send to SIEM if configured
        if self.siem:
            self.siem.send_event(event)

        self.logger.info(f"Logged {event_type.value} event: {action}")
        return event

    def log_data_access(
        self,
        user_id: str,
        resource_id: str,
        access_type: str,
        classification: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
    ) -> AuditEvent:
        """Log data access event."""
        details = metadata or {}
        if failure_reason:
            details["failure_reason"] = failure_reason

        return self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            action=access_type,
            user_id=user_id,
            resource_id=resource_id,
            classification=classification,
            success=success,
            details=details,
        )

    def log_authentication(
        self,
        user_id: str,
        auth_method: str,
        success: bool,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
    ) -> AuditEvent:
        """Log authentication event."""
        details = metadata or {}
        details["auth_method"] = auth_method
        if ip_address:
            details["ip_address"] = ip_address
        if failure_reason:
            details["failure_reason"] = failure_reason

        return self.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            action="authenticate",
            user_id=user_id,
            success=success,
            details=details,
            ip_address=ip_address,
        )

    def validate_fips_compliance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate FIPS-140-2 compliance."""
        result = {"compliant": True, "issues": [], "warnings": []}

        # Check encryption algorithm
        algorithm = config.get("encryption_algorithm", "")
        if algorithm:
            base_algo = algorithm.split("-")[0]  # Get base algorithm
            if algorithm not in self.fips_algorithms and f"{base_algo}-" not in str(
                self.fips_algorithms
            ):
                result["compliant"] = False
                result["issues"].append(f"{algorithm} is not FIPS approved")
            else:
                result["algorithm_approved"] = True

        # Check key length
        key_length = config.get("key_length", 0)
        if key_length:
            if algorithm.startswith("AES") and key_length < 128:
                result["compliant"] = False
                result["issues"].append(f"AES key length {key_length} is too short")
            elif algorithm.startswith("RSA") and key_length < 2048:
                result["compliant"] = False
                result["issues"].append(f"RSA key length {key_length} is too short")

        # Check random source
        random_source = config.get("random_source", "")
        if random_source and random_source not in [
            "/dev/urandom",
            "/dev/random",
            "CryptGenRandom",
        ]:
            result["warnings"].append(
                f"Random source {random_source} may not be FIPS compliant"
            )

        return result

    def validate_stig_requirement(
        self, requirement_id: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate STIG requirement."""
        result = {"requirement_id": requirement_id, "compliant": True, "issues": []}

        # Password complexity (SRG-OS-000069-GPOS-00037)
        if requirement_id == "SRG-OS-000069-GPOS-00037":
            min_length = config.get("min_length", 0)
            if min_length < 15:
                result["compliant"] = False
                result["issues"].append("Password minimum length must be at least 15")

            required_chars = [
                "requires_uppercase",
                "requires_lowercase",
                "requires_numbers",
                "requires_special",
            ]
            for req in required_chars:
                if not config.get(req, False):
                    result["compliant"] = False
                    result["issues"].append(f"Password {req} is not enforced")

        # Audit logging (SRG-OS-000037-GPOS-00015)
        elif requirement_id == "SRG-OS-000037-GPOS-00015":
            if not config.get("audit_enabled", False):
                result["compliant"] = False
                result["issues"].append("Audit logging is not enabled")

            required_events = ["login", "logout", "privilege_escalation"]
            captured_events = config.get("events_captured", [])
            for event in required_events:
                if event not in captured_events:
                    result["compliant"] = False
                    result["issues"].append(f"Required event '{event}' is not captured")

        return result

    def validate_fisma_control(
        self, control_id: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate FISMA control."""
        result = {"control_id": control_id, "compliant": True, "issues": []}

        # Extract control family
        control_family = control_id.split("-")[0]
        family_map = {
            "AC": "Access Control",
            "AU": "Audit and Accountability",
            "IA": "Identification and Authentication",
            "SC": "System and Communications Protection",
        }
        result["control_family"] = family_map.get(control_family, "Unknown")

        # Access Control - Account Management (AC-2)
        if control_id == "AC-2":
            required_fields = [
                "account_types_defined",
                "approval_process",
                "periodic_review",
                "termination_process",
            ]
            for field in required_fields:
                if not config.get(field, False):
                    result["compliant"] = False
                    result["issues"].append(f"{field} is not implemented")

        # Audit Storage Capacity (AU-4)
        elif control_id == "AU-4":
            storage_gb = config.get("storage_capacity_gb", 0)
            if storage_gb < 100:
                result["compliant"] = False
                result["issues"].append("Insufficient audit storage capacity")

            if not config.get("archival_process", False):
                result["compliant"] = False
                result["issues"].append("No archival process defined")

        return result

    def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        classification: Optional[str] = None,
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        with self._lock:
            events = self.audit_events[:]

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if resource_id:
            events = [e for e in events if e.resource_id == resource_id]
        if classification:
            events = [e for e in events if e.classification == classification]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events

    def export_logs(self, format: str = "json") -> str:
        """Export audit logs in specified format."""
        with self._lock:
            events = self.audit_events[:]

        if format == "json":
            export_data = {
                "export_metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_events": len(events),
                    "compliance_level": self.compliance_level.value,
                    "frameworks": [f.value for f in self.frameworks],
                },
                "audit_logs": [e.to_dict() for e in events],
            }
            return json.dumps(export_data, indent=2)

        elif format == "cef":
            cef_lines = []
            for event in events:
                cef_lines.append(event.to_cef())
            return "\n".join(cef_lines)

        elif format == "syslog":
            syslog_lines = []
            for event in events:
                syslog_lines.append(event.to_syslog())
            return "\n".join(syslog_lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def apply_retention_policy(self):
        """Apply retention policy to remove old events."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        with self._lock:
            self.audit_events = [
                e for e in self.audit_events if e.timestamp > cutoff_date
            ]

    def verify_log_integrity(self) -> bool:
        """Verify integrity of audit logs."""
        if not self.enable_tamper_detection:
            return True

        with self._lock:
            for event in self.audit_events:
                if isinstance(event, AuditLogEntry) and event.integrity_hash:
                    # Recalculate hash
                    original_hash = event.integrity_hash
                    event.integrity_hash = None
                    calculated_hash = event.calculate_integrity_hash()

                    if original_hash != calculated_hash:
                        self.logger.error(
                            f"Integrity check failed for event {event.event_id}"
                        )
                        return False

        return True

    def generate_compliance_report(
        self, start_date: datetime, end_date: datetime, include_details: bool = False
    ) -> Dict[str, Any]:
        """Generate compliance report for specified period."""
        # Query events in date range
        events = self.query_events(start_time=start_date, end_time=end_date)

        # Calculate statistics
        event_breakdown = {}
        for event in events:
            event_type = event.event_type.value
            if event_type not in event_breakdown:
                event_breakdown[event_type] = {"total": 0, "successful": 0, "failed": 0}
            event_breakdown[event_type]["total"] += 1
            if event.success:
                event_breakdown[event_type]["successful"] += 1
            else:
                event_breakdown[event_type]["failed"] += 1

        report = {
            "report_generated_at": datetime.utcnow().isoformat(),
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "compliance_level": self.compliance_level.value,
            "frameworks": [f.value for f in self.frameworks],
            "total_events": len(events),
            "event_breakdown": event_breakdown,
            "compliance_status": "compliant"
            if self.verify_log_integrity()
            else "non-compliant",
        }

        if include_details:
            report["events"] = [e.to_dict() for e in events]

        return report


class ComplianceLogger:
    """Compliance logger with SQLite backend."""

    def __init__(self, db_path: Optional[str] = None, maif_path: Optional[str] = None):
        """
        Initialize compliance logger.

        Args:
            db_path: Path to SQLite database file (optional)
            maif_path: Path to MAIF file (optional)
        """
        self.db_path = db_path or ":memory:"
        self.maif_path = maif_path
        self.conn = None
        self.cursor = None
        self.lock = threading.RLock()
        self.last_hash = None
        self.initialized = False

        # Initialize database
        self._initialize_db()

        # Initialize MAIF block storage if path provided
        if self.maif_path:
            self.block_storage = BlockStorage(
                file_path=self.maif_path, enable_mmap=True
            )
        else:
            self.block_storage = None

        # Initialize verifier for signatures
        self.verifier = create_default_verifier()

    def _initialize_db(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Create compliance logs table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_logs (
                entry_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                level INTEGER NOT NULL,
                category TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_id TEXT,
                details TEXT,
                previous_hash TEXT,
                entry_hash TEXT NOT NULL,
                signature TEXT,
                verified INTEGER DEFAULT 0
            )
        """)

        # Create indices for efficient querying
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON compliance_logs(timestamp)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_id ON compliance_logs(user_id)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON compliance_logs(category)"
        )

        self.conn.commit()
        self.initialized = True

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        with self.lock:
            try:
                yield self.cursor
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise e

    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        user_id: str,
        action: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """
        Log a compliance event.

        Args:
            level: Log level
            category: Log category
            user_id: User performing the action
            action: Action being performed
            resource_id: Resource being accessed (optional)
            details: Additional details (optional)

        Returns:
            LogEntry: The created log entry
        """
        with self.lock:
            # Create log entry
            entry = LogEntry(
                timestamp=time.time(),
                level=level,
                category=category,
                user_id=user_id,
                action=action,
                resource_id=resource_id or "",
                details=details or {},
                previous_hash=self.last_hash,
            )

            # Calculate entry hash
            entry_hash = entry.calculate_hash()

            # Sign the entry
            signature = None
            if self.verifier:
                signature_data = json.dumps(
                    {
                        "entry_id": entry.entry_id,
                        "entry_hash": entry_hash,
                        "timestamp": entry.timestamp,
                    },
                    sort_keys=True,
                ).encode()

                signature = sign_block_data(self.verifier, signature_data, "log_entry")

            # Store in database
            with self._transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO compliance_logs 
                    (entry_id, timestamp, level, category, user_id, action, 
                     resource_id, details, previous_hash, entry_hash, signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.entry_id,
                        entry.timestamp,
                        entry.level.value,
                        entry.category.value,
                        entry.user_id,
                        entry.action,
                        entry.resource_id,
                        json.dumps(entry.details),
                        entry.previous_hash,
                        entry_hash,
                        signature,
                    ),
                )

            # Update last hash
            self.last_hash = entry_hash

            # Store in MAIF if available
            if self.block_storage:
                self._store_in_maif(entry, signature)

            return entry

    def _store_in_maif(self, entry: LogEntry, signature: Optional[str]):
        """Store log entry in MAIF file."""
        # Create log block
        log_data = {
            "type": "compliance_log",
            "entry": entry.to_dict(),
            "signature": signature,
        }

        # Add block to MAIF
        self.block_storage.add_block(
            block_type=BlockType.LOG,
            data=json.dumps(log_data).encode(),
            metadata={
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "category": entry.category.value,
            },
        )

    def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        user_id: Optional[str] = None,
        category: Optional[LogCategory] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """
        Query compliance logs.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            user_id: Filter by user ID (optional)
            category: Filter by category (optional)
            limit: Maximum number of results

        Returns:
            List[LogEntry]: Matching log entries
        """
        with self.lock:
            query = "SELECT * FROM compliance_logs WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            if category:
                query += " AND category = ?"
                params.append(category.value)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()

            entries = []
            for row in rows:
                entry_data = {
                    "entry_id": row[0],
                    "timestamp": row[1],
                    "level": row[2],
                    "category": row[3],
                    "user_id": row[4],
                    "action": row[5],
                    "resource_id": row[6],
                    "details": json.loads(row[7]) if row[7] else {},
                    "previous_hash": row[8],
                }
                entries.append(LogEntry.from_dict(entry_data))

            return entries

    def verify_integrity(
        self, start_time: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the log chain.

        Args:
            start_time: Start from this timestamp (optional)

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []

        with self.lock:
            # Get all entries in chronological order
            if start_time:
                cursor = self.conn.execute(
                    "SELECT * FROM compliance_logs WHERE timestamp >= ? ORDER BY timestamp ASC",
                    (start_time,),
                )
            else:
                cursor = self.conn.execute(
                    "SELECT * FROM compliance_logs ORDER BY timestamp ASC"
                )

            rows = cursor.fetchall()

            previous_hash = None
            for row in rows:
                entry_id = row[0]
                stored_previous_hash = row[8]
                stored_entry_hash = row[9]

                # Check previous hash linkage
                if previous_hash and stored_previous_hash != previous_hash:
                    errors.append(
                        f"Entry {entry_id}: Previous hash mismatch. "
                        f"Expected {previous_hash}, got {stored_previous_hash}"
                    )

                # Recreate entry and verify hash
                entry_data = {
                    "entry_id": row[0],
                    "timestamp": row[1],
                    "level": row[2],
                    "category": row[3],
                    "user_id": row[4],
                    "action": row[5],
                    "resource_id": row[6],
                    "details": json.loads(row[7]) if row[7] else {},
                    "previous_hash": row[8],
                }
                entry = LogEntry.from_dict(entry_data)
                calculated_hash = entry.calculate_hash()

                if calculated_hash != stored_entry_hash:
                    errors.append(
                        f"Entry {entry_id}: Hash mismatch. "
                        f"Expected {stored_entry_hash}, calculated {calculated_hash}"
                    )

                previous_hash = stored_entry_hash

        return len(errors) == 0, errors

    def export_to_maif(self, output_path: str, start_time: Optional[float] = None):
        """
        Export logs to a MAIF file.

        Args:
            output_path: Path to output MAIF file
            start_time: Export logs from this timestamp (optional)
        """
        # Create new MAIF file
        export_storage = BlockStorage(file_path=output_path, enable_mmap=False)

        # Query logs
        entries = self.query(start_time=start_time, limit=10000)

        # Add entries to MAIF
        for entry in entries:
            log_data = {"type": "compliance_log", "entry": entry.to_dict()}

            export_storage.add_block(
                block_type=BlockType.LOG,
                data=json.dumps(log_data).encode(),
                metadata={
                    "entry_id": entry.entry_id,
                    "timestamp": entry.timestamp,
                    "category": entry.category.value,
                },
            )

        # Finalize
        export_storage.finalize()

    def get_statistics(self) -> Dict[str, Any]:
        """Get compliance logging statistics."""
        with self.lock:
            # Total entries
            cursor = self.conn.execute("SELECT COUNT(*) FROM compliance_logs")
            total_entries = cursor.fetchone()[0]

            # Entries by category
            cursor = self.conn.execute(
                "SELECT category, COUNT(*) FROM compliance_logs GROUP BY category"
            )
            by_category = dict(cursor.fetchall())

            # Entries by level
            cursor = self.conn.execute(
                "SELECT level, COUNT(*) FROM compliance_logs GROUP BY level"
            )
            by_level = dict(cursor.fetchall())

            # Most active users
            cursor = self.conn.execute(
                "SELECT user_id, COUNT(*) as count FROM compliance_logs "
                "GROUP BY user_id ORDER BY count DESC LIMIT 10"
            )
            top_users = cursor.fetchall()

            return {
                "total_entries": total_entries,
                "by_category": by_category,
                "by_level": by_level,
                "top_users": top_users,
                "database_size": os.path.getsize(self.db_path)
                if self.db_path != ":memory:"
                else 0,
            }
