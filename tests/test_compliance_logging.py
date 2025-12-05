"""
Comprehensive tests for MAIF compliance logging functionality.
"""

import pytest
import json
import time
import os
import requests
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import logging

from maif.compliance_logging import (
    ComplianceLogger,
    EnhancedComplianceLogger,
    SIEMIntegration,
    AuditEvent,
    ComplianceLevel,
    AuditEventType,
    ComplianceFramework,
    AuditLogEntry,
)


class TestComplianceLogger:
    """Test ComplianceLogger functionality."""

    def test_compliance_logger_initialization(self):
        """Test basic ComplianceLogger initialization."""
        logger = EnhancedComplianceLogger(
            compliance_level=ComplianceLevel.FIPS_140_2,
            frameworks=[ComplianceFramework.HIPAA, ComplianceFramework.FISMA],
        )

        assert logger.compliance_level == ComplianceLevel.FIPS_140_2
        assert ComplianceFramework.HIPAA in logger.frameworks
        assert ComplianceFramework.FISMA in logger.frameworks
        assert len(logger.audit_events) == 0

    def test_log_audit_event(self):
        """Test logging audit events."""
        logger = EnhancedComplianceLogger()

        # Log a security event
        event = logger.log_event(
            event_type=AuditEventType.SECURITY,
            action="encryption_key_rotation",
            user_id="admin-001",
            resource_id="key-123",
            details={
                "old_key_id": "key-122",
                "new_key_id": "key-123",
                "algorithm": "AES-256",
            },
            classification="CONFIDENTIAL",
        )

        assert event.event_type == AuditEventType.SECURITY
        assert event.action == "encryption_key_rotation"
        assert event.user_id == "admin-001"
        assert event.resource_id == "key-123"
        assert event.classification == "CONFIDENTIAL"
        assert event.details["algorithm"] == "AES-256"
        assert event.timestamp is not None
        assert event.event_id is not None

        # Verify event was stored
        assert len(logger.audit_events) == 1
        assert logger.audit_events[0] == event

    def test_log_data_access_event(self):
        """Test logging data access events."""
        logger = EnhancedComplianceLogger()

        # Log read access
        read_event = logger.log_data_access(
            user_id="analyst-001",
            resource_id="doc-456",
            access_type="read",
            classification="SECRET",
            success=True,
            metadata={"ip_address": "192.168.1.100"},
        )

        assert read_event.event_type == AuditEventType.DATA_ACCESS
        assert read_event.action == "read"
        assert read_event.success is True
        assert read_event.details["ip_address"] == "192.168.1.100"

        # Log write access failure
        write_event = logger.log_data_access(
            user_id="user-002",
            resource_id="doc-789",
            access_type="write",
            classification="TOP_SECRET",
            success=False,
            failure_reason="Insufficient clearance",
        )

        assert write_event.event_type == AuditEventType.DATA_ACCESS
        assert write_event.action == "write"
        assert write_event.success is False
        assert write_event.details["failure_reason"] == "Insufficient clearance"

    def test_log_authentication_event(self):
        """Test logging authentication events."""
        logger = EnhancedComplianceLogger()

        # Successful authentication
        auth_event = logger.log_authentication(
            user_id="user-001",
            auth_method="PKI_CERTIFICATE",
            success=True,
            ip_address="10.0.0.1",
            metadata={"certificate_serial": "ABC123"},
        )

        assert auth_event.event_type == AuditEventType.AUTHENTICATION
        assert auth_event.details["auth_method"] == "PKI_CERTIFICATE"
        assert auth_event.details["ip_address"] == "10.0.0.1"
        assert auth_event.success is True

        # Failed authentication
        fail_event = logger.log_authentication(
            user_id="attacker-001",
            auth_method="password",
            success=False,
            ip_address="1.2.3.4",
            failure_reason="Invalid credentials",
        )

        assert fail_event.success is False
        assert fail_event.details["failure_reason"] == "Invalid credentials"

    def test_compliance_validation(self):
        """Test compliance validation for different frameworks."""
        logger = EnhancedComplianceLogger(
            compliance_level=ComplianceLevel.FIPS_140_2,
            frameworks=[ComplianceFramework.HIPAA],
        )

        # Test FIPS compliance check
        fips_result = logger.validate_fips_compliance(
            {
                "encryption_algorithm": "AES-256-GCM",
                "key_length": 256,
                "mode": "GCM",
                "random_source": "/dev/urandom",
            }
        )

        assert fips_result["compliant"] is True
        assert fips_result["algorithm_approved"] is True

        # Test non-compliant algorithm
        non_fips = logger.validate_fips_compliance(
            {"encryption_algorithm": "DES", "key_length": 56}
        )

        assert non_fips["compliant"] is False
        assert "DES is not FIPS approved" in non_fips["issues"][0]

    def test_query_audit_logs(self):
        """Test querying audit logs with filters."""
        logger = EnhancedComplianceLogger()

        # Add various events
        logger.log_event(AuditEventType.SECURITY, "key_create", "admin-001")
        logger.log_event(
            AuditEventType.DATA_ACCESS, "read", "user-001", resource_id="doc-1"
        )
        logger.log_event(
            AuditEventType.DATA_ACCESS, "write", "user-002", resource_id="doc-2"
        )
        logger.log_authentication("user-001", "password", True)

        # Query by event type
        security_events = logger.query_events(event_type=AuditEventType.SECURITY)
        assert len(security_events) == 1
        assert security_events[0].action == "key_create"

        # Query by user
        user_events = logger.query_events(user_id="user-001")
        assert len(user_events) == 2

        # Query by time range
        start_time = datetime.utcnow() - timedelta(minutes=5)
        end_time = datetime.utcnow() + timedelta(minutes=5)
        time_events = logger.query_events(start_time=start_time, end_time=end_time)
        assert len(time_events) == 4

    def test_export_audit_logs(self):
        """Test exporting audit logs in various formats."""
        logger = EnhancedComplianceLogger()

        # Add test events
        logger.log_event(AuditEventType.SECURITY, "encryption", "user-001")
        logger.log_data_access("user-002", "doc-123", "read", "CONFIDENTIAL", True)

        # Export as JSON
        json_export = logger.export_logs(format="json")
        logs = json.loads(json_export)
        assert len(logs["audit_logs"]) == 2
        assert logs["export_metadata"]["total_events"] == 2

        # Export as CEF (Common Event Format)
        cef_export = logger.export_logs(format="cef")
        assert "CEF:0|MAIF|ComplianceLogger|" in cef_export

        # Export as syslog format
        syslog_export = logger.export_logs(format="syslog")
        assert "<" in syslog_export  # Syslog priority

    def test_retention_policy(self):
        """Test audit log retention policies."""
        # Create logger with 30-day retention
        logger = EnhancedComplianceLogger(retention_days=30)

        # Add old event (mock timestamp)
        old_event = AuditEvent(
            event_type=AuditEventType.SECURITY, action="old_action", user_id="user-001"
        )
        old_event.timestamp = datetime.utcnow() - timedelta(days=45)
        logger.audit_events.append(old_event)

        # Add recent event
        logger.log_event(AuditEventType.SECURITY, "recent_action", "user-002")

        # Apply retention policy
        logger.apply_retention_policy()

        # Old event should be removed
        assert len(logger.audit_events) == 1
        assert logger.audit_events[0].action == "recent_action"

    def test_tamper_detection(self):
        """Test audit log tamper detection."""
        logger = EnhancedComplianceLogger(enable_tamper_detection=True)

        # Log event with tamper detection
        event = logger.log_event(
            AuditEventType.SECURITY, "sensitive_operation", "admin-001"
        )

        # Verify event has integrity hash
        assert hasattr(event, "integrity_hash")
        assert event.integrity_hash is not None

        # Verify integrity check passes
        assert logger.verify_log_integrity() is True

        # Tamper with event
        event.action = "tampered_action"

        # Integrity check should fail
        assert logger.verify_log_integrity() is False


class TestSIEMIntegration:
    """Test SIEM integration functionality."""

    @patch("boto3.client")
    def test_cloudwatch_integration(self, mock_boto_client):
        """Test AWS CloudWatch SIEM integration."""
        mock_logs_client = MagicMock()
        mock_boto_client.return_value = mock_logs_client

        siem = SIEMIntegration(
            provider="cloudwatch",
            config={
                "log_group": "/aws/maif/compliance",
                "log_stream": "audit-logs",
                "region": "us-east-1",
            },
        )

        # Create audit event
        event = AuditEvent(
            event_type=AuditEventType.SECURITY,
            action="key_rotation",
            user_id="admin-001",
            resource_id="key-456",
            classification="SECRET",
        )

        # Send to SIEM
        siem.send_event(event)

        # Verify CloudWatch was called
        mock_logs_client.put_log_events.assert_called_once()
        call_args = mock_logs_client.put_log_events.call_args[1]
        assert call_args["logGroupName"] == "/aws/maif/compliance"
        assert call_args["logStreamName"] == "audit-logs"
        assert len(call_args["logEvents"]) == 1

        # Verify event format
        log_event = call_args["logEvents"][0]
        assert "timestamp" in log_event
        assert "message" in log_event
        message = json.loads(log_event["message"])
        assert message["event_type"] == "security"
        assert message["classification"] == "SECRET"

    @patch("requests.post")
    def test_splunk_integration(self, mock_post):
        """Test Splunk SIEM integration."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        siem = SIEMIntegration(
            provider="splunk",
            config={
                "hec_url": "https://splunk.example.com:8088/services/collector",
                "hec_token": "test-token",
                "index": "maif_compliance",
            },
        )

        # Create and send event
        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            action="read",
            user_id="analyst-001",
            resource_id="doc-789",
        )

        siem.send_event(event)

        # Verify Splunk HEC was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://splunk.example.com:8088/services/collector"

        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Splunk test-token"

        data = json.loads(call_args[1]["data"])
        assert data["index"] == "maif_compliance"
        assert data["event"]["event_type"] == "data_access"

    def test_elasticsearch_integration(self):
        """Test Elasticsearch SIEM integration."""
        with patch("requests.put") as mock_put:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_put.return_value = mock_response

            siem = SIEMIntegration(
                provider="elastic",
                config={
                    "url": "https://localhost:9200",
                    "index": "maif-compliance",
                    "api_key": "test-api-key",
                },
            )

            # Send test event
            event = AuditEvent(
                event_type=AuditEventType.DATA_ACCESS,
                action="view_document",
                user_id="user-789",
                resource_id="doc-123",
            )

            siem.send_event(event)

            # Verify API call
            assert mock_put.called
            call_args = mock_put.call_args

            # Check URL
            expected_url = (
                f"https://localhost:9200/maif-compliance/_doc/{event.event_id}"
            )
            assert call_args[0][0] == expected_url

            # Check headers
            headers = call_args[1]["headers"]
            assert headers["Content-Type"] == "application/json"
            assert headers["Authorization"] == "ApiKey test-api-key"

            # Check data
            assert call_args[1]["data"] == event.to_json()

    def test_elasticsearch_basic_auth(self):
        """Test Elasticsearch with basic authentication."""
        with patch("requests.put") as mock_put:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_put.return_value = mock_response

            siem = SIEMIntegration(
                provider="elastic",
                config={
                    "url": "https://localhost:9200",
                    "index": "maif-compliance",
                    "username": "elastic",
                    "password": "changeme",
                },
            )

            # Send test event
            event = AuditEvent(
                event_type=AuditEventType.SECURITY,
                action="login_attempt",
                user_id="admin-001",
            )

            siem.send_event(event)

            # Verify basic auth header
            headers = mock_put.call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"].startswith("Basic ")

            # Decode and verify credentials
            import base64

            auth_header = headers["Authorization"].replace("Basic ", "")
            decoded = base64.b64decode(auth_header).decode("ascii")
            assert decoded == "elastic:changeme"

    def test_elasticsearch_failure_handling(self):
        """Test Elasticsearch integration failure handling."""
        with patch("requests.put") as mock_put:
            # Mock network error
            mock_put.side_effect = requests.RequestException("Connection refused")

            siem = SIEMIntegration(
                provider="elastic",
                config={
                    "url": "https://localhost:9200",
                    "index": "maif-compliance",
                    "fallback_file": "/tmp/elastic_fallback.log",
                },
            )

            # Send event that will fail
            event = AuditEvent(
                event_type=AuditEventType.SECURITY,
                action="critical_operation",
                user_id="admin-002",
            )

            # Should not raise exception
            siem.send_event(event)

            # Verify error was logged (event should be in buffer)
            assert mock_put.called

    def test_siem_batch_sending(self):
        """Test batch sending of events to SIEM."""
        with patch("boto3.client") as mock_boto:
            mock_logs_client = MagicMock()
            mock_boto.return_value = mock_logs_client

            siem = SIEMIntegration(
                provider="cloudwatch",
                config={
                    "log_group": "/aws/maif/compliance",
                    "batch_size": 10,
                    "batch_timeout": 60,
                },
            )

            # Send multiple events
            for i in range(15):
                event = AuditEvent(
                    event_type=AuditEventType.SECURITY,
                    action=f"action_{i}",
                    user_id=f"user-{i:03d}",
                )
                siem.send_event(event)

            # Should have sent 1 batch of 10, with 5 pending
            assert mock_logs_client.put_log_events.call_count == 1

            # Force flush remaining
            siem.flush()
            assert mock_logs_client.put_log_events.call_count == 2

    def test_siem_failure_handling(self):
        """Test SIEM integration failure handling."""
        with patch("boto3.client") as mock_boto:
            mock_logs_client = MagicMock()
            mock_logs_client.put_log_events.side_effect = Exception("Network error")
            mock_boto.return_value = mock_logs_client

            siem = SIEMIntegration(
                provider="cloudwatch",
                config={
                    "log_group": "/aws/maif/compliance",
                    "retry_count": 3,
                    "fallback_file": "/tmp/audit_fallback.log",
                },
            )

            # Send event that will fail
            event = AuditEvent(
                event_type=AuditEventType.SECURITY,
                action="critical_operation",
                user_id="admin-001",
            )

            # Should not raise exception
            siem.send_event(event)

            # Should have attempted retries (1 initial + 3 retries = 4 attempts)
            assert mock_logs_client.put_log_events.call_count == 4

            # Check fallback file was created
            assert os.path.exists("/tmp/audit_fallback.log")


class TestComplianceFrameworks:
    """Test compliance framework validation."""

    def test_stig_compliance_validation(self):
        """Test STIG compliance validation."""
        logger = EnhancedComplianceLogger(
            compliance_level=ComplianceLevel.STIG,
            frameworks=[ComplianceFramework.DISA_STIG],
        )

        # Test password policy compliance
        password_result = logger.validate_stig_requirement(
            "SRG-OS-000069-GPOS-00037",  # Password complexity
            {
                "min_length": 15,
                "requires_uppercase": True,
                "requires_lowercase": True,
                "requires_numbers": True,
                "requires_special": True,
                "history_count": 24,
            },
        )

        assert password_result["compliant"] is True
        assert password_result["requirement_id"] == "SRG-OS-000069-GPOS-00037"

        # Test audit logging requirement
        audit_result = logger.validate_stig_requirement(
            "SRG-OS-000037-GPOS-00015",  # Audit record generation
            {
                "audit_enabled": True,
                "events_captured": ["login", "logout", "privilege_escalation"],
                "retention_days": 90,
            },
        )

        assert audit_result["compliant"] is True

    def test_fisma_compliance_validation(self):
        """Test FISMA compliance validation."""
        logger = EnhancedComplianceLogger(
            compliance_level=ComplianceLevel.FISMA_MODERATE,
            frameworks=[ComplianceFramework.FISMA],
        )

        # Test access control validation (AC-2)
        ac_result = logger.validate_fisma_control(
            "AC-2",  # Account Management
            {
                "account_types_defined": True,
                "approval_process": True,
                "periodic_review": True,
                "review_frequency_days": 90,
                "termination_process": True,
            },
        )

        assert ac_result["compliant"] is True
        assert ac_result["control_family"] == "Access Control"

        # Test audit and accountability (AU-4)
        au_result = logger.validate_fisma_control(
            "AU-4",  # Audit Storage Capacity
            {
                "storage_capacity_gb": 1000,
                "retention_days": 365,
                "alert_threshold_percent": 80,
                "archival_process": True,
            },
        )

        assert au_result["compliant"] is True

    def test_generate_compliance_report(self):
        """Test compliance report generation."""
        logger = EnhancedComplianceLogger(
            compliance_level=ComplianceLevel.FIPS_140_2,
            frameworks=[
                ComplianceFramework.HIPAA,
                ComplianceFramework.FISMA,
                ComplianceFramework.DISA_STIG,
            ],
        )

        # Log various events
        logger.log_event(AuditEventType.SECURITY, "encryption_update", "admin-001")
        logger.log_data_access(
            "user-001", "phi-record-123", "read", "CONFIDENTIAL", True
        )
        logger.log_authentication("user-002", "smartcard", True)

        # Generate report
        report = logger.generate_compliance_report(
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow(),
            include_details=True,
        )

        assert report["compliance_level"] == "fips_140_2"
        assert len(report["frameworks"]) == 3
        assert report["total_events"] == 3
        assert "event_breakdown" in report
        assert "compliance_status" in report
        assert report["report_generated_at"] is not None


class TestAuditLogEntry:
    """Test AuditLogEntry functionality."""

    def test_audit_log_entry_creation(self):
        """Test creating audit log entries."""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            event_id="evt-123",
            event_type=AuditEventType.SECURITY,
            user_id="admin-001",
            action="key_rotation",
            resource_id="key-456",
            classification="SECRET",
            success=True,
            ip_address="192.168.1.100",
            user_agent="MAIF-Client/1.0",
            session_id="sess-789",
        )

        # Verify all fields
        assert entry.event_id == "evt-123"
        assert entry.classification == "SECRET"
        assert entry.success is True

        # Test JSON serialization
        json_str = entry.to_json()
        data = json.loads(json_str)
        assert data["event_id"] == "evt-123"
        assert data["classification"] == "SECRET"

        # Test CEF format
        cef_str = entry.to_cef()
        assert "CEF:0|" in cef_str
        assert "act=key_rotation" in cef_str
        assert "suser=admin-001" in cef_str

        # Test syslog format
        syslog_str = entry.to_syslog()
        assert "<" in syslog_str  # Priority
        assert "security" in syslog_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
