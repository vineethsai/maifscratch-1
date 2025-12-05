"""
MAIF Compliance Module

Contains compliance and logging features:
- Compliance logging for regulated industries
- Forensic analysis
- Audit trails
"""

try:
    from .compliance_logging import (
        ComplianceLogger,
        AuditEvent,
        ComplianceReport,
    )
except ImportError:
    ComplianceLogger = None
    AuditEvent = None
    ComplianceReport = None

try:
    from .compliance_logging_enhanced import EnhancedComplianceLogger
except ImportError:
    EnhancedComplianceLogger = None

try:
    from .forensics import ForensicAnalyzer, ForensicEvidence
except ImportError:
    ForensicAnalyzer = None
    ForensicEvidence = None

ForensicReport = None  # Not implemented

try:
    from .logging_config import configure_logging, get_logger
except ImportError:
    configure_logging = None
    get_logger = None

__all__ = [
    # Compliance logging
    "ComplianceLogger",
    "AuditEvent",
    "ComplianceReport",
    "EnhancedComplianceLogger",
    # Forensics
    "ForensicAnalyzer",
    "ForensicEvidence",
    "ForensicReport",
    # Logging config
    "configure_logging",
    "get_logger",
]

