"""
Advanced digital forensics and incident investigation for MAIF.
Implements comprehensive forensic analysis as specified in the paper.
"""

import json
import time
import hashlib
import statistics
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path

from .core import MAIFDecoder, MAIFVersion, MAIFBlock


class SeverityLevel(Enum):
    """Severity levels for forensic findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies detected."""

    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    INTEGRITY = "integrity"
    ACCESS = "access"
    STRUCTURAL = "structural"


@dataclass
class ForensicEvidence:
    """Represents a piece of forensic evidence."""

    evidence_id: str
    timestamp: float
    severity: SeverityLevel
    anomaly_type: AnomalyType
    description: str
    affected_blocks: List[str]
    metadata: Dict[str, Any]
    confidence_score: float


@dataclass
class AgentBehaviorProfile:
    """Profile of agent behavior patterns."""

    agent_id: str
    total_operations: int
    operation_frequency: float  # ops per hour
    common_operations: List[str]
    unusual_patterns: List[str]
    risk_score: float
    last_activity: float


@dataclass
class TimelineEvent:
    """Event in the forensic timeline."""

    timestamp: float
    event_type: str
    agent_id: str
    block_id: str
    operation: str
    details: Dict[str, Any]
    anomaly_indicators: List[str]


class ForensicAnalyzer:
    """Advanced forensic analysis engine for MAIF files."""

    def __init__(self):
        self.evidence_database: List[ForensicEvidence] = []
        self.agent_profiles: Dict[str, AgentBehaviorProfile] = {}
        self.timeline_events: List[TimelineEvent] = []
        self.analysis_metadata = {
            "analysis_start": time.time(),
            "total_files_analyzed": 0,
            "total_evidence_collected": 0,
        }

    def analyze_maif_file(
        self, maif_path: str, manifest_path: str = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive forensic analysis of a MAIF file.
        Note: manifest_path is deprecated in v3 format (self-contained files).
        """
        try:
            decoder = MAIFDecoder(maif_path)
            decoder.load()

            # Perform multiple analysis passes
            version_analysis = self._analyze_version_history(decoder)
            integrity_analysis = self._analyze_integrity_violations(decoder)
            temporal_analysis = self._analyze_temporal_anomalies(decoder)
            behavioral_analysis = self._analyze_agent_behavior(decoder)
            structural_analysis = self._analyze_structural_integrity(decoder)

            # Build comprehensive timeline
            timeline = self._reconstruct_timeline(decoder)

            # Generate recommendations
            recommendations = self._generate_recommendations()

            # Calculate overall risk assessment
            risk_assessment = self._calculate_risk_assessment()

            analysis_result = {
                "file_path": maif_path,
                "analysis_timestamp": time.time(),
                "version_analysis": version_analysis,
                "integrity_analysis": integrity_analysis,
                "temporal_analysis": temporal_analysis,
                "behavioral_analysis": behavioral_analysis,
                "structural_analysis": structural_analysis,
                "timeline": timeline,
                "evidence_summary": self._summarize_evidence(),
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "metadata": self.analysis_metadata,
            }

            self.analysis_metadata["total_files_analyzed"] += 1
            return analysis_result

        except Exception as e:
            return {
                "error": f"Forensic analysis failed: {str(e)}",
                "file_path": maif_path,
                "analysis_timestamp": time.time(),
            }

    def _analyze_version_history(self, decoder: MAIFDecoder) -> Dict[str, Any]:
        """Analyze version history for suspicious patterns."""
        if not hasattr(decoder, "version_history") or not decoder.version_history:
            return {"status": "no_version_history", "findings": []}

        findings = []
        # Flatten version history from dict to list
        all_versions = []
        for versions in decoder.version_history.values():
            all_versions.extend(versions)

        if not all_versions:
            return {"status": "no_version_history", "findings": []}

        version_history = all_versions

        # Check for rapid successive changes
        rapid_changes = self._detect_rapid_changes(version_history)
        if rapid_changes:
            evidence = ForensicEvidence(
                evidence_id=f"rapid_changes_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.MEDIUM,
                anomaly_type=AnomalyType.TEMPORAL,
                description=f"Detected {len(rapid_changes)} rapid successive changes",
                affected_blocks=[change["block_id"] for change in rapid_changes],
                metadata={"rapid_changes": rapid_changes},
                confidence_score=0.8,
            )
            self.evidence_database.append(evidence)
            findings.append("rapid_successive_changes")

        # Check for version gaps
        version_gaps = self._detect_version_gaps(version_history)
        if version_gaps:
            evidence = ForensicEvidence(
                evidence_id=f"version_gaps_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.HIGH,
                anomaly_type=AnomalyType.INTEGRITY,
                description=f"Detected {len(version_gaps)} version gaps indicating potential tampering",
                affected_blocks=[gap["block_id"] for gap in version_gaps],
                metadata={"version_gaps": version_gaps},
                confidence_score=0.9,
            )
            self.evidence_database.append(evidence)
            findings.append("version_gaps")

        # Check for unusual operation patterns
        operation_patterns = self._analyze_operation_patterns(version_history)
        if operation_patterns.get("suspicious_patterns"):
            evidence = ForensicEvidence(
                evidence_id=f"operation_patterns_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.MEDIUM,
                anomaly_type=AnomalyType.BEHAVIORAL,
                description="Detected unusual operation patterns",
                affected_blocks=operation_patterns.get("affected_blocks", []),
                metadata=operation_patterns,
                confidence_score=0.7,
            )
            self.evidence_database.append(evidence)
            findings.append("unusual_operation_patterns")

        return {
            "status": "analyzed",
            "total_versions": len(version_history),
            "findings": findings,
            "operation_patterns": operation_patterns,
        }

    def _analyze_integrity_violations(self, decoder: MAIFDecoder) -> Dict[str, Any]:
        """Analyze integrity violations and tampering evidence."""
        findings = []

        # Check hash consistency
        hash_violations = []
        try:
            if not decoder.verify_integrity():
                # Detailed hash checking
                for block in decoder.blocks:
                    try:
                        # Get content hash from block
                        expected_hash = (
                            block.header.content_hash.hex()
                            if isinstance(block.header.content_hash, bytes)
                            else block.header.content_hash
                        )
                        computed_hash = hashlib.sha256(block.data).hexdigest()

                        if computed_hash != expected_hash:
                            hash_violations.append(
                                {
                                    "block_id": block.header.block_id,
                                    "expected_hash": expected_hash,
                                    "computed_hash": computed_hash,
                                    "block_type": str(block.header.block_type),
                                }
                            )
                    except Exception:
                        continue

                if hash_violations:
                    evidence = ForensicEvidence(
                        evidence_id=f"hash_violations_{int(time.time())}",
                        timestamp=time.time(),
                        severity=SeverityLevel.CRITICAL,
                        anomaly_type=AnomalyType.INTEGRITY,
                        description=f"Detected {len(hash_violations)} hash integrity violations",
                        affected_blocks=[v["block_id"] for v in hash_violations],
                        metadata={"violations": hash_violations},
                        confidence_score=1.0,
                    )
                    self.evidence_database.append(evidence)
                    findings.append("hash_integrity_violations")

        except Exception as e:
            findings.append(f"integrity_check_error: {str(e)}")

        # Check for duplicate blocks (potential duplication attack)
        duplicate_blocks = self._detect_duplicate_blocks(decoder.blocks)
        if duplicate_blocks:
            evidence = ForensicEvidence(
                evidence_id=f"duplicate_blocks_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.MEDIUM,
                anomaly_type=AnomalyType.STRUCTURAL,
                description=f"Detected {len(duplicate_blocks)} duplicate blocks",
                affected_blocks=list(duplicate_blocks.keys()),
                metadata={"duplicates": duplicate_blocks},
                confidence_score=0.8,
            )
            self.evidence_database.append(evidence)
            findings.append("duplicate_blocks")

        # Check for missing expected block types
        missing_blocks = self._detect_missing_block_types(decoder.blocks)
        if missing_blocks:
            evidence = ForensicEvidence(
                evidence_id=f"missing_blocks_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.MEDIUM,
                anomaly_type=AnomalyType.STRUCTURAL,
                description=f"Missing expected block types: {', '.join(missing_blocks)}",
                affected_blocks=[],
                metadata={"missing_types": missing_blocks},
                confidence_score=0.6,
            )
            self.evidence_database.append(evidence)
            findings.append("missing_expected_blocks")

        return {
            "status": "analyzed",
            "findings": findings,
            "hash_violations": len(hash_violations),
            "duplicate_blocks": len(duplicate_blocks),
            "missing_block_types": missing_blocks,
        }

    def _analyze_temporal_anomalies(self, decoder: MAIFDecoder) -> Dict[str, Any]:
        """Analyze temporal anomalies in timestamps."""
        findings = []

        if not hasattr(decoder, "version_history") or not decoder.version_history:
            return {"status": "no_temporal_data", "findings": []}

        # Flatten version history from dict to list
        all_versions = []
        for versions in decoder.version_history.values():
            all_versions.extend(versions)

        if not all_versions:
            return {"status": "no_temporal_data", "findings": []}

        timestamps = [v.timestamp for v in all_versions]

        # Check for future timestamps
        current_time = time.time()
        future_timestamps = [ts for ts in timestamps if ts > current_time]
        if future_timestamps:
            evidence = ForensicEvidence(
                evidence_id=f"future_timestamps_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.HIGH,
                anomaly_type=AnomalyType.TEMPORAL,
                description=f"Detected {len(future_timestamps)} future timestamps",
                affected_blocks=[],
                metadata={"future_timestamps": future_timestamps},
                confidence_score=0.95,
            )
            self.evidence_database.append(evidence)
            findings.append("future_timestamps")

        # Check for timestamp reversals
        timestamp_reversals = []
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                timestamp_reversals.append(
                    {
                        "position": i,
                        "previous": timestamps[i - 1],
                        "current": timestamps[i],
                        "reversal_amount": timestamps[i - 1] - timestamps[i],
                    }
                )

        if timestamp_reversals:
            evidence = ForensicEvidence(
                evidence_id=f"timestamp_reversals_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.HIGH,
                anomaly_type=AnomalyType.TEMPORAL,
                description=f"Detected {len(timestamp_reversals)} timestamp reversals",
                affected_blocks=[],
                metadata={"reversals": timestamp_reversals},
                confidence_score=0.9,
            )
            self.evidence_database.append(evidence)
            findings.append("timestamp_reversals")

        # Check for unusually fast operations
        fast_operations = []
        for i in range(1, len(all_versions)):
            time_diff = all_versions[i].timestamp - all_versions[i - 1].timestamp
            if time_diff < 0.1:  # Less than 100ms between operations
                fast_operations.append(
                    {
                        "operation_index": i,
                        "time_difference": time_diff,
                        "agent_id": all_versions[i].agent_id,
                        "operation": all_versions[i].operation,
                    }
                )

        if len(fast_operations) > 5:  # More than 5 very fast operations
            evidence = ForensicEvidence(
                evidence_id=f"fast_operations_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.MEDIUM,
                anomaly_type=AnomalyType.TEMPORAL,
                description=f"Detected {len(fast_operations)} unusually fast operations",
                affected_blocks=[],
                metadata={"fast_operations": fast_operations[:10]},  # Limit to first 10
                confidence_score=0.7,
            )
            self.evidence_database.append(evidence)
            findings.append("unusually_fast_operations")

        return {
            "status": "analyzed",
            "findings": findings,
            "future_timestamps": len(future_timestamps),
            "timestamp_reversals": len(timestamp_reversals),
            "fast_operations": len(fast_operations),
        }

    def _analyze_agent_behavior(self, decoder: MAIFDecoder) -> Dict[str, Any]:
        """Analyze agent behavior patterns for anomalies."""
        findings = []

        if not hasattr(decoder, "version_history") or not decoder.version_history:
            return {"status": "no_agent_data", "findings": []}

        # Build agent profiles
        # Flatten version history from dict to list
        all_versions = []
        for versions in decoder.version_history.values():
            all_versions.extend(versions)

        if not all_versions:
            return {"status": "no_agent_data", "findings": []}

        agent_activities = {}
        for version in all_versions:
            agent_id = version.agent_id
            if agent_id not in agent_activities:
                agent_activities[agent_id] = []
            agent_activities[agent_id].append(version)

        # Analyze each agent
        for agent_id, activities in agent_activities.items():
            profile = self._build_agent_profile(agent_id, activities)
            self.agent_profiles[agent_id] = profile

            # Check for excessive activity
            if profile.total_operations > 100:
                evidence = ForensicEvidence(
                    evidence_id=f"excessive_activity_{agent_id}_{int(time.time())}",
                    timestamp=time.time(),
                    severity=SeverityLevel.MEDIUM,
                    anomaly_type=AnomalyType.BEHAVIORAL,
                    description=f"Agent {agent_id} performed {profile.total_operations} operations",
                    affected_blocks=[],
                    metadata={"agent_profile": asdict(profile)},
                    confidence_score=0.8,
                )
                self.evidence_database.append(evidence)
                findings.append(f"excessive_activity_{agent_id}")

            # Check for unusual patterns
            if profile.unusual_patterns:
                evidence = ForensicEvidence(
                    evidence_id=f"unusual_patterns_{agent_id}_{int(time.time())}",
                    timestamp=time.time(),
                    severity=SeverityLevel.MEDIUM,
                    anomaly_type=AnomalyType.BEHAVIORAL,
                    description=f"Agent {agent_id} shows unusual patterns: {', '.join(profile.unusual_patterns)}",
                    affected_blocks=[],
                    metadata={"patterns": profile.unusual_patterns},
                    confidence_score=0.7,
                )
                self.evidence_database.append(evidence)
                findings.append(f"unusual_patterns_{agent_id}")

        return {
            "status": "analyzed",
            "findings": findings,
            "total_agents": len(agent_activities),
            "agent_profiles": {
                aid: asdict(profile) for aid, profile in self.agent_profiles.items()
            },
        }

    def _analyze_structural_integrity(self, decoder: MAIFDecoder) -> Dict[str, Any]:
        """Analyze structural integrity of the MAIF file."""
        findings = []

        # Check block size consistency
        size_anomalies = []
        for block in decoder.blocks:
            block_size = (
                block.header.size if hasattr(block.header, "size") else len(block.data)
            )
            block_id = block.header.block_id
            if block_size < 32:  # Minimum size for header
                size_anomalies.append(
                    {"block_id": block_id, "size": block_size, "issue": "too_small"}
                )
            elif block_size > 100 * 1024 * 1024:  # Larger than 100MB
                size_anomalies.append(
                    {
                        "block_id": block_id,
                        "size": block_size,
                        "issue": "unusually_large",
                    }
                )

        if size_anomalies:
            evidence = ForensicEvidence(
                evidence_id=f"size_anomalies_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.MEDIUM,
                anomaly_type=AnomalyType.STRUCTURAL,
                description=f"Detected {len(size_anomalies)} block size anomalies",
                affected_blocks=[a["block_id"] for a in size_anomalies],
                metadata={"anomalies": size_anomalies},
                confidence_score=0.8,
            )
            self.evidence_database.append(evidence)
            findings.append("block_size_anomalies")

        # Check for invalid hash formats
        invalid_hashes = []
        for block in decoder.blocks:
            content_hash = block.header.content_hash
            hash_value = (
                content_hash.hex()
                if isinstance(content_hash, bytes)
                else str(content_hash)
            )
            if not hash_value:
                invalid_hashes.append(
                    {
                        "block_id": block.header.block_id,
                        "hash_value": hash_value,
                        "issue": "empty_hash",
                    }
                )

        if invalid_hashes:
            evidence = ForensicEvidence(
                evidence_id=f"invalid_hashes_{int(time.time())}",
                timestamp=time.time(),
                severity=SeverityLevel.HIGH,
                anomaly_type=AnomalyType.INTEGRITY,
                description=f"Detected {len(invalid_hashes)} invalid hash formats",
                affected_blocks=[h["block_id"] for h in invalid_hashes],
                metadata={"invalid_hashes": invalid_hashes},
                confidence_score=0.9,
            )
            self.evidence_database.append(evidence)
            findings.append("invalid_hash_formats")

        return {
            "status": "analyzed",
            "findings": findings,
            "size_anomalies": len(size_anomalies),
            "invalid_hashes": len(invalid_hashes),
        }

    def _reconstruct_timeline(self, decoder: MAIFDecoder) -> List[Dict[str, Any]]:
        """Reconstruct complete forensic timeline."""
        timeline = []

        if not hasattr(decoder, "version_history") or not decoder.version_history:
            return timeline

        # Flatten version history from dict to list
        all_versions = []
        for versions in decoder.version_history.values():
            all_versions.extend(versions)

        if not all_versions:
            return timeline

        # Sort by timestamp
        sorted_versions = sorted(all_versions, key=lambda v: v.timestamp)

        for version in sorted_versions:
            # Detect anomaly indicators for this event
            anomaly_indicators = []

            # Check if this is part of rapid changes
            rapid_window = 1.0  # 1 second window
            rapid_count = sum(
                1
                for v in sorted_versions
                if abs(v.timestamp - version.timestamp) < rapid_window
            )
            if rapid_count > 3:
                anomaly_indicators.append("rapid_changes")

            # Check if operation is unusual for this agent
            if version.agent_id in self.agent_profiles:
                profile = self.agent_profiles[version.agent_id]
                if version.operation not in profile.common_operations:
                    anomaly_indicators.append("unusual_operation")

            event = TimelineEvent(
                timestamp=version.timestamp,
                event_type="version_change",
                agent_id=version.agent_id,
                block_id=version.block_id,
                operation=version.operation,
                details={
                    "version_number": version.version_number,
                    "previous_hash": version.previous_hash,
                    "current_hash": version.current_hash,
                    "change_description": version.change_description,
                },
                anomaly_indicators=anomaly_indicators,
            )

            self.timeline_events.append(event)
            timeline.append(asdict(event))

        return timeline

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on findings."""
        recommendations = []

        # Analyze evidence by severity
        critical_evidence = [
            e for e in self.evidence_database if e.severity == SeverityLevel.CRITICAL
        ]
        high_evidence = [
            e for e in self.evidence_database if e.severity == SeverityLevel.HIGH
        ]

        if critical_evidence:
            recommendations.append(
                {
                    "priority": "critical",
                    "action": "immediate_investigation",
                    "description": f"Found {len(critical_evidence)} critical security issues requiring immediate attention",
                    "affected_evidence": [e.evidence_id for e in critical_evidence],
                }
            )

        if high_evidence:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "security_review",
                    "description": f"Found {len(high_evidence)} high-priority security concerns",
                    "affected_evidence": [e.evidence_id for e in high_evidence],
                }
            )

        # Check for patterns requiring specific actions
        integrity_issues = [
            e for e in self.evidence_database if e.anomaly_type == AnomalyType.INTEGRITY
        ]
        if integrity_issues:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "integrity_restoration",
                    "description": "Multiple integrity violations detected - consider file restoration from backup",
                    "affected_evidence": [e.evidence_id for e in integrity_issues],
                }
            )

        temporal_issues = [
            e for e in self.evidence_database if e.anomaly_type == AnomalyType.TEMPORAL
        ]
        if temporal_issues:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "timestamp_audit",
                    "description": "Temporal anomalies detected - audit system clocks and time synchronization",
                    "affected_evidence": [e.evidence_id for e in temporal_issues],
                }
            )

        behavioral_issues = [
            e
            for e in self.evidence_database
            if e.anomaly_type == AnomalyType.BEHAVIORAL
        ]
        if behavioral_issues:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "agent_review",
                    "description": "Unusual agent behavior detected - review agent permissions and activities",
                    "affected_evidence": [e.evidence_id for e in behavioral_issues],
                }
            )

        return recommendations

    def _calculate_risk_assessment(self) -> Dict[str, Any]:
        """Calculate overall risk assessment."""
        if not self.evidence_database:
            return {
                "overall_risk": "low",
                "risk_score": 0.0,
                "confidence": 1.0,
                "factors": [],
            }

        # Weight evidence by severity
        severity_weights = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.7,
            SeverityLevel.MEDIUM: 0.4,
            SeverityLevel.LOW: 0.1,
        }

        total_risk = 0.0
        total_confidence = 0.0

        for evidence in self.evidence_database:
            weight = severity_weights[evidence.severity]
            total_risk += weight * evidence.confidence_score
            total_confidence += evidence.confidence_score

        avg_confidence = total_confidence / len(self.evidence_database)
        normalized_risk = min(1.0, total_risk / len(self.evidence_database))

        # Determine risk level
        if normalized_risk >= 0.8:
            risk_level = "critical"
        elif normalized_risk >= 0.6:
            risk_level = "high"
        elif normalized_risk >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Identify key risk factors
        risk_factors = []
        for anomaly_type in AnomalyType:
            type_evidence = [
                e for e in self.evidence_database if e.anomaly_type == anomaly_type
            ]
            if type_evidence:
                avg_severity = statistics.mean(
                    [severity_weights[e.severity] for e in type_evidence]
                )
                risk_factors.append(
                    {
                        "factor": anomaly_type.value,
                        "evidence_count": len(type_evidence),
                        "average_severity": avg_severity,
                    }
                )

        return {
            "overall_risk": risk_level,
            "risk_score": normalized_risk,
            "confidence": avg_confidence,
            "factors": risk_factors,
            "total_evidence": len(self.evidence_database),
        }

    def _summarize_evidence(self) -> Dict[str, Any]:
        """Summarize collected evidence."""
        summary = {
            "total_evidence": len(self.evidence_database),
            "by_severity": {},
            "by_type": {},
            "high_confidence": 0,
        }

        for severity in SeverityLevel:
            count = len([e for e in self.evidence_database if e.severity == severity])
            summary["by_severity"][severity.value] = count

        for anomaly_type in AnomalyType:
            count = len(
                [e for e in self.evidence_database if e.anomaly_type == anomaly_type]
            )
            summary["by_type"][anomaly_type.value] = count

        summary["high_confidence"] = len(
            [e for e in self.evidence_database if e.confidence_score >= 0.8]
        )

        return summary

    # Helper methods for specific analysis tasks

    def _detect_rapid_changes(
        self, version_history: List[MAIFVersion]
    ) -> List[Dict[str, Any]]:
        """Detect rapid successive changes."""
        rapid_changes = []
        window_size = 5.0  # 5 second window

        for i, version in enumerate(version_history):
            # Count operations in time window
            window_start = version.timestamp - window_size
            window_operations = [
                v
                for v in version_history
                if window_start <= v.timestamp <= version.timestamp
            ]

            if len(window_operations) > 5:  # More than 5 operations in 5 seconds
                rapid_changes.append(
                    {
                        "timestamp": version.timestamp,
                        "block_id": version.block_id,
                        "agent_id": version.agent_id,
                        "operations_in_window": len(window_operations),
                    }
                )

        return rapid_changes

    def _detect_version_gaps(
        self, version_history: List[MAIFVersion]
    ) -> List[Dict[str, Any]]:
        """Detect gaps in version chains."""
        gaps = []

        # Group by block_id
        block_versions = {}
        for version in version_history:
            if version.block_id not in block_versions:
                block_versions[version.block_id] = []
            block_versions[version.block_id].append(version)

        # Check each block's version chain
        for block_id, versions in block_versions.items():
            sorted_versions = sorted(versions, key=lambda v: v.version_number)

            for i in range(1, len(sorted_versions)):
                current = sorted_versions[i]
                previous = sorted_versions[i - 1]

                # Check if current.previous_hash matches previous.current_hash
                if current.previous_hash != previous.current_hash:
                    gaps.append(
                        {
                            "block_id": block_id,
                            "version_gap": f"{previous.version_number} -> {current.version_number}",
                            "expected_hash": previous.current_hash,
                            "actual_previous_hash": current.previous_hash,
                        }
                    )

        return gaps

    def _analyze_operation_patterns(
        self, version_history: List[MAIFVersion]
    ) -> Dict[str, Any]:
        """Analyze operation patterns for anomalies."""
        operation_counts = {}
        agent_operations = {}

        for version in version_history:
            # Count operations
            op = version.operation
            operation_counts[op] = operation_counts.get(op, 0) + 1

            # Track per agent
            agent_id = version.agent_id
            if agent_id not in agent_operations:
                agent_operations[agent_id] = {}
            agent_operations[agent_id][op] = agent_operations[agent_id].get(op, 0) + 1

        # Detect suspicious patterns
        suspicious_patterns = []

        # Check for excessive deletions
        delete_count = operation_counts.get("delete", 0)
        total_operations = len(version_history)
        if delete_count > total_operations * 0.3:  # More than 30% deletions
            suspicious_patterns.append("excessive_deletions")

        # Check for agents with unusual operation distributions
        for agent_id, ops in agent_operations.items():
            total_agent_ops = sum(ops.values())
            if total_agent_ops > 1:
                delete_ratio = ops.get("delete", 0) / total_agent_ops
                if delete_ratio > 0.5:  # More than 50% deletions for this agent
                    suspicious_patterns.append(f"agent_{agent_id}_excessive_deletions")

        return {
            "operation_counts": operation_counts,
            "agent_operations": agent_operations,
            "suspicious_patterns": suspicious_patterns,
            "affected_blocks": [],  # Would need more detailed analysis
        }

    def _detect_duplicate_blocks(self, blocks: List[MAIFBlock]) -> Dict[str, List[str]]:
        """Detect duplicate blocks."""
        hash_to_blocks = {}

        for block in blocks:
            content_hash = block.header.content_hash
            hash_val = (
                content_hash.hex()
                if isinstance(content_hash, bytes)
                else str(content_hash)
            )
            if hash_val not in hash_to_blocks:
                hash_to_blocks[hash_val] = []
            hash_to_blocks[hash_val].append(
                block.header.block_id or f"block_{id(block)}"
            )

        # Return only hashes with multiple blocks
        return {
            hash_val: block_ids
            for hash_val, block_ids in hash_to_blocks.items()
            if len(block_ids) > 1
        }

    def _detect_missing_block_types(self, blocks: List[MAIFBlock]) -> List[str]:
        """Detect missing expected block types."""
        present_types = set(str(block.header.block_type) for block in blocks)
        expected_types = {"text_data", "embeddings", "security"}
        missing_types = []

        # Check for basic expected types in a complete MAIF
        if len(blocks) > 5:  # Only check for larger files
            for expected_type in expected_types:
                if expected_type not in present_types:
                    missing_types.append(expected_type)

        return missing_types

    def _build_agent_profile(
        self, agent_id: str, activities: List[MAIFVersion]
    ) -> AgentBehaviorProfile:
        """Build behavioral profile for an agent."""
        if not activities:
            return AgentBehaviorProfile(
                agent_id=agent_id,
                total_operations=0,
                operation_frequency=0.0,
                common_operations=[],
                unusual_patterns=[],
                risk_score=0.0,
                last_activity=0.0,
            )

        # Calculate basic metrics
        total_operations = len(activities)
        timestamps = [a.timestamp for a in activities]
        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1.0
        operation_frequency = (
            total_operations / (time_span / 3600.0) if time_span > 0 else 0.0
        )

        # Analyze operation types
        operation_counts = {}
        for activity in activities:
            op = activity.operation
            operation_counts[op] = operation_counts.get(op, 0) + 1

        # Identify common operations (>20% of total)
        common_operations = [
            op
            for op, count in operation_counts.items()
            if count / total_operations > 0.2
        ]

        # Detect unusual patterns
        unusual_patterns = []

        # High deletion ratio
        delete_ratio = operation_counts.get("delete", 0) / total_operations
        if delete_ratio > 0.3:
            unusual_patterns.append("high_deletion_ratio")

        # Very high activity frequency
        if operation_frequency > 100:  # More than 100 ops per hour
            unusual_patterns.append("excessive_activity_rate")

        # Operations outside normal hours (simplified)
        night_operations = sum(
            1
            for ts in timestamps
            if 22 <= time.localtime(ts).tm_hour or time.localtime(ts).tm_hour <= 6
        )
        if night_operations / total_operations > 0.5:
            unusual_patterns.append("unusual_timing")

        # Calculate risk score
        risk_score = 0.0
        risk_score += min(0.3, delete_ratio)  # Deletion risk
        risk_score += min(0.3, operation_frequency / 1000.0)  # Frequency risk
        risk_score += len(unusual_patterns) * 0.2  # Pattern risk

        return AgentBehaviorProfile(
            agent_id=agent_id,
            total_operations=total_operations,
            operation_frequency=operation_frequency,
            common_operations=common_operations,
            unusual_patterns=unusual_patterns,
            risk_score=min(1.0, risk_score),
            last_activity=max(timestamps),
        )


# Export main classes
__all__ = [
    "ForensicAnalyzer",
    "ForensicEvidence",
    "AgentBehaviorProfile",
    "TimelineEvent",
    "SeverityLevel",
    "AnomalyType",
]
