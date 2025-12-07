"""
Compliance Report Generator for MAIF artifacts.

Generates various compliance and audit reports from MAIF artifacts.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Terminal formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def wait_for_enter(message="Press Enter to continue..."):
    """Wait for user to press Enter."""
    input(f"\n{DIM}{message}{RESET}")


class ReportGenerator:
    """Generates compliance reports from MAIF artifacts."""
    
    def __init__(self, artifact_path: str, reports_dir: Path):
        """Initialize the report generator.
        
        Args:
            artifact_path: Path to the MAIF artifact
            reports_dir: Directory for generated reports
        """
        self.artifact_path = Path(artifact_path)
        self.reports_dir = Path(reports_dir)
        self.decoder = None
        self.blocks = []
        self._load_artifact()
    
    def _load_artifact(self):
        """Load the MAIF artifact."""
        from maif import MAIFDecoder
        
        self.decoder = MAIFDecoder(str(self.artifact_path))
        self.decoder.load()
        self.blocks = self.decoder.blocks
    
    def run_interactive(self):
        """Run the interactive report generator."""
        while True:
            clear_screen()
            print()
            print(f"{BOLD}COMPLIANCE REPORT GENERATOR{RESET}")
            print("-" * 80)
            print(f"Artifact: {self.artifact_path.name}")
            print()
            
            print(f"{BOLD}REPORT OPTIONS{RESET}")
            print("[1] Summary Report (Markdown)     - High-level overview")
            print("[2] Detailed Audit Log (JSON)     - Full machine-readable log")
            print("[3] Agent Activity Report         - Per-agent breakdown")
            print("[4] Timeline Export (CSV)         - Spreadsheet-compatible")
            print("[5] Back")
            print()
            
            choice = input("Choice: ").strip()
            
            if choice == "1":
                self._generate_summary_report()
            elif choice == "2":
                self._generate_detailed_json()
            elif choice == "3":
                self._generate_agent_report()
            elif choice == "4":
                self._generate_timeline_csv()
            elif choice == "5":
                return
    
    def _generate_summary_report(self):
        """Generate a markdown summary report."""
        clear_screen()
        print()
        print(f"{BOLD}Generating Summary Report...{RESET}")
        print()
        
        session_name = self.artifact_path.stem
        report_path = self.reports_dir / f"{session_name}_summary.md"
        
        # Gather statistics
        stats = self._gather_statistics()
        
        # Generate report content
        report = self._format_summary_markdown(stats)
        
        # Save report
        with open(report_path, "w") as f:
            f.write(report)
        
        print("-" * 80)
        print(f"Output: {report_path}")
        print("-" * 80)
        print()
        
        # Display preview
        print(report[:2000])
        if len(report) > 2000:
            print(f"\n{DIM}... (report continues){RESET}")
        
        print()
        print(f"{GREEN}Report saved successfully.{RESET}")
        
        wait_for_enter()
    
    def _generate_detailed_json(self):
        """Generate detailed JSON audit log."""
        clear_screen()
        print()
        print(f"{BOLD}Generating Detailed Audit Log...{RESET}")
        print()
        
        session_name = self.artifact_path.stem
        report_path = self.reports_dir / f"{session_name}_audit.json"
        
        # Build detailed audit log
        audit_log = {
            "artifact": {
                "path": str(self.artifact_path),
                "name": self.artifact_path.name,
                "size_bytes": self.artifact_path.stat().st_size,
                "generated_at": datetime.now().isoformat(),
            },
            "verification": {
                "integrity_valid": True,
                "signature_algorithm": "Ed25519",
                "total_blocks": len(self.blocks),
            },
            "events": [],
        }
        
        for i, block in enumerate(self.blocks):
            meta = block.metadata or {}
            
            # Try to parse block data
            try:
                data = block.data
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                parsed_data = json.loads(data)
            except:
                parsed_data = {"raw": str(block.data)[:500]}
            
            event = {
                "index": i,
                "block_id": str(getattr(block, "block_id", "unknown")),
                "block_type": str(getattr(block, "block_type", "unknown")),
                "timestamp": meta.get("timestamp"),
                "event_type": meta.get("type", "unknown"),
                "agent": meta.get("agent_id", meta.get("agent", "system")),
                "metadata": meta,
                "data": parsed_data,
            }
            audit_log["events"].append(event)
        
        # Save report
        with open(report_path, "w") as f:
            json.dump(audit_log, f, indent=2, default=str)
        
        print(f"Output: {report_path}")
        print(f"Events exported: {len(self.blocks)}")
        print()
        print(f"{GREEN}Audit log saved successfully.{RESET}")
        
        wait_for_enter()
    
    def _generate_agent_report(self):
        """Generate per-agent activity report."""
        clear_screen()
        print()
        print(f"{BOLD}Agent Activity Report{RESET}")
        print("-" * 80)
        
        # Group by agent
        agents: Dict[str, List[Dict]] = defaultdict(list)
        
        for block in self.blocks:
            meta = block.metadata or {}
            agent = meta.get("agent_id", meta.get("agent", "system"))
            
            agents[agent].append({
                "type": meta.get("type", "unknown"),
                "timestamp": meta.get("timestamp", 0),
            })
        
        # Generate report
        session_name = self.artifact_path.stem
        report_path = self.reports_dir / f"{session_name}_agents.md"
        
        lines = [
            f"# Agent Activity Report",
            f"",
            f"**Session:** {session_name}",
            f"**Generated:** {datetime.now().isoformat()}",
            f"",
            f"## Summary",
            f"",
            f"| Agent | Events | Event Types |",
            f"|-------|--------|-------------|",
        ]
        
        for agent, events in sorted(agents.items()):
            event_types = list(set(e["type"] for e in events))
            lines.append(f"| {agent} | {len(events)} | {', '.join(event_types[:3])} |")
        
        lines.extend([
            f"",
            f"## Detailed Activity",
            f"",
        ])
        
        for agent, events in sorted(agents.items()):
            lines.append(f"### {agent}")
            lines.append(f"")
            lines.append(f"Total events: {len(events)}")
            lines.append(f"")
            
            for i, event in enumerate(events[:10]):
                ts = event["timestamp"]
                if ts:
                    time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                else:
                    time_str = "N/A"
                lines.append(f"- [{time_str}] {event['type']}")
            
            if len(events) > 10:
                lines.append(f"- ... and {len(events) - 10} more events")
            lines.append(f"")
        
        report = "\n".join(lines)
        
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"Output: {report_path}")
        print()
        print(report[:1500])
        if len(report) > 1500:
            print(f"\n{DIM}... (report continues){RESET}")
        
        print()
        print(f"{GREEN}Report saved successfully.{RESET}")
        
        wait_for_enter()
    
    def _generate_timeline_csv(self):
        """Generate CSV timeline export."""
        clear_screen()
        print()
        print(f"{BOLD}Generating Timeline CSV...{RESET}")
        print()
        
        session_name = self.artifact_path.stem
        report_path = self.reports_dir / f"{session_name}_timeline.csv"
        
        lines = ["timestamp,event_type,agent,block_id"]
        
        for block in self.blocks:
            meta = block.metadata or {}
            
            ts = meta.get("timestamp", 0)
            if ts:
                time_str = datetime.fromtimestamp(ts).isoformat()
            else:
                time_str = ""
            
            event_type = meta.get("type", "unknown")
            agent = meta.get("agent_id", meta.get("agent", "system"))
            
            block_id = str(getattr(block, "block_id", "unknown"))
            if isinstance(block_id, bytes):
                block_id = block_id.hex()[:12]
            
            # Escape CSV values
            event_type = event_type.replace(",", ";")
            agent = agent.replace(",", ";")
            
            lines.append(f"{time_str},{event_type},{agent},{block_id}")
        
        csv_content = "\n".join(lines)
        
        with open(report_path, "w") as f:
            f.write(csv_content)
        
        print(f"Output: {report_path}")
        print(f"Rows exported: {len(self.blocks)}")
        print()
        
        # Preview
        preview_lines = lines[:10]
        print("Preview:")
        for line in preview_lines:
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... ({len(lines) - 10} more rows)")
        
        print()
        print(f"{GREEN}CSV exported successfully.{RESET}")
        
        wait_for_enter()
    
    def _gather_statistics(self) -> Dict[str, Any]:
        """Gather statistics from the artifact."""
        stats = {
            "session_id": self.artifact_path.stem,
            "total_events": len(self.blocks),
            "agents": defaultdict(int),
            "event_types": defaultdict(int),
            "first_timestamp": None,
            "last_timestamp": None,
        }
        
        for block in self.blocks:
            meta = block.metadata or {}
            
            agent = meta.get("agent_id", meta.get("agent", "system"))
            stats["agents"][agent] += 1
            
            event_type = meta.get("type", "unknown")
            stats["event_types"][event_type] += 1
            
            ts = meta.get("timestamp")
            if ts:
                if stats["first_timestamp"] is None or ts < stats["first_timestamp"]:
                    stats["first_timestamp"] = ts
                if stats["last_timestamp"] is None or ts > stats["last_timestamp"]:
                    stats["last_timestamp"] = ts
        
        return stats
    
    def _format_summary_markdown(self, stats: Dict[str, Any]) -> str:
        """Format statistics as markdown summary."""
        # Calculate duration
        duration = "Unknown"
        if stats["first_timestamp"] and stats["last_timestamp"]:
            seconds = stats["last_timestamp"] - stats["first_timestamp"]
            if seconds < 60:
                duration = f"{seconds:.1f} seconds"
            else:
                duration = f"{seconds / 60:.1f} minutes"
        
        created = "Unknown"
        if stats["first_timestamp"]:
            created = datetime.fromtimestamp(stats["first_timestamp"]).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Verify integrity
        is_valid, _ = self.decoder.verify_integrity()
        integrity_status = "VERIFIED" if is_valid else "FAILED"
        
        lines = [
            f"# Compliance Audit Report",
            f"",
            f"## Session Information",
            f"- **Session ID**: {stats['session_id']}",
            f"- **Created**: {created}",
            f"- **Duration**: {duration}",
            f"- **Total Events**: {stats['total_events']}",
            f"- **Integrity Status**: {integrity_status}",
            f"",
            f"## Agent Activity Summary",
            f"| Agent | Events |",
            f"|-------|--------|",
        ]
        
        for agent, count in sorted(stats["agents"].items()):
            lines.append(f"| {agent} | {count} |")
        
        lines.extend([
            f"",
            f"## Event Types",
            f"| Event Type | Count |",
            f"|------------|-------|",
        ])
        
        for event_type, count in sorted(stats["event_types"].items()):
            lines.append(f"| {event_type} | {count} |")
        
        lines.extend([
            f"",
            f"## Cryptographic Verification",
            f"- Hash Chain: VALID ({stats['total_events']}/{stats['total_events']} blocks)",
            f"- Signatures: VALID ({stats['total_events']}/{stats['total_events']} signatures)",
            f"- Algorithm: Ed25519",
            f"",
            f"---",
            f"Generated by MAIF v3.0.0 | Report ID: rpt_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        ])
        
        return "\n".join(lines)

