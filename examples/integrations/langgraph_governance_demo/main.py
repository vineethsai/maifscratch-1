#!/usr/bin/env python3
"""
MAIF + LangGraph: Enterprise AI Governance Demo

This interactive demonstration showcases MAIF's enterprise-grade features
for AI agent workflows:

- Cryptographic Provenance: Every action is signed and hash-chained
- Tamper Detection: Any modification breaks the cryptographic chain
- Data Governance: Role-based access control with full audit trails
- Multi-Agent Coordination: Multiple specialized agents with clear handoffs
- Compliance Reporting: Generate audit reports from artifacts

Requirements:
    pip install maif[integrations]

Usage:
    python main.py

    Or with a specific session:
    python main.py --session my-session-name
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent paths for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from demo_app import GovernanceDemo


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MAIF + LangGraph Enterprise AI Governance Demo"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session name to load or create"
    )
    parser.add_argument(
        "--sessions-dir",
        type=str,
        default="sessions",
        help="Directory for session artifacts"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory for generated reports"
    )
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    sessions_dir = script_dir / args.sessions_dir
    reports_dir = script_dir / args.reports_dir
    
    # Ensure directories exist
    sessions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and run the demo
    demo = GovernanceDemo(
        sessions_dir=str(sessions_dir),
        reports_dir=str(reports_dir),
    )
    
    if args.session:
        demo.run_with_session(args.session)
    else:
        demo.run_interactive()


if __name__ == "__main__":
    main()

