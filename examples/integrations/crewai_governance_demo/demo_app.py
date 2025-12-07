"""
Main demo application for the CrewAI Governance Demo.

This module provides the interactive menu system and orchestrates
all the demonstration features.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Terminal formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header."""
    print()
    print("=" * 80)
    print(f"{BOLD}    MAIF + CrewAI: Enterprise AI Governance Demo{RESET}")
    print(f"{DIM}    Demonstrating Cryptographic Provenance for Multi-Agent Workflows{RESET}")
    print("=" * 80)
    print()


def print_separator(char="-", width=80):
    """Print a separator line."""
    print(char * width)


def wait_for_enter(message="Press Enter to continue..."):
    """Wait for user to press Enter."""
    input(f"\n{DIM}{message}{RESET}")


class CrewAIGovernanceDemo:
    """Main demo application class."""
    
    def __init__(self, sessions_dir: str, reports_dir: str):
        """Initialize the demo.
        
        Args:
            sessions_dir: Directory for session artifacts
            reports_dir: Directory for generated reports
        """
        self.sessions_dir = Path(sessions_dir)
        self.reports_dir = Path(reports_dir)
        self.current_session: Optional[str] = None
        
        # Ensure directories exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def run_interactive(self):
        """Run the interactive menu."""
        while True:
            clear_screen()
            print_header()
            
            print(f"{BOLD}MAIN MENU{RESET}")
            print_separator()
            print(f"[1] Run Research Crew              {DIM}- Multi-agent workflow with provenance{RESET}")
            print(f"[2] Resume Previous Session        {DIM}- Load from existing MAIF artifact{RESET}")
            print(f"[3] Inspect Artifact Provenance    {DIM}- Deep dive into audit trail{RESET}")
            print(f"[4] Security Verification          {DIM}- Verify signatures and integrity{RESET}")
            print(f"[5] Tamper Detection Demo          {DIM}- See what happens when data is modified{RESET}")
            print(f"[6] Generate Compliance Report     {DIM}- Export audit report{RESET}")
            print(f"[7] Agent Memory Demo              {DIM}- Persistent memory with provenance{RESET}")
            print(f"[8] Exit")
            print()
            
            choice = input(f"Choice: ").strip()
            
            if choice == "1":
                self.run_research_crew()
            elif choice == "2":
                self.resume_previous_session()
            elif choice == "3":
                self.inspect_provenance()
            elif choice == "4":
                self.security_verification()
            elif choice == "5":
                self.tamper_detection_demo()
            elif choice == "6":
                self.generate_compliance_report()
            elif choice == "7":
                self.agent_memory_demo()
            elif choice == "8":
                print("\nGoodbye!")
                sys.exit(0)
            else:
                print(f"\n{RED}Invalid choice. Please try again.{RESET}")
                wait_for_enter()
    
    def run_with_session(self, session_name: str):
        """Run with a specific session."""
        self.current_session = session_name
        artifact_path = self.sessions_dir / f"{session_name}.maif"
        
        if artifact_path.exists():
            self._run_crew(session_name, resume=True)
        else:
            self._run_crew(session_name, resume=False)
    
    def run_research_crew(self):
        """Run a new research crew with provenance tracking."""
        clear_screen()
        print_header()
        print(f"{BOLD}RUN RESEARCH CREW{RESET}")
        print_separator()
        print()
        
        # Get session name
        session_name = input("Enter session name (or press Enter for auto-generated): ").strip()
        if not session_name:
            session_name = f"crew-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Get research topic
        print()
        topic = input("Enter research topic: ").strip()
        if not topic:
            topic = "Best practices for securing AI agent systems"
        
        self._run_crew(session_name, resume=False, topic=topic)
    
    def resume_previous_session(self):
        """Resume a previous session from MAIF artifact."""
        clear_screen()
        print_header()
        print(f"{BOLD}RESUME PREVIOUS SESSION{RESET}")
        print_separator()
        print()
        
        # List available sessions
        sessions = self._list_sessions()
        
        if not sessions:
            print(f"{YELLOW}No previous sessions found.{RESET}")
            wait_for_enter()
            return
        
        print(f"{'':3} {'Session ID':<30} | {'Created':<20} | {'Blocks':<8} | Status")
        print_separator()
        
        for i, session in enumerate(sessions, 1):
            status_color = GREEN if session['valid'] else RED
            status = "Verified" if session['valid'] else "TAMPERED"
            print(f"{i:2}. {session['name']:<30} | {session['created']:<20} | {session['blocks']:<8} | {status_color}{status}{RESET}")
        
        print()
        choice = input("Select session (or 0 to cancel): ").strip()
        
        try:
            idx = int(choice)
            if idx == 0:
                return
            if 1 <= idx <= len(sessions):
                session_name = sessions[idx - 1]['name']
                self._show_session_details(sessions[idx - 1]['path'])
        except ValueError:
            print(f"\n{RED}Invalid selection.{RESET}")
            wait_for_enter()
    
    def inspect_provenance(self):
        """Inspect artifact provenance."""
        clear_screen()
        print_header()
        print(f"{BOLD}PROVENANCE INSPECTOR{RESET}")
        print_separator()
        print()
        
        # Select artifact
        artifact_path = self._select_artifact()
        if not artifact_path:
            return
        
        from provenance_inspector import ProvenanceInspector
        inspector = ProvenanceInspector(artifact_path)
        inspector.run_interactive()
    
    def security_verification(self):
        """Run security verification on an artifact."""
        clear_screen()
        print_header()
        print(f"{BOLD}SECURITY VERIFICATION{RESET}")
        print_separator()
        print()
        
        # Select artifact
        artifact_path = self._select_artifact()
        if not artifact_path:
            return
        
        from security_verifier import SecurityVerifier
        verifier = SecurityVerifier(artifact_path)
        verifier.run_verification()
        wait_for_enter()
    
    def tamper_detection_demo(self):
        """Demonstrate tamper detection."""
        clear_screen()
        print_header()
        print(f"{BOLD}TAMPER DETECTION DEMONSTRATION{RESET}")
        print_separator()
        print()
        print("This demo shows how MAIF detects unauthorized modifications.")
        print()
        
        from tamper_demo import TamperDemo
        demo = TamperDemo(self.sessions_dir)
        demo.run()
        wait_for_enter()
    
    def generate_compliance_report(self):
        """Generate a compliance report."""
        clear_screen()
        print_header()
        print(f"{BOLD}COMPLIANCE REPORT GENERATOR{RESET}")
        print_separator()
        print()
        
        # Select artifact
        artifact_path = self._select_artifact()
        if not artifact_path:
            return
        
        from report_generator import ReportGenerator
        generator = ReportGenerator(artifact_path, self.reports_dir)
        generator.run_interactive()
    
    def agent_memory_demo(self):
        """Demonstrate persistent agent memory."""
        clear_screen()
        print_header()
        print(f"{BOLD}AGENT MEMORY DEMONSTRATION{RESET}")
        print_separator()
        print()
        print("This demo shows how MAIF provides persistent memory for CrewAI agents.")
        print()
        
        from memory_demo import MemoryDemo
        demo = MemoryDemo(self.sessions_dir)
        demo.run()
    
    def _run_crew(self, session_name: str, resume: bool = False, topic: str = None):
        """Run a CrewAI crew with provenance tracking.
        
        Args:
            session_name: Name of the session
            resume: Whether to resume an existing session
            topic: Research topic (for new sessions)
        """
        clear_screen()
        print_header()
        
        artifact_path = self.sessions_dir / f"{session_name}.maif"
        
        print(f"Session: {CYAN}{session_name}{RESET}")
        print(f"Artifact: {DIM}{artifact_path}{RESET}")
        if topic:
            print(f"Topic: {YELLOW}{topic}{RESET}")
        print_separator()
        print()
        
        from crew_runner import CrewRunner
        
        runner = CrewRunner(
            artifact_path=str(artifact_path),
            session_name=session_name,
        )
        
        if resume:
            print(f"{DIM}Loading previous session...{RESET}")
            runner.show_session_summary()
            print()
        else:
            print(f"{DIM}Starting new crew execution...{RESET}")
            print()
            
            # Run the crew
            result = runner.run_research_crew(topic or "AI agent security")
            
            if result:
                print()
                print(f"{GREEN}Crew execution completed successfully.{RESET}")
            else:
                print()
                print(f"{YELLOW}Crew execution completed with warnings.{RESET}")
        
        # Show summary
        print()
        summary = runner.finalize()
        
        print()
        print(f"{GREEN}Session saved successfully.{RESET}")
        print(f"  Tasks completed: {summary.get('tasks_completed', 0)}")
        print(f"  Steps executed: {summary.get('steps_executed', 0)}")
        print(f"  Total events: {summary.get('total_events', 0)}")
        print(f"  Artifact: {artifact_path}")
        print(f"  Integrity: {GREEN}VERIFIED{RESET}")
        
        wait_for_enter()
    
    def _show_session_details(self, artifact_path: str):
        """Show details of a session artifact."""
        clear_screen()
        print_header()
        print(f"{BOLD}SESSION DETAILS{RESET}")
        print_separator()
        print()
        
        try:
            from maif import MAIFDecoder
            import json
            
            decoder = MAIFDecoder(artifact_path)
            decoder.load()
            
            is_valid, errors = decoder.verify_integrity()
            
            print(f"Artifact: {CYAN}{artifact_path}{RESET}")
            print(f"Integrity: {GREEN if is_valid else RED}{'VERIFIED' if is_valid else 'FAILED'}{RESET}")
            print(f"Total blocks: {len(decoder.blocks)}")
            print()
            
            # Show event summary
            event_counts = {}
            for block in decoder.blocks:
                event_type = block.metadata.get("type", "unknown") if block.metadata else "unknown"
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            print(f"{BOLD}Event Summary:{RESET}")
            for event_type, count in sorted(event_counts.items()):
                print(f"  {event_type}: {count}")
            
            # Show recent events
            print()
            print(f"{BOLD}Recent Events (last 5):{RESET}")
            for block in decoder.blocks[-5:]:
                if block.metadata:
                    event_type = block.metadata.get("type", "unknown")
                    timestamp = block.metadata.get("timestamp", 0)
                    ts_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S") if timestamp else "?"
                    print(f"  [{ts_str}] {event_type}")
            
        except Exception as e:
            print(f"{RED}Error loading artifact: {e}{RESET}")
        
        wait_for_enter()
    
    def _list_sessions(self) -> List[Dict[str, Any]]:
        """List available sessions.
        
        Returns:
            List of session metadata dicts
        """
        sessions = []
        
        for artifact_file in self.sessions_dir.glob("*.maif"):
            try:
                from maif import MAIFDecoder
                
                decoder = MAIFDecoder(str(artifact_file))
                decoder.load()
                
                is_valid, _ = decoder.verify_integrity()
                
                # Get creation time from first block
                created = "Unknown"
                if decoder.blocks:
                    first_block = decoder.blocks[0]
                    if first_block.metadata and 'timestamp' in first_block.metadata:
                        ts = first_block.metadata['timestamp']
                        created = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                
                sessions.append({
                    'name': artifact_file.stem,
                    'path': str(artifact_file),
                    'blocks': len(decoder.blocks),
                    'created': created,
                    'valid': is_valid,
                })
            except Exception as e:
                sessions.append({
                    'name': artifact_file.stem,
                    'path': str(artifact_file),
                    'blocks': '?',
                    'created': 'Error',
                    'valid': False,
                })
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x['created'], reverse=True)
        return sessions
    
    def _select_artifact(self) -> Optional[str]:
        """Prompt user to select an artifact.
        
        Returns:
            Path to selected artifact, or None if cancelled
        """
        sessions = self._list_sessions()
        
        if not sessions:
            print(f"{YELLOW}No artifacts found in {self.sessions_dir}{RESET}")
            wait_for_enter()
            return None
        
        print("Available artifacts:")
        print()
        
        for i, session in enumerate(sessions, 1):
            status_color = GREEN if session['valid'] else RED
            print(f"  [{i}] {session['name']}.maif ({session['blocks']} blocks) - {status_color}{'Valid' if session['valid'] else 'Invalid'}{RESET}")
        
        print()
        choice = input("Select artifact (or 0 to cancel): ").strip()
        
        try:
            idx = int(choice)
            if idx == 0:
                return None
            if 1 <= idx <= len(sessions):
                return sessions[idx - 1]['path']
        except ValueError:
            pass
        
        print(f"\n{RED}Invalid selection.{RESET}")
        wait_for_enter()
        return None

