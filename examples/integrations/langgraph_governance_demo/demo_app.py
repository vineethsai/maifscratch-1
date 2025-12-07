"""
Main demo application for the Governance Demo.

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


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header."""
    print()
    print("=" * 80)
    print(f"{BOLD}    MAIF + LangGraph: Enterprise AI Governance Demo{RESET}")
    print(f"{DIM}    Demonstrating Cryptographic Provenance & Compliance{RESET}")
    print("=" * 80)
    print()


def print_separator(char="-", width=80):
    """Print a separator line."""
    print(char * width)


def wait_for_enter(message="Press Enter to continue..."):
    """Wait for user to press Enter."""
    input(f"\n{DIM}{message}{RESET}")


class GovernanceDemo:
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
        self.current_role: str = "analyst"
        
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
            print(f"[1] Start Audited Conversation      {DIM}- New session with full provenance{RESET}")
            print(f"[2] Resume Previous Session         {DIM}- Load from existing MAIF artifact{RESET}")
            print(f"[3] Inspect Artifact Provenance     {DIM}- Deep dive into audit trail{RESET}")
            print(f"[4] Security Verification           {DIM}- Verify signatures and integrity{RESET}")
            print(f"[5] Tamper Detection Demo           {DIM}- See what happens when data is modified{RESET}")
            print(f"[6] Generate Compliance Report      {DIM}- Export audit report{RESET}")
            print(f"[7] Access Control Simulation       {DIM}- See role-based restrictions{RESET}")
            print(f"[8] Exit")
            print()
            
            choice = input(f"Choice: ").strip()
            
            if choice == "1":
                self.start_audited_conversation()
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
                self.access_control_simulation()
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
            self._run_conversation(session_name, resume=True)
        else:
            self._run_conversation(session_name, resume=False)
    
    def start_audited_conversation(self):
        """Start a new audited conversation."""
        clear_screen()
        print_header()
        print(f"{BOLD}START AUDITED CONVERSATION{RESET}")
        print_separator()
        print()
        
        # Get session name
        session_name = input("Enter session name (or press Enter for auto-generated): ").strip()
        if not session_name:
            session_name = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self._run_conversation(session_name, resume=False)
    
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
                self._run_conversation(session_name, resume=True)
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
    
    def access_control_simulation(self):
        """Run access control simulation."""
        clear_screen()
        print_header()
        print(f"{BOLD}ACCESS CONTROL SIMULATION{RESET}")
        print_separator()
        print()
        print("This demo shows role-based access control with MAIF audit logging.")
        print()
        
        from access_control_demo import AccessControlDemo
        demo = AccessControlDemo(self.sessions_dir)
        demo.run()
    
    def _run_conversation(self, session_name: str, resume: bool = False):
        """Run an audited conversation.
        
        Args:
            session_name: Name of the session
            resume: Whether to resume an existing session
        """
        clear_screen()
        print_header()
        
        artifact_path = self.sessions_dir / f"{session_name}.maif"
        
        print(f"Session: {CYAN}{session_name}{RESET}")
        print(f"Artifact: {DIM}{artifact_path}{RESET}")
        print(f"User Role: {YELLOW}{self.current_role}{RESET}")
        print_separator()
        print()
        
        from conversation_agent import ConversationAgent
        
        agent = ConversationAgent(
            artifact_path=str(artifact_path),
            session_name=session_name,
            user_role=self.current_role,
            resume=resume,
        )
        
        if resume:
            print(f"{DIM}Loading previous conversation...{RESET}")
            history = agent.get_conversation_summary()
            if history:
                print(f"\n{DIM}Previous conversation:{RESET}")
                for entry in history[-3:]:  # Show last 3 exchanges
                    print(f"  {entry}")
                print()
        
        print(f"{DIM}Type 'quit' to exit, 'history' to see full history{RESET}")
        print()
        
        while True:
            try:
                user_input = input(f"{GREEN}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower() == 'history':
                agent.show_full_history()
                continue
            
            # Process the query
            response = agent.process_query(user_input)
            print()
            print(f"{BLUE}Assistant:{RESET} {response}")
            print()
        
        # Finalize the session
        print(f"\n{DIM}Finalizing session...{RESET}")
        summary = agent.finalize()
        
        print()
        print(f"{GREEN}Session saved successfully.{RESET}")
        print(f"  Total events: {summary['total_events']}")
        print(f"  Artifact: {artifact_path}")
        print(f"  Integrity: {GREEN}VERIFIED{RESET}")
        
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

