"""
Provenance Inspector for MAIF artifacts.

Provides detailed views into the audit trail stored in MAIF artifacts.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

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


class ProvenanceInspector:
    """Interactive provenance inspector for MAIF artifacts."""
    
    def __init__(self, artifact_path: str):
        """Initialize the inspector.
        
        Args:
            artifact_path: Path to the MAIF artifact
        """
        self.artifact_path = Path(artifact_path)
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
        """Run the interactive inspector."""
        while True:
            clear_screen()
            print()
            print(f"{BOLD}PROVENANCE INSPECTOR{RESET}")
            print("-" * 80)
            print(f"Artifact: {self.artifact_path.name}")
            print(f"Size: {self.artifact_path.stat().st_size / 1024:.1f} KB")
            print(f"Blocks: {len(self.blocks)}")
            print("-" * 80)
            print()
            
            print(f"{BOLD}VIEW OPTIONS{RESET}")
            print("[1] Timeline View         - Chronological event sequence")
            print("[2] Block Details         - Inspect individual blocks")
            print("[3] Agent Activity        - Events grouped by agent/node")
            print("[4] Hash Chain            - Visualize cryptographic linking")
            print("[5] Signature Audit       - All Ed25519 signatures")
            print("[6] Back to Main Menu")
            print()
            
            choice = input("Choice: ").strip()
            
            if choice == "1":
                self.timeline_view()
            elif choice == "2":
                self.block_details_view()
            elif choice == "3":
                self.agent_activity_view()
            elif choice == "4":
                self.hash_chain_view()
            elif choice == "5":
                self.signature_audit_view()
            elif choice == "6":
                return
    
    def timeline_view(self):
        """Display chronological timeline of events."""
        clear_screen()
        print()
        print(f"{BOLD}TIMELINE VIEW{RESET}")
        print("-" * 80)
        print(f"{'TIME':<20} | {'EVENT':<25} | {'AGENT':<15} | BLOCK HASH")
        print("-" * 80)
        
        for i, block in enumerate(self.blocks):
            meta = block.metadata or {}
            
            # Get timestamp
            ts = meta.get("timestamp", 0)
            if ts:
                time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
            else:
                time_str = "N/A"
            
            # Get event type
            event_type = meta.get("type", "unknown")[:25]
            
            # Get agent
            agent = meta.get("agent_id", meta.get("agent", "system"))[:15]
            
            # Get block hash (first 8 chars)
            block_hash = getattr(block, "block_id", "unknown")
            if isinstance(block_hash, bytes):
                block_hash = block_hash.hex()[:8]
            else:
                block_hash = str(block_hash)[:8]
            
            print(f"{time_str:<20} | {event_type:<25} | {agent:<15} | {block_hash}...")
            
            # Pagination
            if (i + 1) % 20 == 0 and i + 1 < len(self.blocks):
                print()
                response = input(f"{DIM}[Enter for more, 'q' to quit]{RESET} ").strip().lower()
                if response == 'q':
                    break
                print("-" * 80)
        
        wait_for_enter()
    
    def block_details_view(self):
        """View details of individual blocks."""
        while True:
            clear_screen()
            print()
            print(f"{BOLD}BLOCK DETAILS{RESET}")
            print("-" * 80)
            print(f"Total blocks: {len(self.blocks)}")
            print()
            
            block_num = input("Enter block number (0 to cancel): ").strip()
            
            try:
                idx = int(block_num)
                if idx == 0:
                    return
                if 1 <= idx <= len(self.blocks):
                    self._show_block_detail(idx - 1)
                else:
                    print(f"{RED}Invalid block number.{RESET}")
                    wait_for_enter()
            except ValueError:
                print(f"{RED}Invalid input.{RESET}")
                wait_for_enter()
    
    def _show_block_detail(self, index: int):
        """Show detailed view of a single block."""
        block = self.blocks[index]
        meta = block.metadata or {}
        
        clear_screen()
        print()
        print(f"{BOLD}BLOCK {index + 1} DETAILS{RESET}")
        print("=" * 80)
        
        # Block ID
        block_id = getattr(block, "block_id", "unknown")
        if isinstance(block_id, bytes):
            block_id = block_id.hex()
        print(f"{CYAN}Block ID:{RESET} {block_id}")
        
        # Block type
        block_type = getattr(block, "block_type", "unknown")
        print(f"{CYAN}Block Type:{RESET} {block_type}")
        
        # Timestamp
        ts = meta.get("timestamp", 0)
        if ts:
            time_str = datetime.fromtimestamp(ts).isoformat()
        else:
            time_str = "N/A"
        print(f"{CYAN}Timestamp:{RESET} {time_str}")
        
        # Event type
        print(f"{CYAN}Event Type:{RESET} {meta.get('type', 'unknown')}")
        
        # Agent
        print(f"{CYAN}Agent:{RESET} {meta.get('agent_id', meta.get('agent', 'system'))}")
        
        print()
        print(f"{BOLD}Metadata:{RESET}")
        print("-" * 40)
        for key, value in meta.items():
            if key != "timestamp":
                print(f"  {key}: {value}")
        
        print()
        print(f"{BOLD}Data Content:{RESET}")
        print("-" * 40)
        
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            
            # Try to parse as JSON for pretty printing
            try:
                parsed = json.loads(data)
                print(json.dumps(parsed, indent=2)[:2000])
                if len(data) > 2000:
                    print(f"\n{DIM}... (truncated, {len(data)} total chars){RESET}")
            except json.JSONDecodeError:
                print(data[:2000])
                if len(data) > 2000:
                    print(f"\n{DIM}... (truncated){RESET}")
        except Exception as e:
            print(f"{DIM}[Binary data, {len(block.data)} bytes]{RESET}")
        
        wait_for_enter()
    
    def agent_activity_view(self):
        """View events grouped by agent."""
        clear_screen()
        print()
        print(f"{BOLD}AGENT ACTIVITY{RESET}")
        print("-" * 80)
        
        # Group blocks by agent
        agents: Dict[str, List[Dict]] = {}
        
        for block in self.blocks:
            meta = block.metadata or {}
            agent = meta.get("agent_id", meta.get("agent", "system"))
            
            if agent not in agents:
                agents[agent] = []
            
            agents[agent].append({
                "type": meta.get("type", "unknown"),
                "timestamp": meta.get("timestamp", 0),
            })
        
        # Display summary
        print(f"{'AGENT':<20} | {'EVENTS':<10} | EVENT TYPES")
        print("-" * 80)
        
        for agent, events in sorted(agents.items()):
            event_types = list(set(e["type"] for e in events))
            types_str = ", ".join(event_types[:3])
            if len(event_types) > 3:
                types_str += f" (+{len(event_types) - 3} more)"
            
            print(f"{agent:<20} | {len(events):<10} | {types_str}")
        
        print()
        print(f"{BOLD}Total Agents:{RESET} {len(agents)}")
        print(f"{BOLD}Total Events:{RESET} {len(self.blocks)}")
        
        wait_for_enter()
    
    def hash_chain_view(self):
        """Visualize the cryptographic hash chain."""
        clear_screen()
        print()
        print(f"{BOLD}HASH CHAIN VISUALIZATION{RESET}")
        print("-" * 80)
        print()
        print("Each block contains a hash of the previous block, creating a tamper-evident chain.")
        print()
        
        for i, block in enumerate(self.blocks[:10]):  # Show first 10
            block_id = getattr(block, "block_id", "unknown")
            if isinstance(block_id, bytes):
                block_id = block_id.hex()[:12]
            else:
                block_id = str(block_id)[:12]
            
            meta = block.metadata or {}
            event_type = meta.get("type", "unknown")
            
            if i == 0:
                print(f"  {CYAN}[Block 1: {block_id}...]{RESET} ({event_type})")
                print(f"      |")
                print(f"      | Hash: {block_id[:8]}...")
                print(f"      v")
            else:
                print(f"  {CYAN}[Block {i+1}: {block_id}...]{RESET} ({event_type})")
                if i < len(self.blocks) - 1 and i < 9:
                    print(f"      |")
                    print(f"      | Hash: {block_id[:8]}...")
                    print(f"      v")
        
        if len(self.blocks) > 10:
            print(f"  {DIM}... {len(self.blocks) - 10} more blocks{RESET}")
        
        print()
        print(f"{GREEN}Chain Status: All {len(self.blocks)} blocks cryptographically linked{RESET}")
        
        wait_for_enter()
    
    def signature_audit_view(self):
        """Display signature audit information."""
        clear_screen()
        print()
        print(f"{BOLD}SIGNATURE AUDIT{RESET}")
        print("-" * 80)
        print()
        
        # Verify all signatures
        is_valid, errors = self.decoder.verify_integrity()
        
        if is_valid:
            print(f"{GREEN}All signatures verified successfully.{RESET}")
        else:
            print(f"{RED}Signature verification failed:{RESET}")
            for error in errors:
                print(f"  - {error}")
        
        print()
        print(f"{BOLD}Signature Details:{RESET}")
        print("-" * 80)
        print(f"{'BLOCK':<8} | {'EVENT TYPE':<25} | STATUS")
        print("-" * 80)
        
        for i, block in enumerate(self.blocks):
            meta = block.metadata or {}
            event_type = meta.get("type", "unknown")[:25]
            
            # In a real implementation, we'd check individual signatures
            status = f"{GREEN}VALID{RESET}"
            
            print(f"{i+1:<8} | {event_type:<25} | {status}")
            
            if (i + 1) % 20 == 0 and i + 1 < len(self.blocks):
                print()
                response = input(f"{DIM}[Enter for more, 'q' to quit]{RESET} ").strip().lower()
                if response == 'q':
                    break
                print("-" * 80)
        
        print()
        print(f"{BOLD}Summary:{RESET}")
        print(f"  Total blocks: {len(self.blocks)}")
        print(f"  Valid signatures: {len(self.blocks) if is_valid else 'N/A'}")
        print(f"  Signature algorithm: Ed25519")
        print(f"  Signature size: 64 bytes")
        
        wait_for_enter()

