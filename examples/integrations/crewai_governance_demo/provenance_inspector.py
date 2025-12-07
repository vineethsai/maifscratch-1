"""
Provenance Inspector for CrewAI MAIF Artifacts.

Provides detailed inspection of audit trails in MAIF artifacts.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Terminal formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"


def print_separator(char="-", width=80):
    """Print a separator line."""
    print(char * width)


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
        self.artifact_path = artifact_path
        self.decoder = None
        self.blocks = []
        self._load_artifact()
    
    def _load_artifact(self):
        """Load the MAIF artifact."""
        from maif import MAIFDecoder
        
        self.decoder = MAIFDecoder(self.artifact_path)
        self.decoder.load()
        self.blocks = self.decoder.blocks
    
    def run_interactive(self):
        """Run interactive inspection menu."""
        while True:
            print()
            print(f"{BOLD}PROVENANCE INSPECTOR{RESET}")
            print(f"Artifact: {CYAN}{self.artifact_path}{RESET}")
            print(f"Total blocks: {len(self.blocks)}")
            print_separator()
            print()
            print("[1] Timeline View           - Chronological event sequence")
            print("[2] Block Details           - Inspect individual blocks")
            print("[3] Event Type Summary      - Events grouped by type")
            print("[4] Agent Activity          - Events grouped by agent")
            print("[5] Task & Step Analysis    - CrewAI-specific events")
            print("[6] Hash Chain View         - Cryptographic linking")
            print("[7] Back to Main Menu")
            print()
            
            choice = input("Choice: ").strip()
            
            if choice == "1":
                self.show_timeline()
            elif choice == "2":
                self.show_block_details()
            elif choice == "3":
                self.show_event_summary()
            elif choice == "4":
                self.show_agent_activity()
            elif choice == "5":
                self.show_task_analysis()
            elif choice == "6":
                self.show_hash_chain()
            elif choice == "7":
                return
            else:
                print(f"\n{RED}Invalid choice.{RESET}")
    
    def show_timeline(self):
        """Show chronological timeline of events."""
        print()
        print(f"{BOLD}TIMELINE VIEW{RESET}")
        print_separator()
        print()
        
        for i, block in enumerate(self.blocks):
            metadata = block.metadata or {}
            event_type = metadata.get("type", "unknown")
            subtype = metadata.get("event_subtype", "")
            timestamp = metadata.get("timestamp", 0)
            
            ts_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3] if timestamp else "?"
            
            # Color-code by event type
            if "error" in event_type.lower():
                color = RED
            elif event_type in ["task_end", "agent_end"]:
                color = GREEN
            elif event_type in ["agent_action", "tool_end"]:
                color = CYAN
            else:
                color = RESET
            
            label = event_type
            if subtype:
                label += f" ({subtype})"
            
            print(f"  {i+1:3}. [{ts_str}] {color}{label}{RESET}")
        
        wait_for_enter()
    
    def show_block_details(self):
        """Show details of a specific block."""
        print()
        print(f"Enter block number (1-{len(self.blocks)}): ", end="")
        try:
            block_num = int(input().strip())
            if 1 <= block_num <= len(self.blocks):
                self._display_block(block_num - 1)
            else:
                print(f"{RED}Invalid block number.{RESET}")
        except ValueError:
            print(f"{RED}Invalid input.{RESET}")
        
        wait_for_enter()
    
    def _display_block(self, index: int):
        """Display details of a block."""
        block = self.blocks[index]
        metadata = block.metadata or {}
        
        print()
        print(f"{BOLD}BLOCK {index + 1}{RESET}")
        print_separator()
        
        # Metadata
        print(f"\n{CYAN}Metadata:{RESET}")
        for key, value in metadata.items():
            if key == "timestamp" and value:
                value = datetime.fromtimestamp(value).isoformat()
            print(f"  {key}: {value}")
        
        # Data preview
        print(f"\n{CYAN}Data Preview:{RESET}")
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            
            parsed = json.loads(data)
            
            # Pretty print with truncation
            formatted = json.dumps(parsed, indent=2)
            if len(formatted) > 1500:
                formatted = formatted[:1500] + "\n... [truncated]"
            print(formatted)
        except Exception as e:
            print(f"  {DIM}(Unable to parse: {e}){RESET}")
    
    def show_event_summary(self):
        """Show summary of events by type."""
        print()
        print(f"{BOLD}EVENT TYPE SUMMARY{RESET}")
        print_separator()
        print()
        
        event_counts: Dict[str, int] = {}
        for block in self.blocks:
            metadata = block.metadata or {}
            event_type = metadata.get("type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Sort by count
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Event Type':<30} | {'Count':>8}")
        print_separator(width=42)
        
        for event_type, count in sorted_events:
            print(f"{event_type:<30} | {count:>8}")
        
        print_separator(width=42)
        print(f"{'TOTAL':<30} | {len(self.blocks):>8}")
        
        wait_for_enter()
    
    def show_agent_activity(self):
        """Show events grouped by agent."""
        print()
        print(f"{BOLD}AGENT ACTIVITY{RESET}")
        print_separator()
        print()
        
        agent_events: Dict[str, List[Dict]] = {}
        
        for block in self.blocks:
            metadata = block.metadata or {}
            agent_id = metadata.get("agent_id", "system")
            
            if agent_id not in agent_events:
                agent_events[agent_id] = []
            
            agent_events[agent_id].append({
                "type": metadata.get("type", "unknown"),
                "timestamp": metadata.get("timestamp", 0),
            })
        
        for agent_id, events in agent_events.items():
            print(f"{CYAN}{agent_id}{RESET}: {len(events)} events")
            
            # Show event type breakdown
            type_counts: Dict[str, int] = {}
            for event in events:
                event_type = event["type"]
                type_counts[event_type] = type_counts.get(event_type, 0) + 1
            
            for event_type, count in type_counts.items():
                print(f"  - {event_type}: {count}")
            print()
        
        wait_for_enter()
    
    def show_task_analysis(self):
        """Show CrewAI-specific task and step analysis."""
        print()
        print(f"{BOLD}TASK & STEP ANALYSIS{RESET}")
        print_separator()
        print()
        
        tasks = []
        steps = []
        
        for block in self.blocks:
            metadata = block.metadata or {}
            event_type = metadata.get("type", "")
            
            if event_type == "task_end":
                try:
                    data = json.loads(block.data.decode("utf-8") if isinstance(block.data, bytes) else block.data)
                    task_data = data.get("data", {})
                    tasks.append({
                        "description": task_data.get("task_description", "")[:50],
                        "agent": task_data.get("agent", "unknown"),
                        "task_number": task_data.get("task_number", 0),
                    })
                except Exception:
                    pass
            
            elif event_type == "agent_action":
                try:
                    data = json.loads(block.data.decode("utf-8") if isinstance(block.data, bytes) else block.data)
                    step_data = data.get("data", {})
                    steps.append({
                        "action": step_data.get("action", "unknown"),
                        "step_number": step_data.get("step_number", 0),
                    })
                except Exception:
                    pass
        
        # Display tasks
        print(f"{CYAN}Tasks Completed:{RESET} {len(tasks)}")
        for task in tasks:
            print(f"  {task['task_number']}. [{task['agent']}] {task['description']}...")
        
        print()
        
        # Display step summary
        print(f"{CYAN}Steps Executed:{RESET} {len(steps)}")
        
        action_counts: Dict[str, int] = {}
        for step in steps:
            action = step["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {action}: {count}")
        
        wait_for_enter()
    
    def show_hash_chain(self):
        """Show the cryptographic hash chain."""
        print()
        print(f"{BOLD}HASH CHAIN VISUALIZATION{RESET}")
        print_separator()
        print()
        print("Each block contains a hash linking to the previous block.")
        print("This creates a tamper-evident chain of custody.")
        print()
        
        # Show first few and last few blocks
        display_count = min(5, len(self.blocks))
        
        for i in range(display_count):
            block = self.blocks[i]
            metadata = block.metadata or {}
            event_type = metadata.get("type", "unknown")
            
            # Get hash info (simplified visualization)
            block_hash = f"0x{hash(str(block.data)) & 0xFFFFFFFF:08x}"
            
            print(f"Block {i+1}: {event_type}")
            print(f"  Hash: {CYAN}{block_hash}{RESET}")
            if i < display_count - 1:
                print(f"    |")
                print(f"    v")
        
        if len(self.blocks) > display_count * 2:
            print(f"    ...")
            print(f"    ({len(self.blocks) - display_count * 2} more blocks)")
            print(f"    ...")
            
            # Show last few
            for i in range(len(self.blocks) - display_count, len(self.blocks)):
                block = self.blocks[i]
                metadata = block.metadata or {}
                event_type = metadata.get("type", "unknown")
                block_hash = f"0x{hash(str(block.data)) & 0xFFFFFFFF:08x}"
                
                if i > len(self.blocks) - display_count:
                    print(f"    |")
                    print(f"    v")
                print(f"Block {i+1}: {event_type}")
                print(f"  Hash: {CYAN}{block_hash}{RESET}")
        
        wait_for_enter()

