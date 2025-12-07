"""
Agent Memory Demonstration.

Shows how MAIF provides persistent memory for CrewAI agents
with full provenance tracking.
"""

import os
from pathlib import Path
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


def print_separator(char="-", width=60):
    """Print a separator line."""
    print(char * width)


def wait_for_enter(message="Press Enter to continue..."):
    """Wait for user to press Enter."""
    input(f"\n{DIM}{message}{RESET}")


class MemoryDemo:
    """Demonstrates MAIF persistent memory for CrewAI agents."""
    
    def __init__(self, sessions_dir: Path):
        """Initialize the demo.
        
        Args:
            sessions_dir: Directory for session artifacts
        """
        self.sessions_dir = Path(sessions_dir)
        self.memory_artifact = self.sessions_dir / "agent_memory_demo.maif"
    
    def run(self):
        """Run the memory demonstration."""
        while True:
            print()
            print(f"{BOLD}AGENT MEMORY DEMO{RESET}")
            print(f"Memory artifact: {CYAN}{self.memory_artifact}{RESET}")
            print_separator()
            print()
            print("[1] Store New Memory")
            print("[2] Search Memories")
            print("[3] View All Memories")
            print("[4] View by Agent")
            print("[5] View by Tags")
            print("[6] Clear All Memories")
            print("[7] Back to Main Menu")
            print()
            
            choice = input("Choice: ").strip()
            
            if choice == "1":
                self.store_memory()
            elif choice == "2":
                self.search_memories()
            elif choice == "3":
                self.view_all_memories()
            elif choice == "4":
                self.view_by_agent()
            elif choice == "5":
                self.view_by_tags()
            elif choice == "6":
                self.clear_memories()
            elif choice == "7":
                return
            else:
                print(f"\n{RED}Invalid choice.{RESET}")
    
    def _get_memory(self):
        """Get the memory instance."""
        from maif.integrations.crewai import MAIFCrewMemory
        return MAIFCrewMemory(str(self.memory_artifact), auto_finalize=False)
    
    def store_memory(self):
        """Store a new memory."""
        print()
        print(f"{BOLD}STORE NEW MEMORY{RESET}")
        print_separator()
        print()
        
        # Get memory content
        content = input("Memory content: ").strip()
        if not content:
            print(f"{YELLOW}Cancelled - no content provided.{RESET}")
            return
        
        # Get agent
        agent = input("Agent name (press Enter for 'default'): ").strip() or "default"
        
        # Get tags
        tags_input = input("Tags (comma-separated, press Enter for none): ").strip()
        tags = [t.strip() for t in tags_input.split(",")] if tags_input else []
        
        # Get importance
        importance_input = input("Importance (0.0-1.0, press Enter for 0.5): ").strip()
        try:
            importance = float(importance_input) if importance_input else 0.5
            importance = max(0.0, min(1.0, importance))
        except ValueError:
            importance = 0.5
        
        # Store the memory
        memory = self._get_memory()
        memory_id = memory.save(
            content=content,
            agent=agent,
            tags=tags,
            importance=importance,
        )
        memory.finalize()
        
        print()
        print(f"{GREEN}Memory stored successfully!{RESET}")
        print(f"  ID: {memory_id}")
        print(f"  Agent: {agent}")
        print(f"  Tags: {tags}")
        print(f"  Importance: {importance}")
        
        wait_for_enter()
    
    def search_memories(self):
        """Search memories by content."""
        print()
        print(f"{BOLD}SEARCH MEMORIES{RESET}")
        print_separator()
        print()
        
        query = input("Search query: ").strip()
        if not query:
            print(f"{YELLOW}Cancelled - no query provided.{RESET}")
            return
        
        memory = self._get_memory()
        results = memory.search(query, limit=10)
        
        print()
        if not results:
            print(f"{YELLOW}No matching memories found.{RESET}")
        else:
            print(f"Found {len(results)} matching memories:")
            print()
            
            for i, mem in enumerate(results, 1):
                self._display_memory(i, mem)
        
        wait_for_enter()
    
    def view_all_memories(self):
        """View all memories."""
        print()
        print(f"{BOLD}ALL MEMORIES{RESET}")
        print_separator()
        print()
        
        memory = self._get_memory()
        all_memories = memory.get_all()
        
        if not all_memories:
            print(f"{YELLOW}No memories stored yet.{RESET}")
        else:
            print(f"Total memories: {len(all_memories)}")
            print()
            
            for i, mem in enumerate(all_memories, 1):
                self._display_memory(i, mem)
        
        wait_for_enter()
    
    def view_by_agent(self):
        """View memories grouped by agent."""
        print()
        print(f"{BOLD}MEMORIES BY AGENT{RESET}")
        print_separator()
        print()
        
        memory = self._get_memory()
        all_memories = memory.get_all()
        
        # Group by agent
        by_agent = {}
        for mem in all_memories:
            agent = mem.get("agent", "unknown")
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(mem)
        
        if not by_agent:
            print(f"{YELLOW}No memories stored yet.{RESET}")
        else:
            for agent, memories in sorted(by_agent.items()):
                print(f"{CYAN}{agent}{RESET}: {len(memories)} memories")
                for mem in memories[:3]:  # Show first 3
                    content = mem.get("content", "")[:50]
                    print(f"  - {content}...")
                if len(memories) > 3:
                    print(f"  ... and {len(memories) - 3} more")
                print()
        
        wait_for_enter()
    
    def view_by_tags(self):
        """View memories filtered by tags."""
        print()
        print(f"{BOLD}MEMORIES BY TAGS{RESET}")
        print_separator()
        print()
        
        tags_input = input("Enter tags to filter (comma-separated): ").strip()
        if not tags_input:
            print(f"{YELLOW}Cancelled - no tags provided.{RESET}")
            return
        
        tags = [t.strip() for t in tags_input.split(",")]
        
        memory = self._get_memory()
        results = memory.get_by_tags(tags)
        
        print()
        if not results:
            print(f"{YELLOW}No memories found with those tags.{RESET}")
        else:
            print(f"Found {len(results)} memories with tags {tags}:")
            print()
            
            for i, mem in enumerate(results, 1):
                self._display_memory(i, mem)
        
        wait_for_enter()
    
    def clear_memories(self):
        """Clear all memories."""
        print()
        print(f"{BOLD}CLEAR ALL MEMORIES{RESET}")
        print_separator()
        print()
        
        confirm = input(f"{RED}Are you sure you want to clear all memories? (yes/no): {RESET}").strip().lower()
        
        if confirm == "yes":
            # Delete the memory artifact
            if self.memory_artifact.exists():
                os.remove(self.memory_artifact)
                print(f"{GREEN}All memories cleared.{RESET}")
            else:
                print(f"{YELLOW}No memories to clear.{RESET}")
        else:
            print("Cancelled.")
        
        wait_for_enter()
    
    def _display_memory(self, index: int, mem: dict):
        """Display a single memory."""
        content = mem.get("content", "")
        agent = mem.get("agent", "unknown")
        tags = mem.get("tags", [])
        importance = mem.get("importance", 0.5)
        created_at = mem.get("created_at", 0)
        
        # Format timestamp
        ts_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M") if created_at else "Unknown"
        
        # Importance color
        if importance >= 0.8:
            imp_color = RED
        elif importance >= 0.5:
            imp_color = YELLOW
        else:
            imp_color = DIM
        
        print(f"{index}. [{agent}] {imp_color}(importance: {importance:.1f}){RESET}")
        print(f"   {content}")
        print(f"   {DIM}Tags: {tags} | Created: {ts_str}{RESET}")
        print()

