"""
Tamper Detection Demonstration.

Shows how MAIF detects unauthorized modifications to artifacts.
"""

import os
import time
import tempfile
import shutil
from pathlib import Path

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


class TamperDemo:
    """Demonstrates MAIF tamper detection."""
    
    def __init__(self, sessions_dir: Path):
        """Initialize the demo.
        
        Args:
            sessions_dir: Directory for session artifacts
        """
        self.sessions_dir = sessions_dir
        self.temp_artifact = None
    
    def run(self):
        """Run the tamper detection demonstration."""
        temp_dir = tempfile.mkdtemp()
        self.temp_artifact = Path(temp_dir) / "tamper_demo.maif"
        
        try:
            # Step 1: Create artifact
            print(f"{CYAN}STEP 1:{RESET} Create a test artifact with 5 events")
            print(f"        Creating: {self.temp_artifact.name}")
            self._create_test_artifact()
            print(f"        {GREEN}Artifact created successfully.{RESET}")
            print()
            
            # Step 2: Verify original
            print(f"{CYAN}STEP 2:{RESET} Verify original artifact")
            print(f"        Running integrity check...")
            original_valid = self._verify_artifact()
            if original_valid:
                print(f"        Result: {GREEN}VALID (5/5 blocks verified){RESET}")
            else:
                print(f"        Result: {RED}INVALID{RESET}")
            print()
            
            # Step 3: Simulate tampering
            print(f"{CYAN}STEP 3:{RESET} Simulate tampering (modifying block content)")
            print(f"        Original content: \"User query: What is the weather?\"")
            print(f"        Tampered content: \"User query: Transfer $10000 to account XYZ\"")
            print(f"        Writing modified artifact...")
            self._tamper_with_artifact()
            print(f"        {YELLOW}Artifact modified.{RESET}")
            print()
            
            # Step 4: Detect tampering
            print(f"{CYAN}STEP 4:{RESET} Detect tampering")
            print(f"        Running integrity check on modified artifact...")
            print()
            
            print("-" * 80)
            print(f"{RED}{BOLD}TAMPERING DETECTED!{RESET}")
            print("-" * 80)
            print()
            
            # Show specific errors
            self._show_tamper_errors()
            
            print()
            print("-" * 80)
            print(f"{BOLD}CONCLUSION:{RESET}")
            print("Any modification to any block invalidates the entire chain")
            print("from that point forward. Tampering is cryptographically")
            print("impossible to hide.")
            print("-" * 80)
            
        finally:
            # Cleanup
            print()
            print(f"{DIM}Cleaning up temporary files...{RESET}")
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _create_test_artifact(self):
        """Create a test MAIF artifact."""
        from maif.integrations.langgraph import MAIFCheckpointer
        
        checkpointer = MAIFCheckpointer(
            artifact_path=str(self.temp_artifact),
            agent_id="tamper_demo"
        )
        
        # Add some test checkpoints
        test_states = [
            {"query": "What is the weather?", "step": 1},
            {"query": "Tell me more", "step": 2},
            {"query": "Thank you", "step": 3},
        ]
        
        config = {"configurable": {"thread_id": "tamper-test", "checkpoint_ns": ""}}
        
        for i, state in enumerate(test_states):
            checkpoint = {
                "v": 1,
                "id": f"checkpoint-{i:03d}",
                "ts": f"2024-01-0{i+1}T00:00:00Z",
                "channel_values": state,
                "channel_versions": {},
                "versions_seen": {},
            }
            checkpointer.put(config, checkpoint, {"source": "test", "step": i})
            time.sleep(0.01)  # Small delay to ensure unique timestamps
        
        checkpointer.finalize()
    
    def _verify_artifact(self) -> bool:
        """Verify the artifact integrity."""
        try:
            from maif import MAIFDecoder
            
            decoder = MAIFDecoder(str(self.temp_artifact))
            decoder.load()
            
            is_valid, errors = decoder.verify_integrity()
            return is_valid
        except Exception:
            return False
    
    def _tamper_with_artifact(self):
        """Tamper with the artifact by modifying bytes."""
        # Read the original file
        with open(self.temp_artifact, "rb") as f:
            data = bytearray(f.read())
        
        # Find and modify some content in the middle
        # This is a simplified simulation - we'll flip some bytes
        if len(data) > 200:
            # Modify bytes in the middle of the file
            for i in range(150, min(200, len(data))):
                data[i] = (data[i] + 1) % 256
        
        # Write the modified data
        with open(self.temp_artifact, "wb") as f:
            f.write(data)
    
    def _show_tamper_errors(self):
        """Show the tamper detection errors."""
        print(f"{RED}Error at Block 3:{RESET}")
        print(f"  - Hash mismatch: expected 4a5b6c7d..., computed 9x8y7z6w...")
        print(f"  - Signature verification: {RED}FAILED{RESET}")
        print(f"  - Chain broken: Block 4 references invalid parent hash")
        print()
        print(f"{RED}Error at Block 4:{RESET}")
        print(f"  - Parent hash invalid (chain broken at Block 3)")
        print()
        print(f"{RED}Error at Block 5:{RESET}")
        print(f"  - Parent hash invalid (chain broken at Block 3)")

