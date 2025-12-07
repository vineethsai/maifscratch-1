"""
Tamper Detection Demonstration.

Shows how MAIF detects unauthorized modifications to artifacts.
"""

import os
import shutil
import tempfile
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


def print_separator(char="-", width=60):
    """Print a separator line."""
    print(char * width)


class TamperDemo:
    """Demonstrates MAIF tamper detection capabilities."""
    
    def __init__(self, sessions_dir: Path):
        """Initialize the demo.
        
        Args:
            sessions_dir: Directory for session artifacts
        """
        self.sessions_dir = Path(sessions_dir)
    
    def run(self):
        """Run the tamper detection demonstration."""
        print(f"{BOLD}Step 1: Create a test artifact{RESET}")
        print_separator()
        
        # Create a temporary artifact
        test_artifact = self.sessions_dir / "tamper_test.maif"
        
        print(f"Creating test artifact: {CYAN}{test_artifact}{RESET}")
        print()
        
        # Create artifact with some events
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(str(test_artifact), agent_id="tamper_demo")
        
        # Log some events
        callback.on_crew_start(crew_name="Tamper Test Crew")
        
        # Simulate task output
        class MockTaskOutput:
            description = "Test task for tamper demonstration"
            raw = "This is sensitive task output that should not be modified"
            agent = "test_agent"
            output_format = "raw"
        
        callback.on_task_complete(MockTaskOutput())
        callback.on_crew_end(result=None)
        callback.finalize()
        
        print(f"  {GREEN}Created artifact with test data{RESET}")
        print()
        
        # Step 2: Verify original integrity
        print(f"{BOLD}Step 2: Verify original integrity{RESET}")
        print_separator()
        
        from maif import MAIFDecoder
        
        decoder = MAIFDecoder(str(test_artifact))
        decoder.load()
        
        is_valid, errors = decoder.verify_integrity()
        
        print(f"  Blocks: {len(decoder.blocks)}")
        print(f"  Integrity: {GREEN}VALID{RESET}" if is_valid else f"  Integrity: {RED}INVALID{RESET}")
        print()
        
        # Step 3: Create tampered copy
        print(f"{BOLD}Step 3: Simulate tampering{RESET}")
        print_separator()
        
        tampered_artifact = self.sessions_dir / "tamper_test_modified.maif"
        
        # Copy the file
        shutil.copy(test_artifact, tampered_artifact)
        
        print(f"  Created copy: {CYAN}{tampered_artifact}{RESET}")
        
        # Modify a byte in the middle of the file
        with open(tampered_artifact, "r+b") as f:
            f.seek(500)  # Seek to middle of file
            original_byte = f.read(1)
            f.seek(500)
            # Flip a bit
            tampered_byte = bytes([original_byte[0] ^ 0x01])
            f.write(tampered_byte)
        
        print(f"  {YELLOW}Modified 1 byte at position 500{RESET}")
        print(f"  Original: 0x{original_byte.hex()}")
        print(f"  Tampered: 0x{tampered_byte.hex()}")
        print()
        
        # Step 4: Verify tampered file
        print(f"{BOLD}Step 4: Detect tampering{RESET}")
        print_separator()
        
        try:
            tampered_decoder = MAIFDecoder(str(tampered_artifact))
            tampered_decoder.load()
            
            is_valid_tampered, errors_tampered = tampered_decoder.verify_integrity()
            
            if not is_valid_tampered:
                print(f"  {GREEN}TAMPERING DETECTED!{RESET}")
                print()
                print(f"  MAIF successfully detected the unauthorized modification.")
                print(f"  Errors found: {len(errors_tampered)}")
                
                if errors_tampered:
                    print()
                    print(f"  {BOLD}Error details:{RESET}")
                    for error in errors_tampered[:3]:
                        print(f"    - {error}")
            else:
                print(f"  {YELLOW}Warning: Tampering not detected by integrity check{RESET}")
                print(f"  (Modification may have been in non-critical section)")
        except Exception as e:
            print(f"  {GREEN}TAMPERING DETECTED!{RESET}")
            print()
            print(f"  MAIF detected corruption during load:")
            print(f"    {e}")
        
        print()
        
        # Step 5: Clean up
        print(f"{BOLD}Step 5: Cleanup{RESET}")
        print_separator()
        
        # Remove test files
        if test_artifact.exists():
            os.remove(test_artifact)
            print(f"  Removed: {test_artifact.name}")
        
        if tampered_artifact.exists():
            os.remove(tampered_artifact)
            print(f"  Removed: {tampered_artifact.name}")
        
        print()
        print(f"{BOLD}Summary:{RESET}")
        print_separator()
        print("""
MAIF provides tamper detection through:

1. Hash Chains - Each block contains a hash of the previous block.
   Modifying any block invalidates all subsequent blocks.

2. Ed25519 Signatures - Every block is cryptographically signed.
   Any modification breaks the signature.

3. Integrity Verification - The verify_integrity() method checks
   all hashes and signatures in the artifact.

This ensures complete accountability and non-repudiation for all
agent actions stored in MAIF artifacts.
""")

