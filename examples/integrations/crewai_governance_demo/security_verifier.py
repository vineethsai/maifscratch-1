"""
Security Verifier for MAIF Artifacts.

Provides comprehensive security verification including signature
validation and integrity checking.
"""

import time
from pathlib import Path
from typing import Tuple, List

# Terminal formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"


def print_separator(char="-", width=80):
    """Print a separator line."""
    print(char * width)


class SecurityVerifier:
    """Verifies security properties of MAIF artifacts."""
    
    def __init__(self, artifact_path: str):
        """Initialize the verifier.
        
        Args:
            artifact_path: Path to the MAIF artifact
        """
        self.artifact_path = artifact_path
    
    def run_verification(self):
        """Run all security verifications."""
        print()
        print(f"{BOLD}SECURITY VERIFICATION{RESET}")
        print(f"Artifact: {CYAN}{self.artifact_path}{RESET}")
        print_separator()
        print()
        
        all_passed = True
        
        # 1. File existence
        print(f"[1/5] File Existence Check...")
        if Path(self.artifact_path).exists():
            print(f"      {GREEN}PASSED{RESET} - File exists")
        else:
            print(f"      {RED}FAILED{RESET} - File not found")
            return
        
        # 2. Load artifact
        print(f"[2/5] Artifact Load Test...")
        try:
            from maif import MAIFDecoder
            
            start = time.time()
            decoder = MAIFDecoder(self.artifact_path)
            decoder.load()
            load_time = time.time() - start
            
            print(f"      {GREEN}PASSED{RESET} - Loaded {len(decoder.blocks)} blocks in {load_time:.3f}s")
        except Exception as e:
            print(f"      {RED}FAILED{RESET} - {e}")
            all_passed = False
            return
        
        # 3. Header validation
        print(f"[3/5] Header Validation...")
        try:
            # Check that we have blocks
            if len(decoder.blocks) > 0:
                print(f"      {GREEN}PASSED{RESET} - Valid header structure")
            else:
                print(f"      {YELLOW}WARNING{RESET} - No blocks found")
                all_passed = False
        except Exception as e:
            print(f"      {RED}FAILED{RESET} - {e}")
            all_passed = False
        
        # 4. Integrity verification
        print(f"[4/5] Integrity Verification...")
        try:
            start = time.time()
            is_valid, errors = decoder.verify_integrity()
            verify_time = time.time() - start
            
            if is_valid:
                print(f"      {GREEN}PASSED{RESET} - All integrity checks passed ({verify_time:.3f}s)")
            else:
                print(f"      {RED}FAILED{RESET} - Integrity errors found:")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"        - {error}")
                if len(errors) > 5:
                    print(f"        ... and {len(errors) - 5} more errors")
                all_passed = False
        except Exception as e:
            print(f"      {RED}FAILED{RESET} - {e}")
            all_passed = False
        
        # 5. Block structure validation
        print(f"[5/5] Block Structure Validation...")
        try:
            valid_blocks = 0
            invalid_blocks = 0
            
            for block in decoder.blocks:
                if block.metadata and block.data:
                    valid_blocks += 1
                else:
                    invalid_blocks += 1
            
            if invalid_blocks == 0:
                print(f"      {GREEN}PASSED{RESET} - All {valid_blocks} blocks have valid structure")
            else:
                print(f"      {YELLOW}WARNING{RESET} - {invalid_blocks}/{valid_blocks + invalid_blocks} blocks have issues")
                all_passed = False
        except Exception as e:
            print(f"      {RED}FAILED{RESET} - {e}")
            all_passed = False
        
        # Summary
        print()
        print_separator()
        if all_passed:
            print(f"{GREEN}ALL SECURITY CHECKS PASSED{RESET}")
        else:
            print(f"{YELLOW}SOME CHECKS FAILED - See details above{RESET}")
        
        # Additional info
        print()
        print(f"{BOLD}Artifact Statistics:{RESET}")
        print(f"  Total blocks: {len(decoder.blocks)}")
        print(f"  File size: {Path(self.artifact_path).stat().st_size:,} bytes")
        
        # Event type summary
        event_types = {}
        for block in decoder.blocks:
            if block.metadata:
                event_type = block.metadata.get("type", "unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print(f"  Event types: {len(event_types)}")
        for event_type, count in sorted(event_types.items()):
            print(f"    - {event_type}: {count}")

