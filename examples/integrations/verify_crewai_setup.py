#!/usr/bin/env python3
"""
MAIF + CrewAI Setup Verification

Run this script to verify your MAIF + CrewAI installation is working correctly.

Usage:
    python verify_crewai_setup.py
    
If all checks pass, you're ready to use MAIF with CrewAI!
"""

import sys
import os
from pathlib import Path

# Add parent path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def print_status(name: str, passed: bool, message: str = ""):
    """Print a status check result."""
    status = "\033[92m✓\033[0m" if passed else "\033[91m✗\033[0m"
    print(f"  {status} {name}")
    if message and not passed:
        print(f"      {message}")


def check_python_version():
    """Check Python version is 3.10+."""
    version = sys.version_info
    passed = version >= (3, 10)
    message = f"Python {version.major}.{version.minor} found, need 3.10+" if not passed else ""
    return passed, message


def check_maif_import():
    """Check MAIF can be imported."""
    try:
        import maif
        return True, ""
    except ImportError as e:
        return False, f"pip install maif - {e}"


def check_crewai_import():
    """Check CrewAI can be imported."""
    try:
        import crewai
        return True, ""
    except ImportError as e:
        return False, f"pip install crewai - {e}"


def check_maif_crewai_integration():
    """Check MAIF CrewAI integration can be imported."""
    try:
        from maif.integrations.crewai import MAIFCrewCallback
        return True, ""
    except ImportError as e:
        return False, f"Integration import failed - {e}"


def check_instrument_function():
    """Check instrument function is available."""
    try:
        from maif.integrations.crewai import instrument
        return True, ""
    except ImportError as e:
        return False, f"instrument() not available - {e}"


def check_patterns():
    """Check pre-built patterns are available."""
    try:
        from maif.integrations.crewai.patterns import (
            create_research_crew,
            create_qa_crew,
            create_code_review_crew,
        )
        return True, ""
    except ImportError as e:
        return False, f"Patterns not available - {e}"


def check_memory():
    """Check MAIFCrewMemory is available."""
    try:
        from maif.integrations.crewai import MAIFCrewMemory
        return MAIFCrewMemory is not None, "MAIFCrewMemory is None"
    except ImportError as e:
        return False, f"Memory import failed - {e}"


def check_cli():
    """Check CLI tools are available."""
    try:
        from maif.integrations.crewai.cli import main
        return True, ""
    except ImportError as e:
        return False, f"CLI not available - {e}"


def test_basic_functionality():
    """Test basic callback functionality."""
    import tempfile
    import os
    
    try:
        from maif.integrations.crewai import MAIFCrewCallback, MAIFCrew
        from maif import MAIFDecoder
        
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = os.path.join(tmpdir, "test.maif")
            
            # Test context manager
            with MAIFCrew(artifact_path) as callback:
                callback.on_crew_start(crew_name="Verification Test")
                
                # Mock step
                class MockStep:
                    thought = "Testing MAIF integration"
                    action = "verify"
                    action_input = "test"
                    observation = "success"
                
                callback.on_step(MockStep())
                
                # Mock task
                class MockTask:
                    description = "Verify MAIF setup"
                    raw = "Verification complete"
                    agent = "verifier"
                    output_format = "raw"
                
                callback.on_task_complete(MockTask())
                callback.on_crew_end(result=None)
            
            # Verify artifact
            decoder = MAIFDecoder(artifact_path)
            decoder.load()
            is_valid, _ = decoder.verify_integrity()
            
            if not is_valid:
                return False, "Artifact integrity check failed"
            
            if len(decoder.blocks) < 4:
                return False, f"Expected 4+ blocks, got {len(decoder.blocks)}"
            
            return True, ""
            
    except Exception as e:
        return False, str(e)


def test_instrument_function():
    """Test instrument() one-liner."""
    import tempfile
    import os
    
    try:
        from maif.integrations.crewai import instrument
        
        # Create a mock crew-like object
        class MockCrew:
            name = "Mock Crew"
            agents = []
            tasks = []
            task_callback = None
            step_callback = None
            
            def kickoff(self, *args, **kwargs):
                # Simulate step callback
                if self.step_callback:
                    class MockStep:
                        thought = "Test"
                        action = "test"
                        action_input = "input"
                        observation = "output"
                    self.step_callback(MockStep())
                
                # Simulate task callback
                if self.task_callback:
                    class MockTask:
                        description = "Test"
                        raw = "output"
                        agent = "agent"
                        output_format = "raw"
                    self.task_callback(MockTask())
                
                return "result"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = os.path.join(tmpdir, "test.maif")
            
            crew = MockCrew()
            crew = instrument(crew, artifact_path)
            result = crew.kickoff()
            
            # Verify artifact was created
            from maif import MAIFDecoder
            decoder = MAIFDecoder(artifact_path)
            decoder.load()
            is_valid, _ = decoder.verify_integrity()
            
            if not is_valid:
                return False, "instrument() artifact failed integrity check"
            
            return True, ""
            
    except Exception as e:
        return False, str(e)


def main():
    """Run all verification checks."""
    print()
    print("=" * 60)
    print("  MAIF + CrewAI Setup Verification")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Environment checks
    print("\033[1mEnvironment:\033[0m")
    
    checks = [
        ("Python 3.10+", check_python_version),
        ("MAIF installed", check_maif_import),
        ("CrewAI installed", check_crewai_import),
    ]
    
    for name, check_fn in checks:
        passed, message = check_fn()
        print_status(name, passed, message)
        if not passed:
            all_passed = False
    
    print()
    
    # Integration checks
    print("\033[1mIntegration:\033[0m")
    
    integration_checks = [
        ("MAIF CrewAI integration", check_maif_crewai_integration),
        ("instrument() function", check_instrument_function),
        ("Pre-built patterns", check_patterns),
        ("MAIFCrewMemory", check_memory),
        ("CLI tools", check_cli),
    ]
    
    for name, check_fn in integration_checks:
        passed, message = check_fn()
        print_status(name, passed, message)
        if not passed:
            all_passed = False
    
    print()
    
    # Functional tests
    print("\033[1mFunctional Tests:\033[0m")
    
    functional_tests = [
        ("Basic callback test", test_basic_functionality),
        ("instrument() test", test_instrument_function),
    ]
    
    for name, test_fn in functional_tests:
        passed, message = test_fn()
        print_status(name, passed, message)
        if not passed:
            all_passed = False
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("\033[92m✓ All checks passed! You're ready to use MAIF + CrewAI.\033[0m")
        print()
        print("Quick start:")
        print("  from crewai import Crew")
        print("  from maif.integrations.crewai import instrument")
        print()
        print("  crew = Crew(agents=[...], tasks=[...])")
        print('  crew = instrument(crew, "audit.maif")')
        print("  result = crew.kickoff()")
        print()
        return 0
    else:
        print("\033[91m✗ Some checks failed. Please fix the issues above.\033[0m")
        print()
        print("Need help? See: docs/guide/integrations/crewai.md")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

