"""
Tests for MAIF CrewAI Integration.

This module contains unit tests and integration tests for MAIFCrewCallback
and MAIFCrewMemory. Unit tests can run without external dependencies.
Integration tests require the GEMINI_API_KEY environment variable and CrewAI.

Run unit tests only:
    pytest tests/integrations/test_crewai.py -m "not integration"

Run all tests including integration:
    GEMINI_API_KEY=your_key pytest tests/integrations/test_crewai.py
"""

import os
import json
import time
import pytest
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Mock CrewAI objects for unit testing without CrewAI installed
@dataclass
class MockTaskOutput:
    """Mock CrewAI TaskOutput for testing."""
    description: str = "Test task description"
    raw: str = "Test task output"
    agent: str = "test_agent"
    output_format: str = "raw"
    pydantic: Any = None
    json_dict: Optional[Dict] = None


@dataclass  
class MockStepOutput:
    """Mock CrewAI step output for testing."""
    thought: str = "I need to think about this"
    action: str = "search"
    action_input: str = "test query"
    observation: str = "Search result: found something"


@dataclass
class MockAgent:
    """Mock CrewAI Agent for testing."""
    role: str = "researcher"
    goal: str = "Find information"
    backstory: str = "An expert researcher"
    allow_delegation: bool = False
    tools: List[Any] = None
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []


@dataclass
class MockTask:
    """Mock CrewAI Task for testing."""
    description: str = "Research the topic"
    expected_output: str = "A detailed report"
    agent: Optional[MockAgent] = None
    tools: List[Any] = None
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []


@dataclass
class MockCrewOutput:
    """Mock CrewAI CrewOutput for testing."""
    raw: str = "Final crew output"
    tasks_output: List[MockTaskOutput] = None
    token_usage: Any = None
    
    def __post_init__(self):
        if self.tasks_output is None:
            self.tasks_output = []


@dataclass
class MockTokenUsage:
    """Mock token usage stats."""
    total_tokens: int = 1000
    prompt_tokens: int = 700
    completion_tokens: int = 300


class TestMAIFCrewCallbackUnit:
    """Unit tests for MAIFCrewCallback (no external API required)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "test_crew.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_import(self):
        """Test that MAIFCrewCallback can be imported."""
        from maif.integrations.crewai import MAIFCrewCallback
        assert MAIFCrewCallback is not None
    
    def test_import_all_classes(self):
        """Test that all public classes can be imported."""
        from maif.integrations.crewai import (
            MAIFCrewCallback,
            MAIFTaskCallback,
            MAIFStepCallback,
        )
        assert MAIFCrewCallback is not None
        assert MAIFTaskCallback is not None
        assert MAIFStepCallback is not None
    
    def test_initialization(self):
        """Test callback initialization."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        assert callback.tracker is not None
        assert callback.task_count == 0
        assert callback.step_count == 0
        assert callback.get_framework_name() == "crewai"
    
    def test_initialization_with_custom_agent_id(self):
        """Test callback initialization with custom agent ID."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(
            self.artifact_path,
            agent_id="custom-crew-agent"
        )
        
        assert callback.tracker.agent_id == "custom-crew-agent"
    
    def test_get_artifact_path(self):
        """Test get_artifact_path method."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        assert callback.get_artifact_path() == self.artifact_path
    
    def test_on_task_complete(self):
        """Test task completion callback."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        # Create mock task output
        task_output = MockTaskOutput(
            description="Research AI frameworks",
            raw="Found 5 popular frameworks: LangChain, CrewAI, ...",
            agent="researcher",
        )
        
        # Call the callback
        callback.on_task_complete(task_output)
        
        assert callback.task_count == 1
        
        callback.finalize()
    
    def test_on_step(self):
        """Test step callback."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        # Create mock step output
        step_output = MockStepOutput(
            thought="I should search for information",
            action="search_tool",
            action_input="AI frameworks comparison",
            observation="Found several articles on the topic",
        )
        
        # Call the callback
        callback.on_step(step_output)
        
        assert callback.step_count == 1
        assert callback.tool_call_count == 1  # Non-final action counts as tool call
        
        callback.finalize()
    
    def test_on_step_final_answer(self):
        """Test step callback with Final Answer action."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        # Create mock step output with Final Answer
        step_output = MockStepOutput(
            thought="I have gathered enough information",
            action="Final Answer",
            action_input="Here is my complete analysis...",
            observation="",
        )
        
        callback.on_step(step_output)
        
        assert callback.step_count == 1
        assert callback.tool_call_count == 0  # Final Answer doesn't count as tool call
        
        callback.finalize()
    
    def test_crew_lifecycle(self):
        """Test full crew lifecycle: start, tasks, steps, end."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        # Create mock agents and tasks
        agents = [
            MockAgent(role="researcher", goal="Find information"),
            MockAgent(role="writer", goal="Write reports"),
        ]
        
        tasks = [
            MockTask(description="Research topic", agent=agents[0]),
            MockTask(description="Write summary", agent=agents[1]),
        ]
        
        # Start crew
        run_id = callback.on_crew_start(
            crew_name="Test Crew",
            agents=agents,
            tasks=tasks,
            inputs={"topic": "AI"}
        )
        
        assert run_id is not None
        assert callback._current_run_id == run_id
        
        # Simulate task completion
        callback.on_task_complete(MockTaskOutput(
            description="Research topic",
            raw="Research completed",
            agent="researcher",
        ))
        
        # Simulate steps
        callback.on_step(MockStepOutput(
            thought="Starting research",
            action="search",
            action_input="AI news",
            observation="Found articles",
        ))
        
        # End crew
        result = MockCrewOutput(
            raw="Final output from the crew",
            tasks_output=[
                MockTaskOutput(description="Research topic", raw="Research done"),
                MockTaskOutput(description="Write summary", raw="Summary written"),
            ],
            token_usage=MockTokenUsage(),
        )
        
        callback.on_crew_end(result=result)
        
        # Verify statistics
        stats = callback.get_statistics()
        assert stats["tasks_completed"] == 1
        assert stats["steps_executed"] == 1
        assert stats["tool_calls"] == 1
        assert stats["run_id"] == run_id
        
        callback.finalize()
    
    def test_crew_lifecycle_with_error(self):
        """Test crew lifecycle with error."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        run_id = callback.on_crew_start(crew_name="Error Crew")
        
        # Simulate error
        error = ValueError("Something went wrong")
        callback.on_crew_end(error=error)
        
        assert callback.error_count == 1
        
        callback.finalize()
    
    def test_reset_statistics(self):
        """Test resetting statistics for new run."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        # Run first crew
        callback.on_crew_start(crew_name="Crew 1")
        callback.on_task_complete(MockTaskOutput())
        callback.on_step(MockStepOutput())
        
        assert callback.task_count == 1
        assert callback.step_count == 1
        
        # Reset for new run
        callback.reset_statistics()
        
        assert callback.task_count == 0
        assert callback.step_count == 0
        assert callback._current_run_id is None
        
        callback.finalize()
    
    def test_artifact_creation(self):
        """Test that MAIF artifact is created correctly."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        callback.on_crew_start(crew_name="Test Crew")
        callback.on_task_complete(MockTaskOutput())
        callback.on_step(MockStepOutput())
        callback.on_crew_end(result=MockCrewOutput())
        
        callback.finalize()
        
        # Verify artifact exists and can be read
        assert os.path.exists(self.artifact_path)
        
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        # Should have multiple blocks
        assert len(decoder.blocks) >= 3
        
        # Verify integrity
        is_valid, errors = decoder.verify_integrity()
        assert is_valid, f"Integrity check failed: {errors}"
    
    def test_task_callback_standalone(self):
        """Test standalone task callback."""
        from maif.integrations.crewai import MAIFTaskCallback
        
        task_callback = MAIFTaskCallback(self.artifact_path)
        
        # Use as callable
        task_callback(MockTaskOutput(
            description="Test task",
            raw="Task result",
        ))
        
        assert task_callback._task_count == 1
        
        task_callback.finalize()
        assert os.path.exists(self.artifact_path)
    
    def test_step_callback_standalone(self):
        """Test standalone step callback."""
        from maif.integrations.crewai import MAIFStepCallback
        
        step_callback = MAIFStepCallback(self.artifact_path)
        
        # Use as callable
        step_callback(MockStepOutput(
            thought="Thinking...",
            action="tool_call",
            action_input="input",
            observation="output",
        ))
        
        assert step_callback._step_count == 1
        
        step_callback.finalize()
        assert os.path.exists(self.artifact_path)


class TestMAIFCrewMemoryUnit:
    """Unit tests for MAIFCrewMemory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "memory.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_import(self):
        """Test that MAIFCrewMemory can be imported."""
        from maif.integrations.crewai import MAIFCrewMemory
        assert MAIFCrewMemory is not None
    
    def test_initialization(self):
        """Test memory initialization."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        assert memory.artifact_path == Path(self.artifact_path)
        assert len(memory) == 0
        
        memory.finalize()
    
    def test_save_and_retrieve(self):
        """Test saving and retrieving memories."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        # Save a memory
        memory_id = memory.save(
            content="The user prefers concise responses",
            agent="assistant",
            tags=["preference", "user"],
            importance=0.8,
        )
        
        assert memory_id is not None
        assert len(memory) == 1
        
        # Get all memories
        all_memories = memory.get_all()
        assert len(all_memories) == 1
        assert all_memories[0]["content"] == "The user prefers concise responses"
        assert all_memories[0]["agent"] == "assistant"
        assert all_memories[0]["importance"] == 0.8
        
        memory.finalize()
    
    def test_search(self):
        """Test memory search."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        # Save multiple memories
        memory.save(content="User likes Python programming", agent="coder", tags=["preference"])
        memory.save(content="User prefers dark mode UI", agent="designer", tags=["preference"])
        memory.save(content="Project deadline is next Friday", agent="manager", tags=["schedule"])
        
        # Search by keyword
        results = memory.search("Python", limit=5)
        assert len(results) == 1
        assert "Python" in results[0]["content"]
        
        # Search with no matches
        results = memory.search("nonexistent", limit=5)
        assert len(results) == 0
        
        memory.finalize()
    
    def test_get_by_agent(self):
        """Test getting memories by agent."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        memory.save(content="Memory 1", agent="researcher")
        memory.save(content="Memory 2", agent="researcher")
        memory.save(content="Memory 3", agent="writer")
        
        researcher_memories = memory.get_by_agent("researcher")
        assert len(researcher_memories) == 2
        
        writer_memories = memory.get_by_agent("writer")
        assert len(writer_memories) == 1
        
        memory.finalize()
    
    def test_get_by_tags(self):
        """Test getting memories by tags."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        memory.save(content="Important finding", tags=["research", "important"])
        memory.save(content="Normal note", tags=["research"])
        memory.save(content="Action item", tags=["todo", "important"])
        
        # Any tag match
        research_mems = memory.get_by_tags(["research"])
        assert len(research_mems) == 2
        
        important_mems = memory.get_by_tags(["important"])
        assert len(important_mems) == 2
        
        # Match all tags
        both_tags = memory.get_by_tags(["research", "important"], match_all=True)
        assert len(both_tags) == 1
        assert "Important finding" in both_tags[0]["content"]
        
        memory.finalize()
    
    def test_get_recent(self):
        """Test getting recent memories."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        for i in range(5):
            memory.save(content=f"Memory {i}", agent="test")
            time.sleep(0.01)  # Small delay for timestamp ordering
        
        recent = memory.get_recent(limit=3)
        assert len(recent) == 3
        
        # Most recent should be "Memory 4"
        assert "Memory 4" in recent[0]["content"]
        
        memory.finalize()
    
    def test_get_important(self):
        """Test getting important memories."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        memory.save(content="Low priority", importance=0.2)
        memory.save(content="Medium priority", importance=0.5)
        memory.save(content="High priority", importance=0.9)
        memory.save(content="Critical", importance=1.0)
        
        important = memory.get_important(min_importance=0.7)
        assert len(important) == 2
        
        # Should be sorted by importance (highest first)
        assert important[0]["importance"] == 1.0
        assert important[1]["importance"] == 0.9
        
        memory.finalize()
    
    def test_update_importance(self):
        """Test updating memory importance."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        memory_id = memory.save(content="Test memory", importance=0.5)
        
        # Update importance
        result = memory.update_importance(memory_id, 0.9)
        assert result is True
        
        # Verify update
        all_mems = memory.get_all()
        assert all_mems[0]["importance"] == 0.9
        
        # Try updating non-existent memory
        result = memory.update_importance("fake_id", 0.5)
        assert result is False
        
        memory.finalize()
    
    def test_delete(self):
        """Test memory deletion."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        memory_id = memory.save(content="To be deleted")
        assert len(memory) == 1
        
        result = memory.delete(memory_id)
        assert result is True
        assert len(memory) == 0
        
        # Try deleting non-existent memory
        result = memory.delete("fake_id")
        assert result is False
        
        memory.finalize()
    
    def test_clear_agent_memories(self):
        """Test clearing all memories for an agent."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        memory.save(content="Agent 1 memory 1", agent="agent1")
        memory.save(content="Agent 1 memory 2", agent="agent1")
        memory.save(content="Agent 2 memory", agent="agent2")
        
        cleared = memory.clear_agent_memories("agent1")
        assert cleared == 2
        assert len(memory) == 1
        assert memory.get_all()[0]["agent"] == "agent2"
        
        memory.finalize()
    
    def test_count(self):
        """Test memory count with filters."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        memory.save(content="M1", agent="a1", tags=["t1"])
        memory.save(content="M2", agent="a1", tags=["t2"])
        memory.save(content="M3", agent="a2", tags=["t1"])
        
        assert memory.count() == 3
        assert memory.count(agent="a1") == 2
        assert memory.count(tags=["t1"]) == 2
        assert memory.count(agent="a1", tags=["t1"]) == 1
        
        memory.finalize()
    
    def test_context_manager(self):
        """Test using memory as context manager."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        with MAIFCrewMemory(self.artifact_path) as memory:
            memory.save(content="Context manager test")
        
        # Should be finalized after exit
        assert os.path.exists(self.artifact_path)
    
    def test_artifact_creation(self):
        """Test that MAIF artifact is created correctly."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory(self.artifact_path)
        
        memory.save(content="Test memory 1", agent="test", tags=["important"])
        memory.save(content="Test memory 2", agent="test", importance=0.9)
        
        memory.finalize()
        
        # Verify artifact exists and can be read
        assert os.path.exists(self.artifact_path)
        
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        # Verify integrity
        is_valid, errors = decoder.verify_integrity()
        assert is_valid, f"Integrity check failed: {errors}"
    
    def test_persistence(self):
        """Test memory persistence across instances."""
        from maif.integrations.crewai import MAIFCrewMemory
        
        # Create and populate memory
        memory1 = MAIFCrewMemory(self.artifact_path, auto_finalize=False)
        memory1.save(content="Persistent memory", agent="test", tags=["persist"])
        memory1.finalize()
        
        # Create new instance from same artifact
        memory2 = MAIFCrewMemory(self.artifact_path)
        
        # Should have loaded the existing memory
        assert len(memory2) >= 1
        
        # Search for the memory
        results = memory2.search("Persistent")
        assert len(results) >= 1
        
        memory2.finalize()


class TestCrewAISerializationEdgeCases:
    """Test edge cases in serialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "edge_cases.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_long_content_truncation(self):
        """Test that very long content is truncated."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        # Create task output with very long content
        long_content = "x" * 100000
        task_output = MockTaskOutput(
            description="Task with long output",
            raw=long_content,
        )
        
        callback.on_task_complete(task_output)
        callback.finalize()
        
        # Should not raise and artifact should be valid
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        is_valid, errors = decoder.verify_integrity()
        assert is_valid
    
    def test_special_characters(self):
        """Test handling of special characters."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        task_output = MockTaskOutput(
            description="Task with émojis 🚀 and spëcial chars: <>&\"'",
            raw="Output: τέστ αβγ\n\t\r",
        )
        
        callback.on_task_complete(task_output)
        callback.finalize()
        
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        is_valid, errors = decoder.verify_integrity()
        assert is_valid
    
    def test_none_values(self):
        """Test handling of None values in outputs."""
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        
        # Create output with None values
        task_output = MockTaskOutput(
            description=None,  # type: ignore
            raw=None,  # type: ignore
            agent=None,  # type: ignore
        )
        
        # Should not raise
        callback.on_task_complete(task_output)
        callback.finalize()


@pytest.mark.integration
class TestMAIFCrewAIIntegration:
    """Integration tests requiring CrewAI and optionally Gemini API."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "integration.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_crewai_available(self):
        """Test that CrewAI is available for integration tests."""
        try:
            import crewai
            assert crewai is not None
        except ImportError:
            pytest.skip("CrewAI not installed")
    
    def test_with_mock_crew(self, gemini_api_key):
        """Test callback integration with mock crew workflow."""
        try:
            from crewai import Agent, Task, Crew
        except ImportError:
            pytest.skip("CrewAI not installed")
        
        from maif.integrations.crewai import MAIFCrewCallback
        
        # Create callback
        callback = MAIFCrewCallback(self.artifact_path)
        
        # This test would use real CrewAI objects with a mock LLM
        # For now, we just verify the callback can be instantiated
        # and used with the expected interface
        
        callback.on_crew_start(crew_name="Integration Test Crew")
        callback.on_task_complete(MockTaskOutput())
        callback.on_step(MockStepOutput())
        callback.on_crew_end(result=MockCrewOutput())
        
        callback.finalize()
        
        # Verify artifact
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        is_valid, errors = decoder.verify_integrity()
        assert is_valid, f"Integrity check failed: {errors}"
    
    def test_real_crew_with_gemini(self, gemini_api_key):
        """Test with a real CrewAI crew using Gemini LLM.
        
        This test requires GEMINI_API_KEY environment variable.
        It runs a simple crew and verifies MAIF provenance is captured.
        """
        try:
            from crewai import Agent, Task, Crew, LLM
        except ImportError:
            pytest.skip("CrewAI not installed")
        
        from maif.integrations.crewai import MAIFCrewCallback
        
        # Set API key in environment for CrewAI's native handling
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Create Gemini LLM using CrewAI's native format
        try:
            llm = LLM(
                model="gemini/gemini-2.0-flash",
                api_key=gemini_api_key,
                temperature=0.1,
            )
        except Exception as e:
            pytest.skip(f"Failed to create Gemini LLM: {e}")
        
        # Create a simple agent
        analyst = Agent(
            role="Calculator",
            goal="Answer simple math questions accurately",
            backstory="You are a calculator that gives brief, accurate answers.",
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        
        # Create a simple task
        analysis_task = Task(
            description="What is 2+2? Answer with just the number.",
            expected_output="The number 4",
            agent=analyst,
        )
        
        # Create MAIF callback
        callback = MAIFCrewCallback(self.artifact_path)
        
        # Create and run the crew
        crew = Crew(
            agents=[analyst],
            tasks=[analysis_task],
            task_callback=callback.on_task_complete,
            step_callback=callback.on_step,
            verbose=False,
        )
        
        # Log crew start
        callback.on_crew_start(crew_name="Gemini Test Crew", agents=[analyst], tasks=[analysis_task])
        
        try:
            # Run the crew
            result = crew.kickoff()
            
            # Log crew end
            callback.on_crew_end(result=result)
        except Exception as e:
            # Log error but still verify we captured some events
            callback.on_crew_end(error=e)
            pytest.skip(f"Gemini API error (may be quota/access issue): {e}")
        finally:
            callback.finalize()
        
        # Verify artifact was created and is valid
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        is_valid, errors = decoder.verify_integrity()
        assert is_valid, f"Integrity check failed: {errors}"
        
        # Verify we captured events
        assert len(decoder.blocks) >= 4, "Expected at least 4 blocks (session_start, agent_start, task_end, session_end)"
        
        # Verify event types
        event_types = set()
        for block in decoder.blocks:
            if block.metadata:
                event_types.add(block.metadata.get("type", ""))
        
        assert "task_end" in event_types, "Expected task_end event"
        
        # Verify statistics were captured
        stats = callback.get_statistics()
        assert stats["tasks_completed"] >= 1, "Expected at least 1 task completed"


class TestInstrumentFunction:
    """Tests for the instrument() one-liner function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "instrument_test.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_instrument_import(self):
        """Test instrument function can be imported."""
        from maif.integrations.crewai import instrument
        assert callable(instrument)
    
    def test_instrument_mock_crew(self):
        """Test instrument() with a mock crew."""
        from maif.integrations.crewai import instrument
        
        # Create a mock crew
        class MockCrew:
            name = "Test Crew"
            agents = []
            tasks = []
            task_callback = None
            step_callback = None
            
            def kickoff(self, *args, **kwargs):
                # Simulate callbacks being called
                if self.step_callback:
                    self.step_callback(MockStepOutput())
                if self.task_callback:
                    self.task_callback(MockTaskOutput())
                return "result"
        
        crew = MockCrew()
        crew = instrument(crew, self.artifact_path)
        
        # Verify callbacks were set
        assert crew.task_callback is not None
        assert crew.step_callback is not None
        
        # Run kickoff
        result = crew.kickoff()
        assert result == "result"
        
        # Verify artifact was created
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        is_valid, _ = decoder.verify_integrity()
        assert is_valid
        assert len(decoder.blocks) >= 4
    
    def test_instrument_auto_finalize(self):
        """Test instrument() auto-finalizes by default."""
        from maif.integrations.crewai import instrument
        
        class MockCrew:
            name = "Auto Finalize Test"
            agents = []
            tasks = []
            task_callback = None
            step_callback = None
            
            def kickoff(self, *args, **kwargs):
                return "done"
        
        crew = MockCrew()
        crew = instrument(crew, self.artifact_path, auto_finalize=True)
        crew.kickoff()
        
        # Check artifact is finalized (can be read)
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        is_valid, _ = decoder.verify_integrity()
        assert is_valid
    
    def test_instrument_no_auto_finalize(self):
        """Test instrument() with auto_finalize=False."""
        from maif.integrations.crewai import instrument
        
        class MockCrew:
            name = "Manual Finalize Test"
            agents = []
            tasks = []
            task_callback = None
            step_callback = None
            
            def kickoff(self, *args, **kwargs):
                return "done"
        
        crew = MockCrew()
        crew = instrument(crew, self.artifact_path, auto_finalize=False)
        crew.kickoff()
        
        # Manually finalize
        crew._maif_callback.finalize()
        
        # Check artifact
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        is_valid, _ = decoder.verify_integrity()
        assert is_valid
    
    def test_instrument_with_error(self):
        """Test instrument() handles errors in kickoff."""
        from maif.integrations.crewai import instrument
        
        class MockCrew:
            name = "Error Test"
            agents = []
            tasks = []
            task_callback = None
            step_callback = None
            
            def kickoff(self, *args, **kwargs):
                raise ValueError("Simulated crew error")
        
        crew = MockCrew()
        crew = instrument(crew, self.artifact_path)
        
        # Kickoff should raise but artifact should still be created
        with pytest.raises(ValueError):
            crew.kickoff()
        
        # Artifact should exist with error logged
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        is_valid, _ = decoder.verify_integrity()
        assert is_valid


class TestMAIFCrewPatternsUnit:
    """Unit tests for CrewAI pattern utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "pattern_test.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_context_manager_basic(self):
        """Test MAIFCrew context manager."""
        from maif.integrations.crewai import MAIFCrew
        
        with MAIFCrew(self.artifact_path) as callback:
            assert callback is not None
            callback.on_crew_start(crew_name="Context Manager Test")
            callback.on_step(MockStepOutput())
            callback.on_task_complete(MockTaskOutput())
            callback.on_crew_end(result=None)
        
        # Verify artifact was finalized
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        is_valid, _ = decoder.verify_integrity()
        assert is_valid
        assert len(decoder.blocks) >= 4
    
    def test_context_manager_with_error(self):
        """Test context manager handles errors gracefully."""
        from maif.integrations.crewai import MAIFCrew
        
        try:
            with MAIFCrew(self.artifact_path) as callback:
                callback.on_crew_start(crew_name="Error Test")
                raise ValueError("Simulated error")
        except ValueError:
            pass
        
        # Artifact should still be finalized
        assert os.path.exists(self.artifact_path)
    
    def test_finalize_crew_utility(self):
        """Test finalize_crew utility function."""
        from maif.integrations.crewai import (
            MAIFCrewCallback,
            finalize_crew,
            get_artifact_path,
            get_crew_statistics,
        )
        
        # Create a mock crew-like object
        class MockCrew:
            pass
        
        crew = MockCrew()
        callback = MAIFCrewCallback(self.artifact_path)
        callback.on_crew_start(crew_name="Utility Test")
        callback.on_step(MockStepOutput())
        
        crew._maif_callback = callback
        
        # Test get_artifact_path
        assert get_artifact_path(crew) == self.artifact_path
        
        # Test get_crew_statistics
        stats = get_crew_statistics(crew)
        assert "steps_executed" in stats
        assert stats["steps_executed"] == 1
        
        # Test finalize_crew
        finalize_crew(crew)
        
        # Verify artifact was finalized
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        is_valid, _ = decoder.verify_integrity()
        assert is_valid
    
    def test_pattern_imports(self):
        """Test that pattern functions can be imported."""
        from maif.integrations.crewai.patterns import (
            create_research_crew,
            create_qa_crew,
            create_code_review_crew,
            MAIFCrew,
            instrument,
        )
        
        assert callable(create_research_crew)
        assert callable(create_qa_crew)
        assert callable(create_code_review_crew)
        assert callable(instrument)
        assert MAIFCrew is not None
    
    def test_instrument_import(self):
        """Test that instrument can be imported from main module."""
        from maif.integrations.crewai import instrument
        assert callable(instrument)


class TestMAIFCrewCLI:
    """Unit tests for CrewAI CLI."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "cli_test.maif")
        
        # Create a test artifact
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback(self.artifact_path)
        callback.on_crew_start(crew_name="CLI Test Crew")
        callback.on_step(MockStepOutput())
        callback.on_task_complete(MockTaskOutput())
        callback.on_crew_end(result=None)
        callback.finalize()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_import(self):
        """Test CLI module can be imported."""
        from maif.integrations.crewai import cli
        assert hasattr(cli, "main")
        assert hasattr(cli, "cmd_inspect")
        assert hasattr(cli, "cmd_verify")
        assert hasattr(cli, "cmd_export")
    
    def test_cmd_inspect_function(self):
        """Test inspect command function directly."""
        from maif.integrations.crewai.cli import cmd_inspect
        
        # Create args object
        class Args:
            artifact = None
            tasks = False
            steps = False
            limit = 10
        
        args = Args()
        args.artifact = self.artifact_path
        
        # Should not raise
        cmd_inspect(args)
    
    def test_cmd_verify_function(self):
        """Test verify command function directly."""
        from maif.integrations.crewai.cli import cmd_verify
        import sys
        
        class Args:
            artifact = None
            verbose = False
        
        args = Args()
        args.artifact = self.artifact_path
        
        # Capture exit
        original_exit = sys.exit
        exit_code = [None]
        
        def mock_exit(code):
            exit_code[0] = code
        
        sys.exit = mock_exit
        try:
            cmd_verify(args)
        finally:
            sys.exit = original_exit
        
        # Should exit with 0 (success)
        assert exit_code[0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

