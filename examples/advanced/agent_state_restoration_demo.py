"""
Agent State Restoration Demo
============================

Demonstrates how to:
1. Create and run an agent
2. Dump its complete state on shutdown
3. Restore the agent from the dump and continue processing
"""

import asyncio
import os
from pathlib import Path
from maif.agentic_framework import MAIFAgent, AgentState
from maif_sdk.aws_backend import AWSConfig
from maif_sdk.types import SecurityLevel


class StatefulProcessingAgent(MAIFAgent):
    """Agent that maintains state across restarts."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_processed = getattr(self, 'items_processed', 0)
        self.processing_history = getattr(self, 'processing_history', [])
    
    async def run(self, items_to_process: int = 10):
        """Process items, maintaining state."""
        await self.initialize()
        
        print(f"Starting processing from item {self.items_processed + 1}")
        
        for i in range(self.items_processed, self.items_processed + items_to_process):
            # Simulate processing
            perception = await self.perceive(
                f"Data item {i+1}: Customer order #{1000+i}",
                "text"
            )
            
            reasoning = await self.reason([perception])
            
            # Track processing
            self.items_processed += 1
            self.processing_history.append({
                "item_number": i + 1,
                "timestamp": str(self.perception.perception_count),
                "status": "processed"
            })
            
            print(f"Processed item {i+1} (Total: {self.items_processed})")
            
            # Simulate interruption after 5 items
            if self.items_processed == 5 and not hasattr(self, '_restored'):
                print("\n[Simulating unexpected shutdown...]")
                break
            
            await asyncio.sleep(0.2)
        
        print(f"\nProcessing complete. Total items processed: {self.items_processed}")
    
    def dump_complete_state(self):
        """Override to include custom state in dump."""
        # Add custom state to config before dumping
        self.config['items_processed'] = self.items_processed
        self.config['processing_history'] = self.processing_history
        
        # Call parent dump method
        return super().dump_complete_state()
    
    def restore_state(self, dump_path):
        """Override to restore custom state."""
        # Call parent restore method
        super().restore_state(dump_path)
        
        # Restore custom state
        self.items_processed = self.config.get('items_processed', 0)
        self.processing_history = self.config.get('processing_history', [])
        self._restored = True


async def demonstrate_state_restoration():
    """Demonstrate agent state restoration."""
    
    print("=== Agent State Restoration Demo ===\n")
    
    # Set AWS credentials
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    aws_config = AWSConfig(
        region_name="us-east-1",
        s3_bucket=os.environ.get('MAIF_S3_BUCKET', 'maif-agent-states')
    )
    
    # Phase 1: Create and run agent
    print("Phase 1: Creating and running agent...")
    agent = StatefulProcessingAgent(
        agent_id="processor_001",
        workspace_path="./agent_workspace/stateful",
        config={"processing_batch_size": 5},
        use_aws=True,
        aws_config=aws_config
    )
    
    # Run agent (will stop after 5 items)
    await agent.run(items_to_process=10)
    
    # Dump state
    print("\nDumping agent state...")
    dump_path = agent.dump_complete_state()
    print(f"State dumped to: {dump_path}")
    
    # Shutdown agent
    agent.shutdown()
    
    print("\n" + "="*50 + "\n")
    
    # Phase 2: Restore agent from dump
    print("Phase 2: Restoring agent from dump...")
    
    # Method 1: Restore into existing agent instance
    restored_agent = StatefulProcessingAgent(
        agent_id="processor_001_restored",
        workspace_path="./agent_workspace/restored",
        use_aws=True,
        aws_config=aws_config,
        restore_from=dump_path  # Specify dump to restore from
    )
    
    print(f"Agent restored! Items previously processed: {restored_agent.items_processed}")
    print("Processing history:", restored_agent.processing_history[-3:] if len(restored_agent.processing_history) > 3 else restored_agent.processing_history)
    
    # Continue processing
    print("\nContinuing processing from where we left off...")
    await restored_agent.run(items_to_process=5)
    
    # Final state
    print(f"\nFinal state: {restored_agent.items_processed} items processed")
    restored_agent.shutdown()
    
    print("\n" + "="*50 + "\n")
    
    # Phase 3: Alternative restoration method
    print("Phase 3: Using from_dump class method...")
    
    # Method 2: Create agent directly from dump
    agent_from_dump = StatefulProcessingAgent.from_dump(
        dump_path=dump_path,
        workspace_path="./agent_workspace/from_dump",
        use_aws=True,
        aws_config=aws_config
    )
    
    print(f"Agent created from dump: {agent_from_dump.agent_id}")
    print(f"Previous work: {agent_from_dump.items_processed} items")
    
    # Process a few more items
    await agent_from_dump.run(items_to_process=2)
    
    agent_from_dump.shutdown()


async def demonstrate_aws_s3_restoration():
    """Demonstrate restoring agent from S3-stored dump."""
    
    print("\n=== S3-Based State Restoration Demo ===\n")
    
    aws_config = AWSConfig(
        region_name="us-east-1",
        s3_bucket="maif-agent-states"
    )
    
    # Create agent
    agent = StatefulProcessingAgent(
        agent_id="s3_processor",
        workspace_path="./agent_workspace/s3_test",
        use_aws=True,
        aws_config=aws_config
    )
    
    await agent.initialize()
    
    # Process some items
    await agent.run(items_to_process=3)
    
    # Dump to S3
    dump_path = agent.dump_complete_state()
    
    # Get S3 artifact ID
    artifact_id = dump_path.name  # When using AWS, this is the artifact ID
    s3_path = f"s3://{aws_config.s3_bucket}/artifacts/{artifact_id}"
    
    print(f"State saved to S3: {s3_path}")
    
    agent.shutdown()
    
    # Restore from S3
    print("\nRestoring from S3...")
    restored = StatefulProcessingAgent.from_dump(
        dump_path=artifact_id,  # Can use artifact ID directly
        use_aws=True,
        aws_config=aws_config
    )
    
    print(f"Restored agent: {restored.agent_id}")
    print(f"Items processed: {restored.items_processed}")


class CheckpointAgent(MAIFAgent):
    """Agent that creates periodic checkpoints."""
    
    def __init__(self, *args, checkpoint_interval: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint = 0
        self.checkpoints = []
    
    async def run(self):
        """Run with periodic checkpointing."""
        await self.initialize()
        
        for i in range(50):
            # Process item
            perception = await self.perceive(f"Item {i+1}", "text")
            
            # Create checkpoint periodically
            if (i + 1) % self.checkpoint_interval == 0:
                checkpoint_path = self.create_checkpoint(i + 1)
                self.checkpoints.append(checkpoint_path)
                print(f"Checkpoint created at item {i+1}: {checkpoint_path}")
            
            await asyncio.sleep(0.1)
        
        print(f"\nCreated {len(self.checkpoints)} checkpoints")
    
    def create_checkpoint(self, item_number: int):
        """Create a checkpoint at current state."""
        # Update config with checkpoint info
        original_config = self.config.copy()
        self.config['checkpoint_number'] = len(self.checkpoints) + 1
        self.config['checkpoint_at_item'] = item_number
        
        # Dump state
        checkpoint_path = self.dump_complete_state()
        
        # Restore original config
        self.config = original_config
        
        return checkpoint_path


async def demonstrate_checkpointing():
    """Demonstrate periodic checkpointing."""
    
    print("\n=== Checkpointing Demo ===\n")
    
    agent = CheckpointAgent(
        agent_id="checkpoint_agent",
        workspace_path="./agent_workspace/checkpoints",
        checkpoint_interval=10,
        use_aws=True
    )
    
    await agent.run()
    
    # Show how to restore from any checkpoint
    if agent.checkpoints:
        print(f"\nRestoring from checkpoint 2...")
        restored = CheckpointAgent.from_dump(
            dump_path=agent.checkpoints[1],  # Second checkpoint
            use_aws=True
        )
        
        config = restored.config
        print(f"Restored to checkpoint {config.get('checkpoint_number')} ")
        print(f"(created at item {config.get('checkpoint_at_item')})")
    
    agent.shutdown()


if __name__ == "__main__":
    # Run basic restoration demo
    asyncio.run(demonstrate_state_restoration())
    
    # Uncomment to run S3 restoration demo
    # asyncio.run(demonstrate_aws_s3_restoration())
    
    # Uncomment to run checkpointing demo
    # asyncio.run(demonstrate_checkpointing())