"""
Event Sourcing for MAIF
=======================

Implements append-only event log with materialized views for queries.
This module provides the core event sourcing capabilities for MAIF,
allowing for complete history tracking and state reconstruction.
"""

import time
import json
import uuid
import hashlib
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Tuple,
    Set,
    Union,
    Callable,
    TypeVar,
    Generic,
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
from pathlib import Path
import pickle
import logging

from .security import MAIFSigner, ProvenanceEntry
from .compression_manager import CompressionManager

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar("T")
EventHandler = Callable[[Dict[str, Any], Dict[str, Any]], None]


class EventType(Enum):
    """Types of events in the event log."""

    BLOCK_CREATED = "block_created"
    BLOCK_UPDATED = "block_updated"
    BLOCK_DELETED = "block_deleted"
    MAIF_CREATED = "maif_created"
    MAIF_UPDATED = "maif_updated"
    MAIF_MERGED = "maif_merged"
    MAIF_SPLIT = "maif_split"
    SCHEMA_UPDATED = "schema_updated"
    ACCESS_GRANTED = "access_granted"
    ACCESS_REVOKED = "access_revoked"
    QUERY_EXECUTED = "query_executed"
    CUSTOM = "custom"


@dataclass
class Event:
    """Represents a single event in the event log."""

    event_id: str
    event_type: EventType
    timestamp: float
    agent_id: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_event_id: Optional[str] = None
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "payload": self.payload,
            "metadata": self.metadata,
            "previous_event_id": self.previous_event_id,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            agent_id=data["agent_id"],
            payload=data["payload"],
            metadata=data.get("metadata", {}),
            previous_event_id=data.get("previous_event_id"),
            signature=data.get("signature"),
        )


class EventLog:
    """
    Append-only event log for MAIF.

    Provides a secure, immutable record of all operations performed on MAIF files.
    """

    def __init__(
        self, log_path: Optional[str] = None, signer: Optional[MAIFSigner] = None
    ):
        self.log_path = Path(log_path) if log_path else None
        self.signer = signer
        self.events: List[Event] = []
        self.event_index: Dict[str, int] = {}  # event_id -> position
        self.last_event_id: Optional[str] = None
        self._lock = threading.RLock()
        self.compressor = CompressionManager()

        # Load existing log if path provided
        if self.log_path and self.log_path.exists():
            self._load_log()

    def append_event(
        self,
        event_type: EventType,
        agent_id: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Append a new event to the log.

        Args:
            event_type: Type of event
            agent_id: ID of agent creating the event
            payload: Event data
            metadata: Additional metadata

        Returns:
            Event ID
        """
        with self._lock:
            event_id = str(uuid.uuid4())
            timestamp = time.time()

            event = Event(
                event_id=event_id,
                event_type=event_type,
                timestamp=timestamp,
                agent_id=agent_id,
                payload=payload,
                metadata=metadata or {},
                previous_event_id=self.last_event_id,
            )

            # Sign event if signer available
            if self.signer:
                event_data = json.dumps(
                    {k: v for k, v in event.to_dict().items() if k != "signature"},
                    sort_keys=True,
                ).encode()
                event.signature = self.signer.sign_data(event_data)

            # Add to in-memory log
            self.events.append(event)
            self.event_index[event_id] = len(self.events) - 1
            self.last_event_id = event_id

            # Persist to disk if path provided
            if self.log_path:
                self._append_to_file(event)

            return event_id

    def get_event(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        with self._lock:
            if event_id in self.event_index:
                return self.events[self.event_index[event_id]]
            return None

    def get_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """
        Get events matching criteria.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            event_types: Filter by event types
            agent_id: Filter by agent ID
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        with self._lock:
            result = []

            for event in self.events:
                # Apply filters
                if start_time is not None and event.timestamp < start_time:
                    continue
                if end_time is not None and event.timestamp > end_time:
                    continue
                if event_types is not None and event.event_type not in event_types:
                    continue
                if agent_id is not None and event.agent_id != agent_id:
                    continue

                result.append(event)

                if limit is not None and len(result) >= limit:
                    break

            return result

    def get_event_chain(self, event_id: str) -> List[Event]:
        """Get chain of events starting from given event ID."""
        with self._lock:
            result = []
            current_id = event_id

            while current_id:
                event = self.get_event(current_id)
                if not event:
                    break

                result.append(event)
                current_id = event.previous_event_id

            return list(reversed(result))  # Return in chronological order

    def _load_log(self):
        """Load event log from file."""
        try:
            with open(self.log_path, "r") as f:
                for i, line in enumerate(f):
                    try:
                        event_data = json.loads(line.strip())
                        event = Event.from_dict(event_data)
                        self.events.append(event)
                        self.event_index[event.event_id] = i
                        self.last_event_id = event.event_id
                    except Exception as e:
                        logger.error(f"Error loading event: {e}")

            logger.info(f"Loaded {len(self.events)} events from {self.log_path}")
        except Exception as e:
            logger.error(f"Error loading event log: {e}")

    def _append_to_file(self, event: Event):
        """Append event to log file."""
        try:
            event_json = json.dumps(event.to_dict())

            # Create directory if it doesn't exist
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.log_path, "a") as f:
                f.write(event_json + "\n")
        except Exception as e:
            logger.error(f"Error appending event to file: {e}")


class MaterializedView(Generic[T]):
    """
    Materialized view for efficient querying of event-sourced data.

    Maintains a derived data structure that is updated based on events.
    """

    def __init__(self, name: str, event_log: EventLog):
        self.name = name
        self.event_log = event_log
        self.state: Dict[str, T] = {}
        self.handlers: Dict[EventType, EventHandler] = {}
        self.last_processed_event_id: Optional[str] = None
        self._lock = threading.RLock()

    def register_handler(self, event_type: EventType, handler: EventHandler):
        """Register handler for event type."""
        self.handlers[event_type] = handler

    def update(self):
        """Update view with new events."""
        with self._lock:
            # Find events to process
            events_to_process = []

            if self.last_processed_event_id:
                # Get events after last processed
                last_index = self.event_log.event_index.get(
                    self.last_processed_event_id, -1
                )
                if last_index >= 0 and last_index + 1 < len(self.event_log.events):
                    events_to_process = self.event_log.events[last_index + 1 :]
            else:
                # Process all events
                events_to_process = self.event_log.events

            # Process events
            for event in events_to_process:
                self._process_event(event)
                self.last_processed_event_id = event.event_id

    def _process_event(self, event: Event):
        """Process a single event."""
        if event.event_type in self.handlers:
            try:
                self.handlers[event.event_type](event.payload, self.state)
            except Exception as e:
                logger.error(f"Error processing event {event.event_id}: {e}")

    def get_state(self) -> Dict[str, T]:
        """Get current state of the view."""
        with self._lock:
            return self.state.copy()

    def get_item(self, key: str) -> Optional[T]:
        """Get item by key."""
        with self._lock:
            return self.state.get(key)


class EventSourcedMAIF:
    """
    Event-sourced MAIF implementation.

    Provides event sourcing capabilities for MAIF, maintaining a complete
    history of all operations and allowing for state reconstruction.
    """

    def __init__(self, maif_id: str, event_log: EventLog, agent_id: str):
        self.maif_id = maif_id
        self.event_log = event_log
        self.agent_id = agent_id
        self.views: Dict[str, MaterializedView] = {}

        # Create standard views
        self._create_standard_views()

    def _create_standard_views(self):
        """Create standard materialized views."""
        # Blocks view
        blocks_view = MaterializedView[Dict[str, Any]](
            name="blocks", event_log=self.event_log
        )

        # Register handlers
        blocks_view.register_handler(
            EventType.BLOCK_CREATED,
            lambda payload, state: state.update({payload["block_id"]: payload}),
        )

        blocks_view.register_handler(
            EventType.BLOCK_UPDATED,
            lambda payload, state: state.update(
                {payload["block_id"]: {**state.get(payload["block_id"], {}), **payload}}
            ),
        )

        blocks_view.register_handler(
            EventType.BLOCK_DELETED,
            lambda payload, state: state.pop(payload["block_id"], None),
        )

        self.views["blocks"] = blocks_view

        # Metadata view
        metadata_view = MaterializedView[Dict[str, Any]](
            name="metadata", event_log=self.event_log
        )

        metadata_view.register_handler(
            EventType.MAIF_CREATED,
            lambda payload, state: state.update({"maif_info": payload}),
        )

        metadata_view.register_handler(
            EventType.MAIF_UPDATED,
            lambda payload, state: state.update(
                {"maif_info": {**state.get("maif_info", {}), **payload}}
            ),
        )

        self.views["metadata"] = metadata_view

    def add_block(
        self,
        block_id: str,
        block_type: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a new block to the MAIF.

        Args:
            block_id: Block ID
            block_type: Block type
            data: Block data
            metadata: Block metadata

        Returns:
            Event ID
        """
        payload = {
            "block_id": block_id,
            "block_type": block_type,
            "data_hash": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "metadata": metadata or {},
        }

        event_id = self.event_log.append_event(
            event_type=EventType.BLOCK_CREATED, agent_id=self.agent_id, payload=payload
        )

        # Update views
        for view in self.views.values():
            view.update()

        return event_id

    def update_block(
        self,
        block_id: str,
        data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Update an existing block using copy-on-write semantics.

        Args:
            block_id: Block ID
            data: New block data (if changed)
            metadata: New block metadata (if changed)

        Returns:
            Event ID
        """
        # Copy-on-write: Check if block exists and if data has actually changed
        current_block = self.get_block(block_id)
        if current_block is None:
            # Block doesn't exist, treat as an add operation
            if data is not None:
                return self.add_block(block_id, "unknown", data, metadata)
            else:
                raise ValueError(
                    f"Block {block_id} not found and no data provided for creation"
                )

        # Check if anything has changed
        if data is None and metadata is None:
            # No changes, return the existing block ID
            return block_id

        # Check if only data has changed
        if data is not None:
            new_data_hash = hashlib.sha256(data).hexdigest()
            if new_data_hash == current_block.get("data_hash") and metadata is None:
                # Data hasn't changed, return the existing block ID
                return block_id

        # Something has changed, create a new event
        payload = {"block_id": block_id}

        if data is not None:
            payload["data_hash"] = hashlib.sha256(data).hexdigest()
            payload["size"] = len(data)

        if metadata is not None:
            payload["metadata"] = metadata

        event_id = self.event_log.append_event(
            event_type=EventType.BLOCK_UPDATED, agent_id=self.agent_id, payload=payload
        )

        # Update views
        for view in self.views.values():
            view.update()

        return event_id

    def delete_block(self, block_id: str) -> str:
        """
        Delete a block.

        Args:
            block_id: Block ID

        Returns:
            Event ID
        """
        payload = {"block_id": block_id}

        event_id = self.event_log.append_event(
            event_type=EventType.BLOCK_DELETED, agent_id=self.agent_id, payload=payload
        )

        # Update views
        for view in self.views.values():
            view.update()

        return event_id

    def get_blocks(self) -> Dict[str, Dict[str, Any]]:
        """Get all blocks in the MAIF."""
        self.views["blocks"].update()
        return self.views["blocks"].get_state()

    def get_block(self, block_id: str) -> Optional[Dict[str, Any]]:
        """Get block by ID."""
        self.views["blocks"].update()
        return self.views["blocks"].get_item(block_id)

    def get_metadata(self) -> Dict[str, Any]:
        """Get MAIF metadata."""
        self.views["metadata"].update()
        return self.views["metadata"].get_state().get("maif_info", {})

    def get_history(self, block_id: Optional[str] = None) -> List[Event]:
        """
        Get history of events.

        Args:
            block_id: If provided, get history for specific block

        Returns:
            List of events
        """
        if block_id:
            # Get events for specific block
            return self.event_log.get_events(
                event_types=[
                    EventType.BLOCK_CREATED,
                    EventType.BLOCK_UPDATED,
                    EventType.BLOCK_DELETED,
                ],
                agent_id=None,
            )
        else:
            # Get all events
            return self.event_log.get_events()

    def get_events(
        self,
        event_types: Optional[List[EventType]] = None,
        agent_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """
        Get events matching criteria.

        Args:
            event_types: Filter by event types
            agent_id: Filter by agent ID
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        return self.event_log.get_events(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            agent_id=agent_id,
            limit=limit,
        )

    def create_custom_view(self, name: str) -> MaterializedView:
        """
        Create a custom materialized view.

        Args:
            name: View name

        Returns:
            New materialized view
        """
        view = MaterializedView(name=name, event_log=self.event_log)
        self.views[name] = view
        return view

    def replay_events(
        self, target_timestamp: Optional[float] = None
    ) -> Dict[str, MaterializedView]:
        """
        Replay events to reconstruct state at a point in time.

        Args:
            target_timestamp: Target timestamp (None for current)

        Returns:
            Dictionary of views
        """
        # Create temporary views
        temp_views = {}
        for name, view in self.views.items():
            temp_view = MaterializedView(name=name, event_log=self.event_log)

            # Copy handlers
            for event_type, handler in view.handlers.items():
                temp_view.register_handler(event_type, handler)

            temp_views[name] = temp_view

        # Get events up to target timestamp
        events = self.event_log.get_events(end_time=target_timestamp)

        # Process events
        for event in events:
            for view in temp_views.values():
                view._process_event(event)

        return temp_views
