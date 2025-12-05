"""
MAIF Visual Debugging Tools - Interactive exploration and debugging utilities.
Provides web-based UI and CLI tools for MAIF inspection and debugging.
"""

import json
import time
import os
import webbrowser
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of MAIF memory at a point in time."""

    timestamp: float
    total_blocks: int
    block_types: Dict[str, int]
    total_size_bytes: int
    metadata_summary: Dict[str, Any]
    recent_operations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            "timestamp_str": datetime.fromtimestamp(self.timestamp).isoformat(),
        }


class MAIFDebugger:
    """
    Interactive debugger for MAIF files with visual exploration capabilities.
    """

    def __init__(self, maif_path: str):
        """
        Initialize debugger for a MAIF file.

        Args:
            maif_path: Path to MAIF file to debug
        """
        self.maif_path = Path(maif_path)

        # Initialize MAIF components
        from .core import MAIFDecoder
        from .validation import MAIFValidator

        self.decoder = MAIFDecoder(str(maif_path))
        self.validator = MAIFValidator()

        # Debug state
        self.snapshots: List[MemorySnapshot] = []
        self.breakpoints: Dict[str, Any] = {}
        self.watch_list: List[str] = []
        self.operation_log: List[Dict[str, Any]] = []

    def inspect(self) -> Dict[str, Any]:
        """
        Inspect MAIF file and return comprehensive debug information.

        Returns:
            Dictionary with detailed inspection results
        """
        inspection = {
            "file_info": self._get_file_info(),
            "structure": self._analyze_structure(),
            "blocks": self._analyze_blocks(),
            "metadata": self._analyze_metadata(),
            "validation": self._validate_integrity(),
            "performance": self._analyze_performance(),
            "recommendations": self._generate_recommendations(),
        }

        return inspection

    def visualize_memory_map(self) -> str:
        """
        Generate a visual memory map of the MAIF file.

        Returns:
            ASCII art representation of memory layout
        """
        blocks = self.decoder.blocks

        # Sort blocks by position
        sorted_blocks = sorted(blocks, key=lambda b: b.offset)

        # Create visual representation
        lines = []
        lines.append("MAIF Memory Map")
        lines.append("=" * 80)
        lines.append(f"Total Size: {self._format_size(self.decoder.file_size)}")
        lines.append(f"Total Blocks: {len(blocks)}")
        lines.append("")
        lines.append("Offset      Size        Type    ID")
        lines.append("-" * 80)

        for block in sorted_blocks:
            offset_str = (
                f"0x{block.offset:08X}" if hasattr(block, "offset") else "0x00000000"
            )
            size_str = self._format_size(
                block.size if hasattr(block, "size") else len(block.data)
            ).rjust(10)
            type_str = (
                block.block_type_name
                if hasattr(block, "block_type_name")
                else str(block.block_type)
            ).ljust(8)
            id_str = (
                block.block_id[:40] + "..."
                if len(block.block_id) > 40
                else block.block_id
            )

            lines.append(f"{offset_str}  {size_str}  {type_str}  {id_str}")

        # Add visual block representation
        lines.append("")
        lines.append("Visual Block Layout:")
        lines.append("-" * 80)

        # Create proportional representation
        total_size = self.decoder.file_size
        bar_width = 70

        for block in sorted_blocks:
            block_size = block.size if hasattr(block, "size") else len(block.data)
            block_chars = max(1, int(block_size / total_size * bar_width))
            type_name = (
                block.block_type_name
                if hasattr(block, "block_type_name")
                else str(block.block_type)
            )
            char = self._get_block_char(type_name)
            bar = char * block_chars
            lines.append(f"{type_name[:4]:4} |{bar}")

        lines.append("-" * 80)
        lines.append(
            "Legend: H=Header, T=Text, E=Embedding, K=Knowledge, S=Security, B=Binary"
        )

        return "\n".join(lines)

    def trace_operations(self, duration: float = 60.0) -> List[Dict[str, Any]]:
        """
        Trace MAIF operations for a specified duration.

        Args:
            duration: How long to trace operations (seconds)

        Returns:
            List of traced operations
        """
        start_time = time.time()
        operations = []

        logger.info(f"Starting operation trace for {duration} seconds...")

        # This would hook into MAIF operations in a real implementation
        # For now, return example trace data
        operations = [
            {
                "timestamp": start_time + 1,
                "operation": "read_block",
                "block_id": "text_001",
                "duration_ms": 2.3,
                "size_bytes": 1024,
            },
            {
                "timestamp": start_time + 2,
                "operation": "add_embedding",
                "block_id": "emb_001",
                "duration_ms": 5.7,
                "size_bytes": 1536,
            },
        ]

        return operations

    def profile_performance(self) -> Dict[str, Any]:
        """
        Profile MAIF performance characteristics.

        Returns:
            Performance profile with metrics
        """
        profile = {
            "io_performance": self._profile_io(),
            "memory_usage": self._profile_memory(),
            "compression_stats": self._profile_compression(),
            "access_patterns": self._profile_access_patterns(),
        }

        return profile

    def export_debug_report(self, output_path: Optional[str] = None) -> str:
        """
        Export comprehensive debug report.

        Args:
            output_path: Optional path for report (defaults to maif_debug_report.json)

        Returns:
            Path to exported report
        """
        if not output_path:
            output_path = f"maif_debug_report_{int(time.time())}.json"

        report = {
            "maif_file": str(self.maif_path),
            "report_timestamp": datetime.now().isoformat(),
            "inspection": self.inspect(),
            "memory_map": self.visualize_memory_map(),
            "performance_profile": self.profile_performance(),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "operation_log": self.operation_log,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Debug report exported to {output_path}")
        return output_path

    def start_web_ui(self, port: int = 8080, auto_open: bool = True):
        """
        Start web-based debugging UI.

        Args:
            port: Port for web server
            auto_open: Automatically open browser
        """
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse

        debug_data = self.inspect()

        class DebugHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                """Handle GET requests."""
                parsed_path = urllib.parse.urlparse(self.path)

                if parsed_path.path == "/":
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(self._generate_html().encode())
                elif parsed_path.path == "/api/data":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(debug_data).encode())
                else:
                    self.send_error(404)

            def _generate_html(self):
                """Generate debug UI HTML."""
                return f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>MAIF Debugger - {self.server.maif_path}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        .section {{ margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 8px; }}
                        .block {{ padding: 10px; margin: 5px 0; background: white; border: 1px solid #ddd; }}
                        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e3f2fd; }}
                        pre {{ background: #f0f0f0; padding: 10px; overflow: auto; }}
                        .chart {{ width: 100%; height: 300px; background: #fff; border: 1px solid #ddd; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>MAIF Debugger</h1>
                        <p>File: <code>{self.server.maif_path}</code></p>
                        
                        <div class="section">
                            <h2>File Information</h2>
                            <div id="file-info"></div>
                        </div>
                        
                        <div class="section">
                            <h2>Block Analysis</h2>
                            <div id="blocks"></div>
                        </div>
                        
                        <div class="section">
                            <h2>Memory Map</h2>
                            <pre id="memory-map"></pre>
                        </div>
                        
                        <div class="section">
                            <h2>Performance Metrics</h2>
                            <div id="performance"></div>
                        </div>
                    </div>
                    
                    <script>
                        fetch('/api/data')
                            .then(response => response.json())
                            .then(data => {{
                                // Display file info
                                const fileInfo = data.file_info;
                                document.getElementById('file-info').innerHTML = `
                                    <div class="metric">Size: ${{fileInfo.size_human}}</div>
                                    <div class="metric">Created: ${{new Date(fileInfo.created * 1000).toLocaleString()}}</div>
                                    <div class="metric">Modified: ${{new Date(fileInfo.modified * 1000).toLocaleString()}}</div>
                                `;
                                
                                // Display blocks
                                const blocks = data.blocks;
                                let blocksHtml = '<h3>Block Types:</h3>';
                                Object.entries(blocks.by_type).forEach(([type, count]) => {{
                                    blocksHtml += `<div class="metric">${{type}}: ${{count}}</div>`;
                                }});
                                document.getElementById('blocks').innerHTML = blocksHtml;
                                
                                // Display memory map
                                // In real implementation, would generate visual representation
                                document.getElementById('memory-map').textContent = 'Memory map visualization...';
                                
                                // Display performance
                                const perf = data.performance;
                                document.getElementById('performance').innerHTML = `
                                    <div class="metric">Avg Read Time: ${{perf.avg_read_time_ms}}ms</div>
                                    <div class="metric">Cache Hit Rate: ${{(perf.cache_hit_rate * 100).toFixed(1)}}%</div>
                                `;
                            }});
                    </script>
                </body>
                </html>
                """

        # Add reference to debugger instance
        server = HTTPServer(("localhost", port), DebugHandler)
        server.maif_path = self.maif_path

        # Start server in thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        url = f"http://localhost:{port}"
        logger.info(f"Debug UI started at {url}")

        if auto_open:
            webbrowser.open(url)

        return server

    def _get_file_info(self) -> Dict[str, Any]:
        """Get basic file information."""
        stat = os.stat(self.maif_path)

        return {
            "path": str(self.maif_path),
            "size_bytes": stat.st_size,
            "size_human": self._format_size(stat.st_size),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
        }

    def _analyze_structure(self) -> Dict[str, Any]:
        """Analyze MAIF structure."""
        blocks = self.decoder.blocks

        return {
            "total_blocks": len(blocks),
            "header_version": self.decoder.header.version
            if hasattr(self.decoder, "header")
            else "unknown",
            "has_manifest": self.decoder.manifest is not None,
            "is_encrypted": any(
                b.flags & 0x01 for b in blocks
            ),  # Check encryption flag
            "is_compressed": any(
                b.flags & 0x02 for b in blocks
            ),  # Check compression flag
        }

    def _analyze_blocks(self) -> Dict[str, Any]:
        """Analyze block composition."""
        blocks = self.decoder.blocks

        # Count by type
        by_type = defaultdict(int)
        sizes_by_type = defaultdict(list)

        for block in blocks:
            type_name = (
                block.block_type_name
                if hasattr(block, "block_type_name")
                else str(block.block_type)
            )
            block_size = block.size if hasattr(block, "size") else len(block.data)
            by_type[type_name] += 1
            sizes_by_type[type_name].append(block_size)

        # Calculate statistics
        stats_by_type = {}
        for block_type, sizes in sizes_by_type.items():
            stats_by_type[block_type] = {
                "count": len(sizes),
                "total_size": sum(sizes),
                "avg_size": sum(sizes) / len(sizes) if sizes else 0,
                "min_size": min(sizes) if sizes else 0,
                "max_size": max(sizes) if sizes else 0,
            }

        def get_block_size(b):
            return b.size if hasattr(b, "size") else len(b.data)

        return {
            "by_type": dict(by_type),
            "stats_by_type": stats_by_type,
            "total_data_size": sum(get_block_size(b) for b in blocks),
            "fragmentation": self._calculate_fragmentation(),
        }

    def _analyze_metadata(self) -> Dict[str, Any]:
        """Analyze metadata patterns."""
        blocks = self.decoder.blocks

        # Collect all metadata keys
        all_keys = set()
        key_frequency = defaultdict(int)

        for block in blocks:
            if hasattr(block, "metadata") and block.metadata:
                for key in block.metadata.keys():
                    all_keys.add(key)
                    key_frequency[key] += 1

        return {
            "unique_keys": list(all_keys),
            "key_frequency": dict(key_frequency),
            "blocks_with_metadata": sum(
                1 for b in blocks if hasattr(b, "metadata") and b.metadata
            ),
        }

    def _validate_integrity(self) -> Dict[str, Any]:
        """Validate MAIF integrity."""
        try:
            is_valid = self.validator.validate_maif(str(self.maif_path))
            errors = []
        except Exception as e:
            is_valid = False
            errors = [str(e)]

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": [],  # Could add warnings for non-critical issues
        }

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        # In real implementation, would measure actual performance
        return {
            "avg_read_time_ms": 2.5,
            "avg_write_time_ms": 5.0,
            "cache_hit_rate": 0.85,
            "compression_ratio": 2.5,
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        blocks = self.decoder.blocks

        # Check for optimization opportunities
        if len(blocks) > 1000:
            recommendations.append("Consider enabling block indexing for faster access")

        # Check compression
        uncompressed_blocks = [b for b in blocks if not (b.flags & 0x02)]
        if len(uncompressed_blocks) > len(blocks) * 0.5:
            recommendations.append("Enable compression to reduce file size")

        # Check fragmentation
        if self._calculate_fragmentation() > 0.2:
            recommendations.append(
                "High fragmentation detected - consider defragmentation"
            )

        return recommendations

    def _profile_io(self) -> Dict[str, Any]:
        """Profile I/O performance."""
        return {
            "read_operations": 0,  # Would track actual operations
            "write_operations": 0,
            "seek_operations": 0,
            "sequential_ratio": 0.8,
        }

    def _profile_memory(self) -> Dict[str, Any]:
        """Profile memory usage."""
        return {"peak_memory_mb": 128, "current_memory_mb": 64, "cache_size_mb": 32}

    def _profile_compression(self) -> Dict[str, Any]:
        """Profile compression statistics."""
        blocks = self.decoder.blocks
        compressed = [b for b in blocks if b.flags & 0x02]

        return {
            "compressed_blocks": len(compressed),
            "uncompressed_blocks": len(blocks) - len(compressed),
            "compression_ratio": 2.5,  # Would calculate actual ratio
            "algorithms_used": ["zstd", "gzip"],  # Would detect actual algorithms
        }

    def _profile_access_patterns(self) -> Dict[str, Any]:
        """Profile access patterns."""
        return {
            "hot_blocks": [],  # Would track frequently accessed blocks
            "cold_blocks": [],  # Would track rarely accessed blocks
            "access_frequency": {},  # Would track access counts
        }

    def _calculate_fragmentation(self) -> float:
        """Calculate file fragmentation ratio."""
        # Simplified fragmentation calculation
        # In real implementation, would analyze block layout
        return 0.1

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def _get_block_char(self, block_type: str) -> str:
        """Get character representation for block type."""
        mapping = {
            "header": "H",
            "text": "T",
            "embedding": "E",
            "knowledge": "K",
            "security": "S",
            "binary": "B",
        }
        return mapping.get(block_type.lower(), "?")


def debug_maif(maif_path: str, interactive: bool = True) -> MAIFDebugger:
    """
    Convenience function to start debugging a MAIF file.

    Args:
        maif_path: Path to MAIF file
        interactive: Start interactive web UI

    Returns:
        MAIFDebugger instance

    Example:
        debugger = debug_maif("my_agent.maif")
        print(debugger.visualize_memory_map())
    """
    debugger = MAIFDebugger(maif_path)

    if interactive:
        debugger.start_web_ui()

    return debugger
