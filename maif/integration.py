"""
Integration utilities for MAIF format conversion and processing.
"""

import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ConversionResult:
    """Result of a conversion operation."""

    success: bool
    message: str = ""
    error: str = ""
    output_path: str = ""
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key):
        """Allow dict-style access for backward compatibility."""
        return getattr(self, key)


class MAIFConverter:
    """MAIF format converter."""

    def __init__(self):
        self.supported_formats = ["json", "xml", "csv", "txt"]

    def convert_to_maif(
        self, input_path: str, output_path: str, input_format: str
    ) -> ConversionResult:
        """Convert various formats to MAIF (v3 format - self-contained)."""
        try:
            from .core import MAIFEncoder

            encoder = MAIFEncoder(output_path, agent_id="format_converter")

            if input_format == "json":
                with open(input_path, "r") as f:
                    data = json.load(f)
                content = json.dumps(data, indent=2)
                encoder.add_text_block(content, metadata={"source_format": "json"})
            elif input_format == "xml":
                # Use the processor's XML conversion method
                processor = EnhancedMAIFProcessor()
                return processor.convert_xml_to_maif(input_path, output_path)
            else:
                with open(input_path, "r") as f:
                    content = f.read()
                encoder.add_text_block(
                    content, metadata={"source_format": input_format}
                )

            encoder.finalize()
            return ConversionResult(
                success=True,
                message=f"{input_format} converted to MAIF successfully",
                output_path=output_path,
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))

    def export_from_maif(
        self, maif_path: str, output_path: str, output_format: str
    ) -> ConversionResult:
        """Export MAIF to other formats (v3 format - self-contained)."""
        try:
            from .core import MAIFDecoder, BlockType

            decoder = MAIFDecoder(maif_path)
            decoder.load()

            if output_format == "json":
                # Export all text blocks to a JSON file
                content = []
                for block in decoder.blocks:
                    block_type = block.header.block_type
                    if block_type == BlockType.TEXT:
                        # Try to parse as JSON if possible
                        try:
                            block_content = (
                                json.loads(block.data.decode("utf-8"))
                                if block.data
                                else ""
                            )
                        except:
                            block_content = (
                                block.data.decode("utf-8") if block.data else ""
                            )

                        content.append(
                            {
                                "id": block.header.block_id,
                                "type": block_type.name,
                                "content": block_content,
                                "metadata": block.metadata,
                            }
                        )

                with open(output_path, "w") as f:
                    json.dump(content, f, indent=2)

                return ConversionResult(
                    success=True,
                    message=f"MAIF exported to {output_format} successfully",
                    output_path=output_path,
                )
            else:
                return ConversionResult(
                    success=False, error=f"Unsupported export format: {output_format}"
                )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))


class EnhancedMAIFProcessor:
    """Enhanced MAIF processor for format conversion and integration."""

    def __init__(self):
        self.supported_formats = ["json", "xml", "csv", "txt", "zip", "tar"]

    def _mime_to_format(self, mime_type: str) -> str:
        """Convert MIME type to format string."""
        mime_mapping = {
            "application/json": "json",
            "text/xml": "xml",
            "application/xml": "xml",
            "text/csv": "csv",
            "text/plain": "txt",
            "application/zip": "zip",
            "application/x-tar": "tar",
        }
        return mime_mapping.get(mime_type, "unknown")

    def convert_to_maif(
        self, input_path: str, output_path: str, input_format: str
    ) -> ConversionResult:
        """Convert various formats to MAIF (v3 format - self-contained)."""
        try:
            from .core import MAIFEncoder

            encoder = MAIFEncoder(output_path, agent_id="format_converter")

            if input_format == "json":
                with open(input_path, "r") as f:
                    data = json.load(f)
                content = json.dumps(data, indent=2)
                encoder.add_text_block(content, metadata={"source_format": "json"})
            elif input_format == "xml":
                return self.convert_xml_to_maif(input_path, output_path)
            else:
                with open(input_path, "r") as f:
                    content = f.read()
                encoder.add_text_block(
                    content, metadata={"source_format": input_format}
                )

            encoder.finalize()
            return ConversionResult(
                success=True,
                message=f"{input_format} converted to MAIF successfully",
                output_path=output_path,
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))

    def convert_xml_to_maif(self, xml_path: str, output_path: str) -> ConversionResult:
        """Convert XML file to MAIF format (v3 format - self-contained)."""
        try:
            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Convert XML to text representation
            xml_content = ET.tostring(root, encoding="unicode")

            # Create MAIF using core encoder (v3 format)
            from .core import MAIFEncoder

            encoder = MAIFEncoder(output_path, agent_id="xml_converter")
            encoder.add_text_block(xml_content, metadata={"source_format": "xml"})
            encoder.finalize()

            return ConversionResult(
                success=True,
                message="XML converted to MAIF successfully",
                output_path=output_path,
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))


class MAIFPluginManager:
    """Plugin manager for MAIF extensions."""

    def __init__(self):
        self.plugins = []
        self.hooks = {
            "pre_conversion": [],
            "post_conversion": [],
            "pre_validation": [],
            "post_validation": [],
        }

    def register_plugin(self, plugin):
        """Register a plugin."""
        self.plugins.append(plugin)

    def register_hook(self, hook_name: str, callback):
        """Register a hook callback."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)

    def execute_hooks(self, hook_name: str, *args, **kwargs):
        """Execute all plugins for a specific hook."""
        results = []
        if hook_name in self.hooks:
            for hook_func in self.hooks[hook_name]:
                results.append(hook_func(*args, **kwargs))
        return results

    def execute_hook(self, hook_name: str, *args, **kwargs):
        """Execute all plugins for a specific hook (alias for execute_hooks)."""
        return self.execute_hooks(hook_name, *args, **kwargs)
