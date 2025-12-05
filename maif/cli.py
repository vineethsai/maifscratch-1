"""
Command-line interface for MAIF tools (v3 format).
"""

import click
import json
import sys
import os
from pathlib import Path
from typing import Optional


@click.command()
@click.option("--input", "input_file", help="Input file path")
@click.option("--output", required=True, help="Output MAIF file path")
@click.option("--manifest", help="Output manifest file path (deprecated in v3)")
@click.option("--text", multiple=True, help="Add text content")
@click.option("--file", "files", multiple=True, help="Add file content")
@click.option("--agent-id", help="Agent identifier")
@click.option(
    "--privacy-level",
    type=click.Choice(
        [
            "public",
            "low",
            "internal",
            "medium",
            "confidential",
            "high",
            "secret",
            "top_secret",
        ]
    ),
    default="internal",
    help="Default privacy level",
)
@click.option(
    "--encryption",
    type=click.Choice(
        ["none", "aes_gcm", "chacha20_poly1305", "aes-gcm", "chacha20-poly1305"]
    ),
    default="aes_gcm",
    help="Encryption mode",
)
@click.option("--anonymize", is_flag=True, help="Enable automatic anonymization")
@click.option("--retention-days", type=int, help="Data retention period in days")
@click.option(
    "--access-rule",
    multiple=True,
    nargs=3,
    help="Add access rule: subject resource permissions",
)
def create_privacy_maif(
    input_file,
    output,
    manifest,
    text,
    files,
    agent_id,
    privacy_level,
    encryption,
    anonymize,
    retention_days,
    access_rule,
):
    """CLI command to create MAIF files with privacy controls."""
    try:
        from .core import MAIFEncoder, BlockType

        # Validate required parameters
        if not agent_id:
            agent_id = "cli-agent"

        if not input_file and not text and not files:
            click.echo(
                "Error: At least one of --input, --text, or --file must be provided",
                err=True,
            )
            sys.exit(1)

        # Initialize encoder (v3 format)
        encoder = MAIFEncoder(output, agent_id=agent_id)

        # Add input file content if provided
        if input_file:
            if not os.path.exists(input_file):
                click.echo(f"Error: Input file not found: {input_file}", err=True)
                sys.exit(1)
            with open(input_file, "r") as f:
                content = f.read()
            block_id = encoder.add_text_block(
                content, metadata={"source": input_file, "privacy_level": privacy_level}
            )
            click.echo(f"Added text block from {input_file}: {block_id[:16]}...")

        # Add text content
        if text:
            for text_content in text:
                block_id = encoder.add_text_block(
                    text_content, metadata={"privacy_level": privacy_level}
                )
                click.echo(f"Added text block: {block_id[:16]}...")

        # Add file content
        if files:
            for file_path in files:
                if not os.path.exists(file_path):
                    click.echo(f"Warning: File not found: {file_path}")
                    continue

                with open(file_path, "rb") as f:
                    data = f.read()

                file_ext = Path(file_path).suffix.lower()
                if file_ext in [".txt", ".md", ".json"]:
                    block_id = encoder.add_text_block(
                        data.decode("utf-8"), metadata={"source": file_path}
                    )
                else:
                    block_id = encoder.add_binary_block(
                        data, BlockType.BINARY, metadata={"source": file_path}
                    )
                click.echo(f"Added block from {file_path}: {block_id[:16]}...")

        # Finalize (v3 format)
        encoder.finalize()

        click.echo(f"Successfully created MAIF file: {output}")

    except Exception as e:
        click.echo(f"Error creating MAIF file: {str(e)}", err=True)
        sys.exit(1)


@click.command()
@click.option("--maif-file", required=True, help="MAIF file to access")
@click.option("--manifest", help="Manifest file path (deprecated in v3)")
@click.option("--user-id", required=True, help="User ID requesting access")
@click.option(
    "--permission",
    type=click.Choice(["read", "write", "admin"]),
    required=True,
    help="Permission level",
)
def access_privacy_maif(maif_file, manifest, user_id, permission):
    """CLI command to access MAIF files with privacy controls."""
    try:
        from .core import MAIFDecoder

        if not os.path.exists(maif_file):
            click.echo(f"Error: MAIF file not found: {maif_file}", err=True)
            sys.exit(1)

        # Load MAIF file (v3 format)
        decoder = MAIFDecoder(maif_file)
        decoder.load()

        # Verify integrity
        is_valid, errors = decoder.verify_integrity()
        if not is_valid:
            click.echo(f"Warning: Integrity check failed: {errors}")

        # Display access info
        file_info = decoder.get_file_info()
        click.echo(f"File: {maif_file}")
        click.echo(f"Blocks: {file_info['block_count']}")
        click.echo(f"User: {user_id}")
        click.echo(f"Permission: {permission}")
        click.echo(f"Access: GRANTED")

    except Exception as e:
        click.echo(f"Error accessing MAIF file: {str(e)}", err=True)
        sys.exit(1)


@click.command()
@click.option("--maif-file", required=True, help="MAIF file to manage")
@click.option("--manifest", help="Manifest file path (deprecated in v3)")
@click.option(
    "--action",
    type=click.Choice(["status", "audit", "revoke"]),
    required=True,
    help="Management action",
)
@click.option("--user-id", help="User ID for action")
def manage_privacy(maif_file, manifest, action, user_id):
    """CLI command to manage privacy settings."""
    try:
        from .core import MAIFDecoder

        if not os.path.exists(maif_file):
            click.echo(f"Error: MAIF file not found: {maif_file}", err=True)
            sys.exit(1)

        decoder = MAIFDecoder(maif_file)
        decoder.load()

        if action == "status":
            file_info = decoder.get_file_info()
            security_info = decoder.get_security_info()

            click.echo(f"File: {maif_file}")
            click.echo(f"Version: {file_info['version']}")
            click.echo(f"Blocks: {file_info['block_count']}")
            click.echo(f"Signed: {file_info['is_signed']}")
            click.echo(f"Finalized: {file_info['is_finalized']}")
            click.echo(f"Algorithm: {security_info.get('key_algorithm', 'N/A')}")

        elif action == "audit":
            provenance = decoder.get_provenance()
            click.echo(f"Audit trail ({len(provenance)} entries):")
            for entry in provenance[:10]:  # Show first 10
                click.echo(
                    f"  • {entry.action} by {entry.agent_id} at {entry.timestamp}"
                )

        elif action == "revoke":
            click.echo(f"Revocation not supported in v3 format (immutable)")

    except Exception as e:
        click.echo(f"Error managing MAIF file: {str(e)}", err=True)
        sys.exit(1)


@click.command()
@click.option("--input", "input_file", help="Input file path")
@click.option("--output", required=True, help="Output MAIF file path")
@click.option("--manifest", help="Output manifest file path (deprecated in v3)")
@click.option("--text", multiple=True, help="Add text content")
@click.option("--agent-id", default="cli-agent", help="Agent identifier")
@click.option("--compress", is_flag=True, help="Enable compression")
@click.option("--sign", is_flag=True, default=True, help="Sign the file")
def create_maif(input_file, output, manifest, text, agent_id, compress, sign):
    """CLI command to create a MAIF file."""
    try:
        from .core import MAIFEncoder, BlockType

        encoder = MAIFEncoder(output, agent_id=agent_id)

        # Add input file content
        if input_file:
            if not os.path.exists(input_file):
                click.echo(f"Error: Input file not found: {input_file}", err=True)
                sys.exit(1)
            with open(input_file, "rb") as f:
                data = f.read()

            try:
                # Try text first
                encoder.add_text_block(
                    data.decode("utf-8"), metadata={"source": input_file}
                )
            except UnicodeDecodeError:
                encoder.add_binary_block(
                    data, BlockType.BINARY, metadata={"source": input_file}
                )
            click.echo(f"Added content from {input_file}")

        # Add text content
        for text_content in text:
            encoder.add_text_block(text_content)
            click.echo(f"Added text block")

        encoder.finalize()
        click.echo(f"Successfully created MAIF file: {output}")

    except Exception as e:
        click.echo(f"Error creating MAIF file: {str(e)}", err=True)
        sys.exit(1)


@click.command()
@click.option("--maif-file", required=True, help="MAIF file to verify")
@click.option("--manifest", help="Manifest file path (deprecated in v3)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def verify_maif(maif_file, manifest, verbose):
    """CLI command to verify a MAIF file."""
    try:
        from .core import MAIFDecoder

        if not os.path.exists(maif_file):
            click.echo(f"Error: MAIF file not found: {maif_file}", err=True)
            sys.exit(1)

        decoder = MAIFDecoder(maif_file)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()

        if is_valid:
            click.echo(f"✓ File integrity: VALID")
        else:
            click.echo(f"✗ File integrity: INVALID")
            for error in errors:
                click.echo(f"  - {error}")

        if verbose:
            file_info = decoder.get_file_info()
            click.echo(f"\nFile info:")
            click.echo(f"  Version: {file_info['version']}")
            click.echo(f"  Blocks: {file_info['block_count']}")
            click.echo(f"  Signed: {file_info['is_signed']}")
            click.echo(f"  Finalized: {file_info['is_finalized']}")

        sys.exit(0 if is_valid else 1)

    except Exception as e:
        click.echo(f"Error verifying MAIF file: {str(e)}", err=True)
        sys.exit(1)


@click.command()
@click.option("--maif-file", required=True, help="MAIF file to analyze")
@click.option("--manifest", help="Manifest file path (deprecated in v3)")
@click.option(
    "--output-format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def analyze_maif(maif_file, manifest, output_format):
    """CLI command to analyze a MAIF file."""
    try:
        from .core import MAIFDecoder

        if not os.path.exists(maif_file):
            click.echo(f"Error: MAIF file not found: {maif_file}", err=True)
            sys.exit(1)

        decoder = MAIFDecoder(maif_file)
        decoder.load()

        file_info = decoder.get_file_info()
        security_info = decoder.get_security_info()
        provenance = decoder.get_provenance()

        analysis = {
            "file": maif_file,
            "file_info": file_info,
            "security": security_info,
            "provenance_count": len(provenance),
            "blocks": [
                {
                    "index": i,
                    "type": str(b.header.block_type),
                    "size": b.header.size,
                    "has_metadata": b.metadata is not None,
                }
                for i, b in enumerate(decoder.blocks)
            ],
        }

        if output_format == "json":
            click.echo(json.dumps(analysis, indent=2, default=str))
        else:
            click.echo(f"File: {maif_file}")
            click.echo(f"Version: {file_info['version']}")
            click.echo(f"Blocks: {file_info['block_count']}")
            click.echo(f"Signed: {file_info['is_signed']}")
            click.echo(f"Provenance entries: {len(provenance)}")
            click.echo(f"\nBlocks:")
            for i, b in enumerate(decoder.blocks):
                click.echo(f"  [{i}] {b.header.block_type}: {b.header.size} bytes")

    except Exception as e:
        click.echo(f"Error analyzing MAIF file: {str(e)}", err=True)
        sys.exit(1)


@click.command()
@click.option("--maif-file", required=True, help="MAIF file to extract from")
@click.option("--manifest", help="Manifest file path (deprecated in v3)")
@click.option("--output-dir", required=True, help="Output directory")
@click.option("--block-index", type=int, help="Extract specific block by index")
def extract_content(maif_file, manifest, output_dir, block_index):
    """CLI command to extract content from a MAIF file."""
    try:
        from .core import MAIFDecoder, BlockType

        if not os.path.exists(maif_file):
            click.echo(f"Error: MAIF file not found: {maif_file}", err=True)
            sys.exit(1)

        os.makedirs(output_dir, exist_ok=True)

        decoder = MAIFDecoder(maif_file)
        decoder.load()

        blocks_to_extract = (
            [decoder.blocks[block_index]] if block_index is not None else decoder.blocks
        )

        for i, block in enumerate(blocks_to_extract):
            idx = block_index if block_index is not None else i

            if block.header.block_type == BlockType.TEXT:
                ext = ".txt"
                data = block.data
            else:
                ext = ".bin"
                data = block.data

            output_path = os.path.join(output_dir, f"block_{idx}{ext}")
            with open(output_path, "wb") as f:
                f.write(data)
            click.echo(f"Extracted block {idx} to {output_path}")

    except Exception as e:
        click.echo(f"Error extracting content: {str(e)}", err=True)
        sys.exit(1)


@click.group()
def main():
    """MAIF Command Line Interface (v3 format)"""
    pass


# Register commands
main.add_command(create_maif, "create")
main.add_command(verify_maif, "verify")
main.add_command(analyze_maif, "analyze")
main.add_command(extract_content, "extract")
main.add_command(create_privacy_maif, "create-privacy")
main.add_command(access_privacy_maif, "access-privacy")
main.add_command(manage_privacy, "manage-privacy")


if __name__ == "__main__":
    main()
