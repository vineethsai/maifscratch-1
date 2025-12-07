"""
CLI tools for inspecting and managing MAIF CrewAI artifacts.

Usage:
    python -m maif.integrations.crewai.cli inspect crew_audit.maif
    python -m maif.integrations.crewai.cli verify crew_audit.maif
    python -m maif.integrations.crewai.cli export crew_audit.maif --format json
    python -m maif.integrations.crewai.cli tasks crew_audit.maif
    python -m maif.integrations.crewai.cli steps crew_audit.maif
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="maif-crewai",
        description="Inspect and manage MAIF CrewAI artifacts"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a MAIF artifact")
    inspect_parser.add_argument("artifact", help="Path to MAIF artifact")
    inspect_parser.add_argument("--tasks", "-t", action="store_true",
                                help="Show task details")
    inspect_parser.add_argument("--steps", "-s", action="store_true",
                                help="Show step details")
    inspect_parser.add_argument("--limit", "-n", type=int, default=10,
                                help="Limit number of items shown")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify artifact integrity")
    verify_parser.add_argument("artifact", help="Path to MAIF artifact")
    verify_parser.add_argument("--verbose", "-v", action="store_true",
                               help="Show detailed verification")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export artifact data")
    export_parser.add_argument("artifact", help="Path to MAIF artifact")
    export_parser.add_argument("--format", "-f", choices=["json", "csv", "markdown"],
                               default="json", help="Export format")
    export_parser.add_argument("--output", "-o", help="Output file (stdout if omitted)")
    
    # Tasks command
    tasks_parser = subparsers.add_parser("tasks", help="List all tasks")
    tasks_parser.add_argument("artifact", help="Path to MAIF artifact")
    tasks_parser.add_argument("--verbose", "-v", action="store_true",
                              help="Show task outputs")
    
    # Steps command
    steps_parser = subparsers.add_parser("steps", help="List agent reasoning steps")
    steps_parser.add_argument("artifact", help="Path to MAIF artifact")
    steps_parser.add_argument("--limit", "-n", type=int, default=20,
                              help="Limit number of steps shown")
    
    # Memory command
    memory_parser = subparsers.add_parser("memory", help="List stored memories")
    memory_parser.add_argument("artifact", help="Path to MAIF artifact")
    memory_parser.add_argument("--agent", "-a", help="Filter by agent")
    memory_parser.add_argument("--search", "-s", help="Search memories")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate HTML audit report")
    report_parser.add_argument("artifact", help="Path to MAIF artifact")
    report_parser.add_argument("--output", "-o", help="Output file (default: artifact_report.html)")
    report_parser.add_argument("--open", action="store_true", help="Open in browser after generation")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "tasks":
        cmd_tasks(args)
    elif args.command == "steps":
        cmd_steps(args)
    elif args.command == "memory":
        cmd_memory(args)


def cmd_inspect(args):
    """Inspect a MAIF CrewAI artifact."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    # Basic info
    print(f"\n{'='*60}")
    print(f"MAIF CrewAI Artifact: {artifact_path.name}")
    print(f"{'='*60}")
    print(f"Size: {artifact_path.stat().st_size / 1024:.1f} KB")
    print(f"Blocks: {len(decoder.blocks)}")
    
    # Verify integrity
    is_valid, errors = decoder.verify_integrity()
    status = "VALID" if is_valid else "INVALID"
    print(f"Integrity: {status}")
    
    if errors:
        for err in errors[:3]:
            print(f"  - {err}")
    
    # Count by type
    type_counts = {}
    tasks_completed = 0
    steps_executed = 0
    agents = set()
    
    for block in decoder.blocks:
        meta = block.metadata or {}
        event_type = meta.get("type", "unknown")
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        if event_type == "task_end":
            tasks_completed += 1
        elif event_type == "agent_action":
            steps_executed += 1
        
        agent_id = meta.get("agent_id")
        if agent_id:
            agents.add(agent_id)
    
    print(f"\nAgents: {len(agents)}")
    print(f"Tasks completed: {tasks_completed}")
    print(f"Steps executed: {steps_executed}")
    print(f"\nEvent Types:")
    for event_type, count in sorted(type_counts.items()):
        print(f"  {event_type}: {count}")
    
    # Show tasks if requested
    if args.tasks:
        print(f"\n{'='*60}")
        print("Tasks")
        print(f"{'='*60}")
        _show_tasks(decoder, args.limit)
    
    # Show steps if requested
    if args.steps:
        print(f"\n{'='*60}")
        print("Agent Steps")
        print(f"{'='*60}")
        _show_steps(decoder, args.limit)


def _show_tasks(decoder, limit):
    """Show task details."""
    count = 0
    for block in decoder.blocks:
        meta = block.metadata or {}
        if meta.get("type") != "task_end":
            continue
        
        count += 1
        if count > limit:
            print(f"\n... and more (use --limit to see more)")
            break
        
        ts = meta.get("timestamp", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "N/A"
        
        # Parse task data
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            task_data = json.loads(data).get("data", {})
        except:
            task_data = {}
        
        desc = task_data.get("task_description", "Unknown")[:50]
        agent = task_data.get("agent", "N/A")
        
        print(f"\n[{count}] {desc}...")
        print(f"    Agent: {agent}")
        print(f"    Time: {time_str}")


def _show_steps(decoder, limit):
    """Show agent step details."""
    count = 0
    for block in decoder.blocks:
        meta = block.metadata or {}
        if meta.get("type") != "agent_action":
            continue
        
        count += 1
        if count > limit:
            print(f"\n... and more (use --limit to see more)")
            break
        
        ts = meta.get("timestamp", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "N/A"
        
        # Parse step data
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            step_data = json.loads(data).get("data", {})
        except:
            step_data = {}
        
        action = step_data.get("action", "unknown")
        thought = step_data.get("thought", "")[:40]
        
        print(f"[{count}] [{time_str}] {action}: {thought}...")


def cmd_verify(args):
    """Verify artifact integrity."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nVerifying: {artifact_path.name}")
    print("-" * 60)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    # Check header
    print("[1/4] File header... ", end="")
    try:
        with open(artifact_path, "rb") as f:
            magic = f.read(4)
        if magic == b"MAIF":
            print("OK (MAIF format)")
        else:
            print(f"FAIL (got {magic})")
    except Exception as e:
        print(f"FAIL ({e})")
    
    # Check blocks
    print(f"[2/4] Block count... ", end="")
    print(f"OK ({len(decoder.blocks)} blocks)")
    
    # Verify integrity
    print("[3/4] Hash chain... ", end="")
    is_valid, errors = decoder.verify_integrity()
    if is_valid:
        print("OK (all blocks linked)")
    else:
        print("FAIL")
        for err in errors:
            print(f"       - {err}")
    
    # Check signatures
    print("[4/4] Signatures... ", end="")
    if is_valid:
        print(f"OK ({len(decoder.blocks)} verified)")
    else:
        print("FAIL (integrity check failed)")
    
    print("-" * 60)
    if is_valid:
        print("RESULT: Artifact integrity VERIFIED")
        sys.exit(0)
    else:
        print("RESULT: Artifact integrity FAILED")
        sys.exit(1)


def cmd_export(args):
    """Export artifact data."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    # Collect data
    events = []
    tasks = []
    steps = []
    
    for block in decoder.blocks:
        meta = block.metadata or {}
        event_type = meta.get("type", "unknown")
        
        # Parse block data
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            parsed_data = json.loads(data)
        except:
            parsed_data = {"raw": str(block.data)[:500]}
        
        event = {
            "type": event_type,
            "agent_id": meta.get("agent_id", ""),
            "timestamp": meta.get("timestamp", 0),
            "data": parsed_data,
        }
        events.append(event)
        
        if event_type == "task_end":
            tasks.append(parsed_data.get("data", {}))
        elif event_type == "agent_action":
            steps.append(parsed_data.get("data", {}))
    
    # Format output
    if args.format == "json":
        output = json.dumps({
            "artifact": str(artifact_path),
            "summary": {
                "total_events": len(events),
                "tasks_completed": len(tasks),
                "steps_executed": len(steps),
            },
            "tasks": tasks,
            "steps": steps,
            "events": events,
        }, indent=2, default=str)
    
    elif args.format == "csv":
        lines = ["timestamp,type,agent_id,action"]
        for e in events:
            ts = datetime.fromtimestamp(e["timestamp"]).isoformat() if e["timestamp"] else ""
            action = e["data"].get("data", {}).get("action", "") if isinstance(e["data"], dict) else ""
            lines.append(f"{ts},{e['type']},{e['agent_id']},{action}")
        output = "\n".join(lines)
    
    elif args.format == "markdown":
        lines = [
            f"# CrewAI Audit Export: {artifact_path.name}",
            "",
            f"**Total Events:** {len(events)}",
            f"**Tasks Completed:** {len(tasks)}",
            f"**Steps Executed:** {len(steps)}",
            "",
            "## Tasks",
            "",
            "| # | Description | Agent |",
            "|---|-------------|-------|",
        ]
        for i, task in enumerate(tasks[:20], 1):
            desc = task.get("task_description", "Unknown")[:40]
            agent = task.get("agent", "N/A")
            lines.append(f"| {i} | {desc}... | {agent} |")
        
        if len(tasks) > 20:
            lines.append(f"\n*... and {len(tasks) - 20} more tasks*")
        
        lines.extend([
            "",
            "## Agent Steps",
            "",
            "| Time | Action | Thought |",
            "|------|--------|---------|",
        ])
        for step in steps[:30]:
            action = step.get("action", "unknown")
            thought = step.get("thought", "")[:30]
            lines.append(f"| - | {action} | {thought}... |")
        
        if len(steps) > 30:
            lines.append(f"\n*... and {len(steps) - 30} more steps*")
        
        output = "\n".join(lines)
    
    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Exported to: {args.output}")
    else:
        print(output)


def cmd_tasks(args):
    """List all tasks in artifact."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    print(f"\nTasks in {artifact_path.name}:")
    print("-" * 70)
    print(f"{'#':<3} | {'Agent':<20} | {'Description':<40}")
    print("-" * 70)
    
    count = 0
    for block in decoder.blocks:
        meta = block.metadata or {}
        if meta.get("type") != "task_end":
            continue
        
        count += 1
        
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            task_data = json.loads(data).get("data", {})
        except:
            task_data = {}
        
        desc = task_data.get("task_description", "Unknown")[:38]
        agent = task_data.get("agent", "N/A")[:18]
        
        print(f"{count:<3} | {agent:<20} | {desc}...")
        
        if args.verbose:
            output = task_data.get("output", {})
            if isinstance(output, dict):
                raw = output.get("raw", "")[:100]
            else:
                raw = str(output)[:100]
            print(f"    Output: {raw}...")
    
    print("-" * 70)
    print(f"Total: {count} tasks")


def cmd_steps(args):
    """List agent reasoning steps."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    print(f"\nAgent Steps in {artifact_path.name}:")
    print("-" * 70)
    print(f"{'#':<4} | {'Action':<15} | {'Thought':<45}")
    print("-" * 70)
    
    count = 0
    for block in decoder.blocks:
        meta = block.metadata or {}
        if meta.get("type") != "agent_action":
            continue
        
        count += 1
        if count > args.limit:
            print(f"\n... and more (use --limit to see more)")
            break
        
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            step_data = json.loads(data).get("data", {})
        except:
            step_data = {}
        
        action = step_data.get("action", "unknown")[:13]
        thought = step_data.get("thought", "")[:43]
        
        print(f"{count:<4} | {action:<15} | {thought}...")
    
    print("-" * 70)
    print(f"Showing {min(count, args.limit)} of {count} steps")


def cmd_memory(args):
    """List stored memories."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    print(f"\nMemories in {artifact_path.name}:")
    print("-" * 70)
    
    memories = []
    for block in decoder.blocks:
        meta = block.metadata or {}
        if meta.get("type") not in ["memory_save", "memory"]:
            continue
        
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            mem_data = json.loads(data).get("data", {})
        except:
            mem_data = {}
        
        # Filter by agent if specified
        if args.agent and mem_data.get("agent") != args.agent:
            continue
        
        # Filter by search if specified
        if args.search:
            content = mem_data.get("content", "").lower()
            if args.search.lower() not in content:
                continue
        
        memories.append(mem_data)
    
    if not memories:
        print("No memories found.")
        return
    
    print(f"{'#':<3} | {'Agent':<15} | {'Importance':<10} | Content")
    print("-" * 70)
    
    for i, mem in enumerate(memories, 1):
        agent = mem.get("agent", "unknown")[:13]
        importance = mem.get("importance", 0.5)
        content = mem.get("content", "")[:35]
        
        print(f"{i:<3} | {agent:<15} | {importance:<10.2f} | {content}...")
    
    print("-" * 70)
    print(f"Total: {len(memories)} memories")


def cmd_report(args):
    """Generate an HTML audit report."""
    from maif import MAIFDecoder
    import webbrowser
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    # Default output path
    output_path = args.output or f"{artifact_path.stem}_report.html"
    
    # Verify integrity
    is_valid, errors = decoder.verify_integrity()
    
    # Collect data
    tasks = []
    steps = []
    event_counts = {}
    
    for block in decoder.blocks:
        meta = block.metadata or {}
        event_type = meta.get("type", "unknown")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            parsed = json.loads(data)
        except:
            parsed = {}
        
        if event_type == "task_end":
            task_data = parsed.get("data", {})
            tasks.append({
                "description": task_data.get("task_description", "Unknown")[:80],
                "agent": task_data.get("agent", "unknown"),
                "timestamp": meta.get("timestamp", 0),
            })
        
        elif event_type == "agent_action":
            step_data = parsed.get("data", {})
            steps.append({
                "action": step_data.get("action", "unknown"),
                "thought": step_data.get("thought", "")[:100],
                "timestamp": meta.get("timestamp", 0),
            })
    
    # Generate HTML
    html = _generate_html_report(
        artifact_name=artifact_path.name,
        is_valid=is_valid,
        errors=errors,
        total_blocks=len(decoder.blocks),
        event_counts=event_counts,
        tasks=tasks,
        steps=steps,
    )
    
    # Write report
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"Report generated: {output_path}")
    
    # Open in browser if requested
    if args.open:
        webbrowser.open(f"file://{Path(output_path).absolute()}")


def _generate_html_report(
    artifact_name: str,
    is_valid: bool,
    errors: list,
    total_blocks: int,
    event_counts: dict,
    tasks: list,
    steps: list,
) -> str:
    """Generate HTML report content."""
    
    status_color = "#22c55e" if is_valid else "#ef4444"
    status_text = "VERIFIED" if is_valid else "FAILED"
    
    # Build event rows
    event_rows = ""
    for event_type, count in sorted(event_counts.items()):
        event_rows += f"<tr><td>{event_type}</td><td>{count}</td></tr>\n"
    
    # Build task rows
    task_rows = ""
    for i, task in enumerate(tasks, 1):
        ts = datetime.fromtimestamp(task["timestamp"]).strftime("%H:%M:%S") if task["timestamp"] else "-"
        task_rows += f"""<tr>
            <td>{i}</td>
            <td>{task['agent']}</td>
            <td>{task['description']}...</td>
            <td>{ts}</td>
        </tr>\n"""
    
    # Build step rows (limit to 50)
    step_rows = ""
    for i, step in enumerate(steps[:50], 1):
        ts = datetime.fromtimestamp(step["timestamp"]).strftime("%H:%M:%S") if step["timestamp"] else "-"
        step_rows += f"""<tr>
            <td>{i}</td>
            <td><code>{step['action']}</code></td>
            <td>{step['thought']}...</td>
            <td>{ts}</td>
        </tr>\n"""
    
    if len(steps) > 50:
        step_rows += f"<tr><td colspan='4'><em>... and {len(steps) - 50} more steps</em></td></tr>"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAIF Audit Report - {artifact_name}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card: #1e293b;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --border: #334155;
            --accent: #3b82f6;
            --success: #22c55e;
            --danger: #ef4444;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'SF Mono', 'Fira Code', monospace;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ 
            font-size: 1.5rem; 
            margin-bottom: 0.5rem;
            color: var(--accent);
        }}
        h2 {{ 
            font-size: 1.1rem; 
            margin: 1.5rem 0 0.75rem;
            color: var(--muted);
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--border);
        }}
        .status {{
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
            background: {status_color}20;
            color: {status_color};
            border: 1px solid {status_color};
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: var(--card);
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid var(--border);
        }}
        .card-label {{ color: var(--muted); font-size: 0.8rem; }}
        .card-value {{ font-size: 1.5rem; font-weight: bold; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.85rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: var(--card);
            color: var(--muted);
            font-weight: 500;
        }}
        tr:hover {{ background: var(--card); }}
        code {{
            background: var(--card);
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-size: 0.85em;
        }}
        .footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--muted);
            font-size: 0.8rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>MAIF Audit Report</h1>
                <span style="color: var(--muted);">{artifact_name}</span>
            </div>
            <div class="status">{status_text}</div>
        </div>
        
        <div class="cards">
            <div class="card">
                <div class="card-label">Total Events</div>
                <div class="card-value">{total_blocks}</div>
            </div>
            <div class="card">
                <div class="card-label">Tasks Completed</div>
                <div class="card-value">{len(tasks)}</div>
            </div>
            <div class="card">
                <div class="card-label">Agent Steps</div>
                <div class="card-value">{len(steps)}</div>
            </div>
            <div class="card">
                <div class="card-label">Integrity</div>
                <div class="card-value" style="color: {status_color};">{'Pass' if is_valid else 'Fail'}</div>
            </div>
        </div>
        
        <h2>Event Summary</h2>
        <table>
            <thead>
                <tr><th>Event Type</th><th>Count</th></tr>
            </thead>
            <tbody>
                {event_rows}
            </tbody>
        </table>
        
        <h2>Tasks Completed ({len(tasks)})</h2>
        <table>
            <thead>
                <tr><th>#</th><th>Agent</th><th>Description</th><th>Time</th></tr>
            </thead>
            <tbody>
                {task_rows if task_rows else '<tr><td colspan="4">No tasks recorded</td></tr>'}
            </tbody>
        </table>
        
        <h2>Agent Reasoning Steps ({len(steps)})</h2>
        <table>
            <thead>
                <tr><th>#</th><th>Action</th><th>Thought</th><th>Time</th></tr>
            </thead>
            <tbody>
                {step_rows if step_rows else '<tr><td colspan="4">No steps recorded</td></tr>'}
            </tbody>
        </table>
        
        <div class="footer">
            Generated by MAIF CrewAI Integration | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>"""
    
    return html


if __name__ == "__main__":
    main()

