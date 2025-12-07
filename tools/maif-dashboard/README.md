# MAIF Dashboard

A comprehensive web-based dashboard for viewing and managing MAIF artifacts with multi-artifact support, session management, and compliance reporting.

## Features

### Overview Dashboard
- Total artifacts, blocks, and signature statistics
- Recent activity feed
- Block type distribution chart
- Provenance timeline

### Artifacts View
- Load and manage multiple MAIF artifacts
- Table view with sorting and filtering
- Select and compare artifacts
- Detailed artifact inspection

### Sessions View
- Automatic session detection from thread IDs
- Session timeline and statistics
- Block-level session tracking

### Integrations View
- LangGraph checkpoint statistics
- LangChain call tracking
- CrewAI crew and task metrics
- Integration status overview

### Compliance View
- Integrity verification checks
- Security summary
- Complete audit log
- Report generation

## Quick Start

```bash
# Start a local server
cd tools/maif-dashboard
python3 -m http.server 8080

# Open in browser
open http://localhost:8080
```

## Usage

1. Click **Open Artifact** to load one or more `.maif` files
2. Navigate between views using the top navigation
3. Use the **Artifacts** view to manage loaded files
4. Check **Compliance** view for integrity verification
5. Generate reports for audit purposes

## Architecture

The dashboard builds on the existing MAIF Explorer parser:

```
maif-dashboard/
├── index.html        # Main HTML structure
├── styles.css        # CSS styling (dark/light themes)
├── dashboard.js      # Application logic
└── README.md         # This file

Uses from ../maif-explorer/:
└── maif-parser.js    # MAIF binary parser
```

## Requirements

- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- No build step required

## Related

- [MAIF Explorer](../maif-explorer/) - Single-artifact viewer
- [VSCode Extension](../vscode-maif/) - VS Code integration
- [MAIF Python Library](../../maif/) - Python implementation

