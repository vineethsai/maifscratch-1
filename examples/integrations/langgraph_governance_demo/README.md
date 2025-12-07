# MAIF + LangGraph: Enterprise AI Governance Demo

Interactive demonstration of MAIF's enterprise-grade features for AI agent workflows with full cryptographic provenance tracking.

## Overview

This demo showcases how MAIF provides:

- **Cryptographic Provenance**: Every agent action is Ed25519 signed and hash-chained
- **Tamper Detection**: Any modification breaks the cryptographic chain
- **Data Governance**: Role-based access control with full audit trails
- **Multi-Agent Coordination**: Multiple specialized agents with clear handoffs
- **Compliance Reporting**: Generate audit reports from artifacts

## Requirements

```bash
pip install maif[integrations]
```

## Quick Start

```bash
cd examples/integrations/langgraph_governance_demo
python main.py
```

## Features

### 1. Audited Conversation

Start a new conversation session with full provenance tracking. Every query goes through:
- Access control gate (permission checking)
- Query router (intent classification)
- Specialized agents (financial, general)
- Compliance checker (policy validation)

All actions are logged to a MAIF artifact with cryptographic signatures.

### 2. Resume Previous Session

Load and continue a previous conversation from its MAIF artifact. The system:
- Verifies artifact integrity before loading
- Restores full conversation state
- Continues logging to the same artifact

### 3. Inspect Artifact Provenance

Deep dive into the audit trail with multiple views:
- **Timeline View**: Chronological event sequence
- **Block Details**: Inspect individual blocks and their data
- **Agent Activity**: Events grouped by agent
- **Hash Chain**: Visualize cryptographic linking
- **Signature Audit**: Verify all Ed25519 signatures

### 4. Security Verification

Run comprehensive security checks:
- File header validation
- Hash chain verification
- Ed25519 signature verification
- Footer checksum validation

### 5. Tamper Detection Demo

See what happens when artifact data is modified:
1. Creates a test artifact
2. Verifies original integrity
3. Simulates tampering (byte modification)
4. Shows how MAIF detects the tampering

### 6. Generate Compliance Report

Export audit reports in multiple formats:
- Summary Report (Markdown)
- Detailed Audit Log (JSON)
- Agent Activity Report
- Timeline Export (CSV)

### 7. Access Control Simulation

Test role-based access control:
- **analyst**: Read financial summaries only
- **manager**: Full financial data + limited PII
- **admin**: Full access
- **auditor**: Audit logs only

Every access attempt (granted or denied) is cryptographically logged.

## Directory Structure

```
langgraph_governance_demo/
    main.py                 # Entry point
    demo_app.py            # Main application and menu
    conversation_agent.py   # Multi-agent LangGraph
    provenance_inspector.py # Artifact inspection
    security_verifier.py    # Security verification
    tamper_demo.py         # Tamper detection demo
    report_generator.py    # Report generation
    access_control_demo.py # Access control simulation
    README.md              # This file
    sessions/              # Session artifacts
    reports/               # Generated reports
```

## Architecture

```
User Query
    |
    v
+------------------+
| Access Control   |  <-- Check permissions, log attempt
+--------+---------+
         |
         v
+------------------+
|   Query Router   |  <-- Classify intent, route to agent
+--------+---------+
         |
    +----+----+
    v         v
+--------+ +----------+
|Financial| | General  |  <-- Specialized processing
| Agent  | |  Agent   |
+----+---+ +----+-----+
     |          |
     +----+-----+
          v
+------------------+
|   Synthesizer    |  <-- Generate response
+--------+---------+
         |
         v
+------------------+
|   Compliance     |  <-- Check for policy violations
|    Checker       |
+------------------+
```

All nodes log their actions to the MAIF artifact via the checkpointer.

## Key MAIF Concepts Demonstrated

### Hash Chain

Each block contains a hash of the previous block:

```
Block 1 (genesis)
    |
    | Hash: a3f2b1c8...
    v
Block 2
    |
    | Hash: 7d4e9f2a...
    v
Block 3
    ...
```

Modifying any block invalidates all subsequent blocks.

### Ed25519 Signatures

Every block is signed with Ed25519:
- 64-byte compact signatures
- Fast verification
- Cryptographically secure

### Provenance Events

Standard event types for consistent logging:
- `session_start` / `session_end`
- `access_check` (granted/denied)
- `state_checkpoint`
- `node_start` / `node_end`
- Custom events

## Example Session

```
$ python main.py

================================================================================
    MAIF + LangGraph: Enterprise AI Governance Demo
================================================================================

MAIN MENU
[1] Start Audited Conversation
[2] Resume Previous Session
[3] Inspect Artifact Provenance
[4] Security Verification
[5] Tamper Detection Demo
[6] Generate Compliance Report
[7] Access Control Simulation
[8] Exit

Choice: 1

Enter session name: quarterly-review

Session: quarterly-review
Artifact: sessions/quarterly-review.maif
User Role: analyst

You: What are our Q3 revenue projections?

[ACCESS CONTROL] Checking permissions for role: analyst
[ACCESS CONTROL] Permission: GRANTED
[ROUTER] Query classified as: financial (confidence: 2.00)
[ROUTER] Routing to: financial_agent
[FINANCIAL AGENT] Processing financial query...
[FINANCIAL AGENT] Retrieved 1 relevant facts
[SYNTHESIZER] Generating response...
[COMPLIANCE] Checking response for policy violations...
[COMPLIANCE] Check result: PASSED
