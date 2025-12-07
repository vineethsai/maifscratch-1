# Customer Support Multi-Agent System

A production-ready customer support system with multiple specialized agents and full audit trail.

## Architecture

```
Customer Query
     │
     ▼
┌─────────┐
│ Triage  │ ── Classify category & priority
└────┬────┘
     │
     ├──────────┬──────────┬──────────┐
     ▼          ▼          ▼          ▼
┌─────────┐┌─────────┐┌─────────┐┌──────────┐
│Technical││ Billing ││ General ││Escalation│
└─────────┘└─────────┘└─────────┘└──────────┘
```

## Features

- **Intelligent Triage**: Auto-classify tickets by category and priority
- **Specialized Agents**: Technical, Billing, General, Escalation
- **Priority-based Routing**: Urgent tickets auto-escalate
- **Full Audit Trail**: Every agent action logged with provenance

## Quick Start

```bash
pip install maif langgraph
python main.py
```

## Example Session

```
Customer Support System with MAIF Provenance
============================================================
Ticket ID: TKT-20241207123456

Customer: I can't login to my account

[TRIAGE] Category: technical, Priority: medium
[TECHNICAL] Resolved: True

[TECHNICAL]: Here's how to resolve your issue:

For login issues: 1) Clear browser cache, 2) Reset password...
```

## Compliance

All agent actions are cryptographically signed and stored in a tamper-evident MAIF artifact, suitable for:
- SOC2 compliance audits
- Customer support QA reviews
- Dispute resolution
- Agent performance tracking

