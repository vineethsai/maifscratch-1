# Research Assistant with MAIF Provenance

A multi-step research assistant with full source tracking and verification.

## Pipeline

```
Query → Search → Analyze → Verify → Synthesize → Report
                    ↓
              MAIF Artifact (audit trail)
```

## Features

- **Source Search**: Find relevant documents
- **Analysis**: Extract key findings
- **Verification**: Cross-reference claims
- **Synthesis**: Generate cited summaries
- **Full Provenance**: Every step logged

## Quick Start

```bash
pip install maif langgraph
python main.py
```

## Example

```
Research Query: climate change impacts

[SEARCH] Found 2 sources
[ANALYZE] Extracted 4 findings
[VERIFY] 3/4 findings verified
[SYNTHESIZE] Generated summary

## Research Summary: climate change impacts

**Sources Analyzed:** 2
**Key Findings:** 4

### Findings:
1. Global temperatures have risen 1.1°C above pre-industrial levels (High)
2. Arctic sea ice has declined by 13% per decade since 1979 (High)
...
```

