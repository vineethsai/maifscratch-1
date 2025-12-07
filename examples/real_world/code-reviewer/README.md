# AI Code Reviewer with MAIF Provenance

An automated code review system with multi-aspect analysis and full audit trail.

## Features

- **Security Analysis**: Detect vulnerabilities, hardcoded secrets, injection risks
- **Style Review**: Line length, formatting, documentation
- **Performance Check**: Inefficient patterns, blocking operations
- **Audit Trail**: Every review decision logged with cryptographic provenance

## Quick Start

```bash
pip install maif langgraph
python main.py                    # Review sample code
python main.py your_file.py       # Review your code
```

## Example Output

```
AI Code Reviewer with MAIF Provenance
============================================================

Reviewing: sample.py

[SECURITY] Found 5 security issues
[STYLE] Found 3 style issues
[PERFORMANCE] Found 2 performance issues
[SUMMARY] Score: 35/100

# Code Review Summary

**File:** sample.py
**Score:** 35/100

## Issues Found: 10
- Errors: 3
- Warnings: 5
- Info/Suggestions: 2

### Security
- [ERROR] Line 4: Hardcoded password detected
- [ERROR] Line 5: Hardcoded API key detected
- [ERROR] Line 10: Use of eval() is dangerous
```

## Compliance

All review decisions are cryptographically signed, suitable for:
- Code review audit requirements
- Compliance documentation
- Review history tracking

