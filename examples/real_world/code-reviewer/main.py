#!/usr/bin/env python3
"""
AI Code Reviewer with MAIF Provenance

An automated code review system that demonstrates:
- Multi-aspect code analysis (security, style, performance)
- Actionable feedback generation
- Review audit trail for compliance
- Integration-ready architecture

Usage:
    python main.py [file_to_review.py]

Requirements:
    pip install maif langgraph
"""

import os
import sys
import re
from pathlib import Path
from typing import TypedDict, List, Annotated, Optional
from operator import add
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from langgraph.graph import StateGraph, START, END
from maif.integrations.langgraph import MAIFCheckpointer


# =============================================================================
# Types
# =============================================================================

class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


class ReviewCategory(str, Enum):
    SECURITY = "security"
    STYLE = "style"
    PERFORMANCE = "performance"
    BEST_PRACTICES = "best_practices"
    DOCUMENTATION = "documentation"


class Issue(TypedDict):
    line: Optional[int]
    severity: str
    category: str
    message: str
    suggestion: Optional[str]


class ReviewState(TypedDict):
    """State for code review workflow."""
    code: str
    filename: str
    language: str
    issues: Annotated[List[Issue], add]
    summary: str
    score: float
    review_log: Annotated[List[dict], add]


# =============================================================================
# Review Agents
# =============================================================================

def security_reviewer(state: ReviewState) -> ReviewState:
    """Check for security vulnerabilities."""
    code = state["code"]
    issues = []
    
    # Security patterns to check
    security_checks = [
        (r"eval\s*\(", "Use of eval() is dangerous - consider safer alternatives", Severity.ERROR),
        (r"exec\s*\(", "Use of exec() is dangerous - consider safer alternatives", Severity.ERROR),
        (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password detected", Severity.ERROR),
        (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key detected", Severity.ERROR),
        (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret detected", Severity.ERROR),
        (r"import\s+pickle", "pickle can execute arbitrary code - use json for untrusted data", Severity.WARNING),
        (r"subprocess\.call\(.*shell\s*=\s*True", "shell=True is dangerous with untrusted input", Severity.WARNING),
        (r"os\.system\(", "os.system() is vulnerable to injection - use subprocess with shell=False", Severity.WARNING),
        (r"SELECT.*\+.*\+", "Possible SQL injection - use parameterized queries", Severity.ERROR),
        (r"MD5|SHA1", "Weak hash algorithm - use SHA256 or better", Severity.WARNING),
    ]
    
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        for pattern, message, severity in security_checks:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append({
                    "line": i,
                    "severity": severity.value,
                    "category": ReviewCategory.SECURITY.value,
                    "message": message,
                    "suggestion": None,
                })
    
    print(f"[SECURITY] Found {len(issues)} security issues")
    
    return {
        "issues": issues,
        "code": "",
        "filename": "",
        "language": "",
        "summary": "",
        "score": 0.0,
        "review_log": [{
            "agent": "security",
            "issues_found": len(issues),
            "timestamp": datetime.now().isoformat(),
        }],
    }


def style_reviewer(state: ReviewState) -> ReviewState:
    """Check code style and formatting."""
    code = state["code"]
    issues = []
    
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        # Line length
        if len(line) > 100:
            issues.append({
                "line": i,
                "severity": Severity.WARNING.value,
                "category": ReviewCategory.STYLE.value,
                "message": f"Line exceeds 100 characters ({len(line)} chars)",
                "suggestion": "Break into multiple lines",
            })
        
        # Trailing whitespace
        if line.rstrip() != line:
            issues.append({
                "line": i,
                "severity": Severity.INFO.value,
                "category": ReviewCategory.STYLE.value,
                "message": "Trailing whitespace",
                "suggestion": "Remove trailing whitespace",
            })
        
        # TODO/FIXME comments
        if "TODO" in line or "FIXME" in line:
            issues.append({
                "line": i,
                "severity": Severity.INFO.value,
                "category": ReviewCategory.STYLE.value,
                "message": "TODO/FIXME comment found",
                "suggestion": "Address or create a ticket",
            })
    
    # Missing docstring check
    if "def " in code and '"""' not in code and "'''" not in code:
        issues.append({
            "line": 1,
            "severity": Severity.SUGGESTION.value,
            "category": ReviewCategory.DOCUMENTATION.value,
            "message": "Functions appear to lack docstrings",
            "suggestion": "Add docstrings to document function purpose and parameters",
        })
    
    print(f"[STYLE] Found {len(issues)} style issues")
    
    return {
        "issues": issues,
        "code": "",
        "filename": "",
        "language": "",
        "summary": "",
        "score": 0.0,
        "review_log": [{
            "agent": "style",
            "issues_found": len(issues),
            "timestamp": datetime.now().isoformat(),
        }],
    }


def performance_reviewer(state: ReviewState) -> ReviewState:
    """Check for performance issues."""
    code = state["code"]
    issues = []
    
    # Performance patterns
    perf_checks = [
        (r"for.*in\s+range\(len\(", "Use enumerate() instead of range(len())", Severity.SUGGESTION),
        (r"\+\s*=.*\+\s*=.*\+\s*=", "Multiple string concatenations - consider using join()", Severity.WARNING),
        (r"import\s+\*", "Wildcard imports can slow startup and cause conflicts", Severity.WARNING),
        (r"\.append\(.*for.*in", "Consider using list comprehension for better performance", Severity.SUGGESTION),
        (r"time\.sleep\(\d+\)", "Blocking sleep - consider async alternatives", Severity.INFO),
        (r"global\s+\w+", "Global variables can cause performance issues", Severity.WARNING),
    ]
    
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        for pattern, message, severity in perf_checks:
            if re.search(pattern, line):
                issues.append({
                    "line": i,
                    "severity": severity.value,
                    "category": ReviewCategory.PERFORMANCE.value,
                    "message": message,
                    "suggestion": None,
                })
    
    print(f"[PERFORMANCE] Found {len(issues)} performance issues")
    
    return {
        "issues": issues,
        "code": "",
        "filename": "",
        "language": "",
        "summary": "",
        "score": 0.0,
        "review_log": [{
            "agent": "performance",
            "issues_found": len(issues),
            "timestamp": datetime.now().isoformat(),
        }],
    }


def summarize_review(state: ReviewState) -> ReviewState:
    """Generate review summary and score."""
    issues = state["issues"]
    
    # Calculate score
    error_count = sum(1 for i in issues if i["severity"] == Severity.ERROR.value)
    warning_count = sum(1 for i in issues if i["severity"] == Severity.WARNING.value)
    info_count = sum(1 for i in issues if i["severity"] in [Severity.INFO.value, Severity.SUGGESTION.value])
    
    # Score: 100 - (errors * 20) - (warnings * 5) - (info * 1)
    score = max(0, 100 - (error_count * 20) - (warning_count * 5) - (info_count * 1))
    
    # Generate summary
    summary_parts = [
        f"# Code Review Summary",
        f"\n**File:** {state['filename']}",
        f"**Score:** {score}/100",
        f"\n## Issues Found: {len(issues)}",
        f"- Errors: {error_count}",
        f"- Warnings: {warning_count}",
        f"- Info/Suggestions: {info_count}",
    ]
    
    # Group by category
    by_category = {}
    for issue in issues:
        cat = issue["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(issue)
    
    for category, cat_issues in by_category.items():
        summary_parts.append(f"\n### {category.replace('_', ' ').title()}")
        for issue in cat_issues[:5]:  # Top 5 per category
            line_info = f"Line {issue['line']}: " if issue['line'] else ""
            summary_parts.append(f"- [{issue['severity'].upper()}] {line_info}{issue['message']}")
    
    summary = "\n".join(summary_parts)
    
    print(f"[SUMMARY] Score: {score}/100")
    
    return {
        "summary": summary,
        "score": score,
        "issues": [],
        "code": "",
        "filename": "",
        "language": "",
        "review_log": [{
            "agent": "summarize",
            "score": score,
            "total_issues": len(issues),
            "timestamp": datetime.now().isoformat(),
        }],
    }


# =============================================================================
# Build Graph
# =============================================================================

def create_code_reviewer(artifact_path: str = "code_review.maif"):
    """Create the code review graph."""
    
    graph = StateGraph(ReviewState)
    
    graph.add_node("security", security_reviewer)
    graph.add_node("style", style_reviewer)
    graph.add_node("performance", performance_reviewer)
    graph.add_node("summarize", summarize_review)
    
    # Parallel review (conceptually - runs sequentially in this impl)
    graph.add_edge(START, "security")
    graph.add_edge("security", "style")
    graph.add_edge("style", "performance")
    graph.add_edge("performance", "summarize")
    graph.add_edge("summarize", END)
    
    checkpointer = MAIFCheckpointer(artifact_path, agent_id="code_reviewer")
    app = graph.compile(checkpointer=checkpointer)
    
    return app, checkpointer


# =============================================================================
# Main
# =============================================================================

SAMPLE_CODE = '''
import pickle
from os import system

password = "super_secret_123"
api_key = "sk-1234567890"

def process_data(data):
    result = ""
    for i in range(len(data)):
        result += str(data[i])
    eval(data)
    return result

# TODO: fix this later
def query_db(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    return query
'''


def main():
    print("=" * 60)
    print("AI Code Reviewer with MAIF Provenance")
    print("=" * 60)
    print()
    
    # Check for file argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            with open(filepath) as f:
                code = f.read()
            filename = os.path.basename(filepath)
        else:
            print(f"File not found: {filepath}")
            return
    else:
        print("No file provided, using sample code with intentional issues...")
        code = SAMPLE_CODE
        filename = "sample.py"
    
    print(f"\nReviewing: {filename}")
    print("-" * 60)
    
    app, checkpointer = create_code_reviewer()
    config = {"configurable": {"thread_id": f"review-{filename}"}}
    
    result = app.invoke(
        {
            "code": code,
            "filename": filename,
            "language": "python",
            "issues": [],
            "summary": "",
            "score": 0.0,
            "review_log": [],
        },
        config=config
    )
    
    print("\n" + "=" * 60)
    print(result["summary"])
    print("=" * 60)
    
    # Finalize
    checkpointer.finalize()
    
    print(f"\nReview saved to: {checkpointer.get_artifact_path()}")
    print(f"Review score: {result['score']}/100")


if __name__ == "__main__":
    main()

