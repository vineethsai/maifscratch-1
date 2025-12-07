"""
Access Control Simulation Demo.

Demonstrates role-based access control with MAIF audit logging.
"""

import os
import time
from pathlib import Path
from datetime import datetime

# Terminal formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def wait_for_enter(message="Press Enter to continue..."):
    """Wait for user to press Enter."""
    input(f"\n{DIM}{message}{RESET}")


# Role definitions with permissions
ROLES = {
    "analyst": {
        "description": "Read financial summaries, no PII access",
        "permissions": {
            "financial_summary": True,
            "reports": True,
            "customer_pii": False,
            "raw_transactions": False,
            "admin_settings": False,
        },
    },
    "manager": {
        "description": "Read all financial data, limited PII",
        "permissions": {
            "financial_summary": True,
            "reports": True,
            "customer_pii": True,
            "raw_transactions": True,
            "admin_settings": False,
        },
    },
    "admin": {
        "description": "Full access to all data",
        "permissions": {
            "financial_summary": True,
            "reports": True,
            "customer_pii": True,
            "raw_transactions": True,
            "admin_settings": True,
        },
    },
    "auditor": {
        "description": "Read-only access to audit logs only",
        "permissions": {
            "financial_summary": False,
            "reports": True,
            "customer_pii": False,
            "raw_transactions": False,
            "admin_settings": False,
        },
    },
}

# Sample queries with their required permissions
SAMPLE_QUERIES = [
    {
        "query": "Show me Q3 revenue summary",
        "resource": "financial_summary",
        "description": "Financial summary data",
    },
    {
        "query": "Show me customer John Smith's transaction history",
        "resource": "customer_pii",
        "description": "Customer PII data",
    },
    {
        "query": "Generate compliance report for last month",
        "resource": "reports",
        "description": "Report generation",
    },
    {
        "query": "Show all raw transaction logs",
        "resource": "raw_transactions",
        "description": "Raw transaction data",
    },
    {
        "query": "Change system configuration",
        "resource": "admin_settings",
        "description": "Admin settings",
    },
]


class AccessControlDemo:
    """Interactive access control demonstration."""
    
    def __init__(self, sessions_dir: Path):
        """Initialize the demo.
        
        Args:
            sessions_dir: Directory for session artifacts
        """
        self.sessions_dir = sessions_dir
        self.current_role = None
        self.artifact_path = None
        self.checkpointer = None
        self.access_log = []
    
    def run(self):
        """Run the access control simulation."""
        # Select role
        self.current_role = self._select_role()
        if not self.current_role:
            return
        
        # Initialize MAIF artifact for audit logging
        self._init_artifact()
        
        # Run the simulation
        self._run_simulation()
        
        # Show summary
        self._show_summary()
    
    def _select_role(self) -> str:
        """Prompt user to select a role."""
        clear_screen()
        print()
        print(f"{BOLD}ACCESS CONTROL SIMULATION{RESET}")
        print("-" * 80)
        print()
        print(f"{BOLD}AVAILABLE ROLES{RESET}")
        print()
        
        role_list = list(ROLES.keys())
        for i, role in enumerate(role_list, 1):
            info = ROLES[role]
            print(f"[{i}] {role:<12} - {info['description']}")
        
        print()
        choice = input("Select role: ").strip()
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(role_list):
                return role_list[idx - 1]
        except ValueError:
            pass
        
        return None
    
    def _init_artifact(self):
        """Initialize the MAIF artifact for audit logging."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.artifact_path = self.sessions_dir / f"access_control_{timestamp}.maif"
        
        from maif.integrations.langgraph import MAIFCheckpointer
        
        self.checkpointer = MAIFCheckpointer(
            artifact_path=str(self.artifact_path),
            agent_id="access_control_demo"
        )
    
    def _run_simulation(self):
        """Run the access control simulation."""
        clear_screen()
        print()
        print(f"{BOLD}ACCESS CONTROL SIMULATION{RESET}")
        print("-" * 80)
        print()
        
        permissions = ROLES[self.current_role]["permissions"]
        
        # Show current role info
        print(f"Logged in as: {CYAN}{self.current_role}{RESET}")
        print(f"Permissions: {DIM}", end="")
        granted = [k for k, v in permissions.items() if v]
        print(", ".join(granted) if granted else "none")
        print(f"{RESET}Restrictions: {DIM}", end="")
        denied = [k for k, v in permissions.items() if not v]
        print(", ".join(denied) if denied else "none")
        print(f"{RESET}")
        print("-" * 80)
        print()
        print(f"{DIM}Type queries to test access control. Type 'quit' to exit.{RESET}")
        print(f"{DIM}Try: 'show revenue', 'customer data', 'transaction logs', 'admin settings'{RESET}")
        print()
        
        while True:
            try:
                user_input = input(f"{GREEN}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                break
            
            # Process the query
            self._process_query(user_input)
            print()
        
        # Finalize the artifact
        self.checkpointer.finalize()
    
    def _process_query(self, query: str):
        """Process a user query and check access control."""
        query_lower = query.lower()
        
        # Determine required resource
        resource = self._determine_resource(query_lower)
        
        # Check permission
        permissions = ROLES[self.current_role]["permissions"]
        has_permission = permissions.get(resource, False)
        
        # Log the access attempt
        log_entry = {
            "timestamp": time.time(),
            "user_role": self.current_role,
            "query": query,
            "resource": resource,
            "granted": has_permission,
        }
        self.access_log.append(log_entry)
        
        # Log to MAIF
        self._log_to_maif(log_entry)
        
        # Display result
        print()
        print(f"{DIM}[ACCESS CONTROL] Checking permissions...{RESET}")
        print(f"{DIM}[ACCESS CONTROL] Resource: {resource}{RESET}")
        print(f"{DIM}[ACCESS CONTROL] Required: {resource}:read{RESET}")
        print(f"{DIM}[ACCESS CONTROL] User has: {resource}:{'read' if has_permission else 'denied'}{RESET}")
        print()
        
        if has_permission:
            print(f"{GREEN}[ACCESS CONTROL] GRANTED{RESET}")
            print()
            self._show_mock_response(resource)
        else:
            print("-" * 80)
            print(f"{RED}{BOLD}ACCESS DENIED{RESET}")
            print("-" * 80)
            print(f"Your role ({self.current_role}) does not have permission to access {resource}.")
            print(f"This access attempt has been logged to the audit trail.")
            print()
            print(f"{DIM}Logged Event:{RESET}")
            print(f"  - Type: access_denied")
            print(f"  - User: {self.current_role}")
            print(f"  - Resource: {resource}")
            print(f"  - Query: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
            print(f"  - Timestamp: {datetime.now().isoformat()}")
    
    def _determine_resource(self, query: str) -> str:
        """Determine the required resource based on query content."""
        if any(word in query for word in ["customer", "client", "user", "pii", "personal"]):
            return "customer_pii"
        elif any(word in query for word in ["transaction", "raw", "log"]):
            return "raw_transactions"
        elif any(word in query for word in ["admin", "config", "setting", "system"]):
            return "admin_settings"
        elif any(word in query for word in ["report", "compliance", "audit"]):
            return "reports"
        else:
            return "financial_summary"
    
    def _show_mock_response(self, resource: str):
        """Show a mock response for the granted access."""
        responses = {
            "financial_summary": "Q3 2024 Revenue: $42.5M (+15% YoY)\nNet Profit: $8.2M (19.3% margin)",
            "reports": "Generating compliance report...\nReport ready: 47 pages, 0 violations found.",
            "customer_pii": "Customer: John Smith\nAccount: ****4521\nEmail: j***@example.com",
            "raw_transactions": "Transaction Log (last 10):\n- TXN001: $150.00 - 2024-12-01\n- TXN002: $89.50 - 2024-12-02\n...",
            "admin_settings": "System Configuration:\n- Max concurrent users: 100\n- Session timeout: 30 min\n- Audit level: FULL",
        }
        
        print(f"{BLUE}[RESPONSE]{RESET}")
        print(responses.get(resource, "Data retrieved successfully."))
    
    def _log_to_maif(self, log_entry: dict):
        """Log access attempt to MAIF artifact."""
        config = {"configurable": {"thread_id": "access_control", "checkpoint_ns": ""}}
        
        checkpoint = {
            "v": 1,
            "id": f"access_{int(log_entry['timestamp'] * 1000)}",
            "ts": datetime.fromtimestamp(log_entry["timestamp"]).isoformat(),
            "channel_values": log_entry,
            "channel_versions": {},
            "versions_seen": {},
        }
        
        metadata = {
            "source": "access_control",
            "action": "access_granted" if log_entry["granted"] else "access_denied",
        }
        
        self.checkpointer.put(config, checkpoint, metadata)
    
    def _show_summary(self):
        """Show summary of the access control simulation."""
        print()
        print("-" * 80)
        print(f"{BOLD}SESSION SUMMARY{RESET}")
        print("-" * 80)
        print()
        print(f"Role: {self.current_role}")
        print(f"Total access attempts: {len(self.access_log)}")
        
        granted = sum(1 for e in self.access_log if e["granted"])
        denied = len(self.access_log) - granted
        
        print(f"  Granted: {GREEN}{granted}{RESET}")
        print(f"  Denied: {RED}{denied}{RESET}")
        print()
        print(f"Audit log saved to: {self.artifact_path}")
        print()
        print(f"{GREEN}All access attempts have been cryptographically logged.{RESET}")
        
        wait_for_enter()

