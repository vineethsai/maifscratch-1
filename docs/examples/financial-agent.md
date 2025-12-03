# Financial AI Agent

This example demonstrates building a privacy-compliant financial AI agent with MAIF for transaction analysis, risk assessment, and audit trails.

## Features Demonstrated

- **Privacy-by-Design**: Automatic encryption and anonymization
- **Audit Trails**: Complete transaction logging
- **Risk Assessment**: Pattern-based fraud detection
- **Compliance Ready**: GDPR/PCI DSS patterns

## Code Example

```python
from maif_api import create_maif, load_maif
from maif.privacy import PrivacyEngine
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
from typing import Dict, List, Optional
import os

@dataclass
class Transaction:
    """Financial transaction data."""
    transaction_id: str
    account_id: str
    amount: Decimal
    currency: str
    transaction_type: str  # 'debit', 'credit', 'transfer'
    timestamp: datetime
    merchant_id: Optional[str] = None
    description: Optional[str] = None

@dataclass
class RiskAssessment:
    """Risk assessment result."""
    transaction_id: str
    risk_score: float  # 0.0 to 1.0
    risk_factors: List[str]
    recommendation: str  # 'approve', 'review', 'decline'

class FinancialAgent:
    """Privacy-compliant financial AI agent."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.data_dir = f"financial_data/{agent_id}"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create memory stores with privacy
        self.transactions = create_maif(
            f"{agent_id}-transactions", 
            enable_privacy=True
        )
        self.audit_log = create_maif(
            f"{agent_id}-audit",
            enable_privacy=True
        )
        self.risk_models = create_maif(f"{agent_id}-risk")
        
        # Statistics
        self.stats = {
            "transactions_processed": 0,
            "fraud_detected": 0
        }
    
    def process_transaction(self, tx: Transaction) -> RiskAssessment:
        """Process a transaction with full audit trail."""
        
        # 1. Store transaction with encryption
        self._store_transaction(tx)
        
        # 2. Assess risk
        risk = self._assess_risk(tx)
        
        # 3. Log to audit trail
        self._log_audit(tx, risk)
        
        # 4. Update stats
        self.stats["transactions_processed"] += 1
        if risk.recommendation == "decline":
            self.stats["fraud_detected"] += 1
        
        return risk
    
    def _store_transaction(self, tx: Transaction):
        """Store transaction with privacy protection."""
        self.transactions.add_text(
            f"TX: {tx.transaction_id} | "
            f"Account: {tx.account_id} | "
            f"Amount: {tx.amount} {tx.currency} | "
            f"Type: {tx.transaction_type}",
            title=f"Transaction {tx.transaction_id}",
            encrypt=True,
            anonymize=True  # Anonymize account ID
        )
    
    def _assess_risk(self, tx: Transaction) -> RiskAssessment:
        """Simple rule-based risk assessment."""
        risk_score = 0.0
        risk_factors = []
        
        # High value check
        if tx.amount > 10000:
            risk_score += 0.3
            risk_factors.append("high_value")
        
        # Unusual timing (outside business hours)
        if tx.timestamp.hour < 6 or tx.timestamp.hour > 22:
            risk_score += 0.2
            risk_factors.append("unusual_timing")
        
        # Transfer type gets extra scrutiny
        if tx.transaction_type == "transfer":
            risk_score += 0.1
            risk_factors.append("transfer_type")
        
        # Determine recommendation
        if risk_score < 0.3:
            recommendation = "approve"
        elif risk_score < 0.6:
            recommendation = "review"
        else:
            recommendation = "decline"
        
        return RiskAssessment(
            transaction_id=tx.transaction_id,
            risk_score=min(risk_score, 1.0),
            risk_factors=risk_factors,
            recommendation=recommendation
        )
    
    def _log_audit(self, tx: Transaction, risk: RiskAssessment):
        """Create audit trail entry."""
        self.audit_log.add_text(
            f"[{datetime.now().isoformat()}] "
            f"TX: {tx.transaction_id} | "
            f"Risk: {risk.risk_score:.2f} | "
            f"Decision: {risk.recommendation} | "
            f"Factors: {', '.join(risk.risk_factors)}",
            title=f"Audit: {tx.transaction_id}"
        )
    
    def search_transactions(self, query: str, top_k: int = 5) -> list:
        """Search transaction history."""
        return self.transactions.search(query, top_k=top_k)
    
    def get_audit_trail(self) -> list:
        """Get complete audit trail."""
        return self.audit_log.get_content_list()
    
    def save(self):
        """Save all agent data."""
        self.transactions.save(
            f"{self.data_dir}/transactions.maif", 
            sign=True
        )
        self.audit_log.save(
            f"{self.data_dir}/audit.maif", 
            sign=True
        )
        self.risk_models.save(
            f"{self.data_dir}/risk.maif", 
            sign=True
        )
        print(f"Agent data saved to {self.data_dir}")
    
    def load(self):
        """Load existing agent data."""
        tx_path = f"{self.data_dir}/transactions.maif"
        audit_path = f"{self.data_dir}/audit.maif"
        
        if os.path.exists(tx_path):
            self.transactions = load_maif(tx_path)
        if os.path.exists(audit_path):
            self.audit_log = load_maif(audit_path)

# Usage
def main():
    agent = FinancialAgent("fintech-001")
    
    # Process sample transactions
    transactions = [
        Transaction(
            transaction_id="tx_001",
            account_id="acc_12345",
            amount=Decimal("150.00"),
            currency="USD",
            transaction_type="debit",
            timestamp=datetime.now(),
            merchant_id="merch_001",
            description="Online purchase"
        ),
        Transaction(
            transaction_id="tx_002",
            account_id="acc_12345",
            amount=Decimal("25000.00"),  # High value
            currency="USD",
            transaction_type="transfer",
            timestamp=datetime.now(),
            description="Wire transfer"
        )
    ]
    
    for tx in transactions:
        risk = agent.process_transaction(tx)
        print(f"\nTransaction {tx.transaction_id}:")
        print(f"  Amount: {tx.amount} {tx.currency}")
        print(f"  Risk Score: {risk.risk_score:.2f}")
        print(f"  Recommendation: {risk.recommendation}")
        print(f"  Factors: {', '.join(risk.risk_factors) or 'None'}")
    
    # Save agent state
    agent.save()
    
    # Show audit trail
    print("\n📋 Audit Trail:")
    for entry in agent.get_audit_trail():
        print(f"  {entry.get('title', 'Entry')}")

if __name__ == "__main__":
    main()
```

## What You'll Learn

- Privacy-compliant data handling
- Transaction audit trails
- Risk assessment patterns
- Encrypted storage with MAIF

## Running the Example

```bash
cd maifscratch-1
pip install -e .
python3 -c "
# Paste the code above and run main()
"
```

## Key Patterns

### 1. Privacy Protection

```python
# Encrypt sensitive data
artifact.add_text(
    "Account: 1234-5678",
    encrypt=True,      # AES-GCM encryption
    anonymize=True     # Remove PII
)
```

### 2. Audit Trails

```python
# Log every action
audit.add_text(
    f"[{timestamp}] Action: {action} | Result: {result}",
    title=f"Audit Entry"
)
```

### 3. Integrity Verification

```python
# Verify data hasn't been tampered
loaded = load_maif("financial_data.maif")
if loaded.verify_integrity():
    print("✅ Data is authentic")
```

## Next Steps

- [Privacy Guide](/guide/privacy) - Advanced privacy features
- [Security Model](/guide/security-model) - Cryptographic details
- [LangGraph Example](/examples/langgraph-rag) - Multi-agent system
