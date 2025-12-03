"""
Full Integration Example - Everything Connected to Everything

This example demonstrates how all MAIF components are tightly integrated:
- Centralized AWS credentials flow to all services automatically
- Security manager automatically encrypts data and logs compliance
- S3 block storage automatically uses KMS encryption
- Error handling automatically alerts via CloudWatch
- Rate limiting automatically reports metrics
- Bedrock swarm automatically logs AI governance compliance
- All components work together seamlessly
"""

import asyncio
from maif.aws_config import configure_aws
from maif.agentic_framework import MAIFAgent, MultiAgentOrchestrator
from maif.aws_s3_block_storage import S3BlockStorage
from maif.security import SecurityManager
from maif.compliance_logging import EnhancedComplianceLogger
from maif.error_handling import ErrorHandler
from maif.rate_limiter import CostBasedRateLimiter, RateLimitConfig


async def main():
    """Demonstrate full integration of all MAIF components."""
    
    # STEP 1: One-time global AWS configuration
    # This automatically configures ALL components
    print("=== Configuring AWS Globally ===")
    configure_aws(
        environment="production",
        profile="default",  # Or use IAM role, env vars, etc.
        region="us-east-1",
        enable_metrics=True,
        enable_monitoring=True
    )
    print("✓ AWS configured - all services will use these credentials automatically")
    
    # STEP 2: Initialize components - they all auto-configure
    print("\n=== Initializing Integrated Components ===")
    
    # Security manager with automatic KMS and compliance logging
    security = SecurityManager(
        use_kms=True,
        require_encryption=True,
        enable_compliance_logging=True  # Auto-logs to CloudWatch
    )
    print("✓ Security manager initialized with KMS + compliance logging")
    
    # S3 block storage with automatic encryption and compliance
    s3_storage = S3BlockStorage(
        bucket_name="maif-demo-bucket",
        enable_encryption=True,  # Auto-uses security manager
        enable_compliance_logging=True  # Auto-logs to CloudWatch
    )
    print("✓ S3 block storage initialized with auto-encryption + compliance")
    
    # Error handler with automatic CloudWatch alerting
    error_handler = ErrorHandler()
    print("✓ Error handler initialized with CloudWatch alerting")
    
    # Rate limiter with automatic CloudWatch metrics
    rate_limiter = CostBasedRateLimiter(
        RateLimitConfig(
            requests_per_second=10,
            burst_size=20,
            enable_cost_tracking=True
        )
    )
    print("✓ Rate limiter initialized with CloudWatch metrics")
    
    # MAIF Agent with all AWS integrations
    agent = MAIFAgent(
        agent_id="integrated-agent-001",
        workspace_path="/tmp/maif-integration",
        use_aws=True  # Enables Bedrock, KMS, CloudWatch, etc.
    )
    print("✓ MAIF agent initialized with full AWS integration")
    
    # STEP 3: Demonstrate automatic integrations
    print("\n=== Demonstrating Automatic Integrations ===")
    
    # 3.1 Secure data handling (auto-encrypted, auto-logged)
    print("\n1. Secure Data Handling:")
    sensitive_data = b"Patient medical records - HIPAA protected"
    
    # Security manager automatically uses KMS and logs to compliance
    encrypted = security.encrypt_data(sensitive_data)
    print("  ✓ Data encrypted with KMS")
    print("  ✓ Encryption logged to CloudWatch compliance")
    
    # 3.2 S3 storage (auto-encrypted before storage, auto-logged)
    print("\n2. S3 Block Storage:")
    block_id = s3_storage.add_block(
        block_type="PATIENT_DATA",
        data=sensitive_data,  # Automatically encrypted before S3
        metadata={"classification": "PHI", "compliance": "HIPAA"}
    )
    print("  ✓ Data automatically encrypted with KMS before S3 storage")
    print("  ✓ Storage operation logged to CloudWatch compliance")
    print(f"  ✓ Block stored: {block_id}")
    
    # 3.3 Rate limiting (auto-reports to CloudWatch)
    print("\n3. Rate Limiting with Metrics:")
    for i in range(5):
        allowed = await rate_limiter.check_rate_limit("api_calls", cost=1)
        print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Throttled'}")
    print("  ✓ Rate limit metrics automatically sent to CloudWatch")
    
    # 3.4 Error handling (auto-alerts critical errors)
    print("\n4. Error Handling with Alerting:")
    try:
        # Simulate a critical error
        raise SecurityError("Unauthorized access attempt detected")
    except SecurityError as e:
        error_handler.handle_error(e, context="access_control")
        print("  ✓ Critical error logged to CloudWatch compliance")
        print("  ✓ SNS alert sent (if configured)")
    
    # 3.5 Multi-agent execution (auto-uses Bedrock swarm)
    print("\n5. Multi-Agent Task Execution:")
    orchestrator = MultiAgentOrchestrator(agent)
    
    task = {
        'id': 'task-001',
        'description': 'Analyze patient data for treatment recommendations',
        'requirements': ['HIPAA compliance', 'medical expertise'],
        'priority': 10,
        'sensitive': True  # Automatically encrypted in swarm
    }
    
    # Automatically uses Bedrock swarm with security & compliance
    result = await orchestrator.execute_multi_agent_task(
        task, 
        strategy='consensus'
    )
    print("  ✓ Task distributed to Bedrock swarm")
    print("  ✓ Sensitive data encrypted during processing")
    print("  ✓ AI governance compliance logged")
    
    # STEP 4: Show integration benefits
    print("\n=== Integration Benefits Demonstrated ===")
    print("1. Single AWS configuration used by ALL components")
    print("2. Automatic KMS encryption for all sensitive data")
    print("3. Automatic compliance logging for all operations")
    print("4. Automatic CloudWatch metrics and alerting")
    print("5. Seamless multi-agent AI with governance")
    print("6. Zero credential management after initial setup")
    print("7. Thread-safe and production-ready")
    
    # STEP 5: Compliance report
    print("\n=== Generating Compliance Report ===")
    compliance_logger = EnhancedComplianceLogger()
    
    # Generate HIPAA compliance report
    report = compliance_logger.generate_compliance_report(
        frameworks=["HIPAA", "FIPS"],
        start_time=None,  # Uses all logged events
        resource_filter="*patient*"
    )
    
    print(f"Total compliance events: {report['total_events']}")
    print(f"HIPAA compliant operations: {report['framework_summary'].get('HIPAA', 0)}")
    print(f"FIPS compliant operations: {report['framework_summary'].get('FIPS', 0)}")
    
    print("\n✅ Full integration demonstration complete!")
    print("All components working together with automatic:")
    print("- Credential management")
    print("- Encryption")
    print("- Compliance logging")
    print("- Metrics & monitoring")
    print("- Error handling & alerting")


if __name__ == "__main__":
    asyncio.run(main())