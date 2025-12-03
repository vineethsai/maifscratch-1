# AWS Integration (Optional)

## Note

AWS integration has been removed to keep the core library lightweight.

## If You Need AWS

MAIF core functionality works without AWS. If you need AWS features:

### Option 1: Use boto3 Directly
```python
import boto3
from maif_api import create_maif

# Use MAIF for local artifacts
maif = create_maif("agent")
maif.add_text("data")
maif.save("local.maif")

# Use boto3 for AWS operations
s3 = boto3.client('s3')
s3.upload_file("local.maif", "my-bucket", "artifacts/local.maif")
```

### Option 2: Fork and Add AWS
The previous version had extensive AWS integration. You can:
1. Check git history for AWS modules
2. Add back what you need
3. Install: `pip install boto3 botocore`

### What Was Removed:
- 17 AWS integration modules (`aws_*.py`)
- 14 AWS example scripts
- CloudFormation templates
- AWS-specific tests

### Why Removed:
- Simplifies core library
- Reduces dependencies
- Easier to maintain
- Most users don't need AWS

## Recommendations

**For S3 storage:**
```python
# Just use boto3 directly
import boto3
s3 = boto3.client('s3')
s3.upload_file("artifact.maif", "bucket", "key")
```

**For KMS encryption:**
```python
# Use MAIF's built-in crypto (no AWS needed)
from maif.security import MAIFSigner
signer = MAIFSigner()
encrypted = signer.encrypt_data(data)
```

**For Bedrock:**
```python
# Use Gemini, OpenAI, or Anthropic directly
# See examples/langgraph/ for Gemini integration
```

The core MAIF library provides everything needed for cryptographic provenance without AWS dependencies.

