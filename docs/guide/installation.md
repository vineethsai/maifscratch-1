# Installation

This guide covers installing MAIF and verifying your setup.

## Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **pip** package manager
- **Git** (for development installation)

## Installation Methods

### Development Installation (Recommended)

For development and full access to all features, clone the repository and install in editable mode:

```bash
# Clone the repository
git clone https://github.com/vineethsai/maifscratch-1.git
cd maifscratch-1

# Install in editable mode with all dependencies
pip install -e .
```

### Install with Optional Dependencies

MAIF supports various optional features through extras:

```bash
# Install with full feature set (recommended)
pip install -e ".[full]"

# Install specific feature sets
pip install -e ".[ml]"          # Machine learning features
pip install -e ".[vision]"      # Image/video processing
pip install -e ".[compression]" # Advanced compression
pip install -e ".[dev]"         # Development tools
```

**Available extras:**
- `full`: All optional dependencies
- `ml`: sentence-transformers, faiss-cpu, scipy
- `vision`: opencv-python, pillow
- `compression`: brotli, zstandard, lz4
- `async`: aiofiles
- `cli`: click, tqdm
- `monitoring`: prometheus-client
- `dev`: pytest, black, mypy, flake8

## Verifying Installation

### Basic Verification

```python
# Verify MAIF is installed
import maif
print(f"MAIF version: {maif.__version__}")

# Verify core components are available
from maif.core import MAIFEncoder, MAIFDecoder
from maif.security import MAIFSigner, MAIFVerifier
from maif.privacy import PrivacyEngine

print("Core components loaded successfully!")
```

### Test Simple API

```python
# Test the simple MAIF API
from maif_api import MAIF, create_maif

# Create a new MAIF file
maif = create_maif("test-agent")
maif.add_text("Hello, MAIF!", title="Test")
maif.save("test.maif")

print("MAIF file created successfully!")
```

### Verify All Components

```python
# Comprehensive verification
import maif

# Check core functionality
print("Checking core components...")
from maif.core import MAIFEncoder, MAIFDecoder
print("  ✓ MAIFEncoder, MAIFDecoder")

# Check security
print("Checking security...")
from maif.security import MAIFSigner, MAIFVerifier
print("  ✓ MAIFSigner, MAIFVerifier")

# Check privacy
print("Checking privacy...")
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode
print("  ✓ PrivacyEngine, PrivacyLevel, EncryptionMode")

# Check semantic features
print("Checking semantic features...")
from maif.semantic import SemanticEmbedder, CrossModalAttention
print("  ✓ SemanticEmbedder, CrossModalAttention")

# Check streaming
print("Checking streaming...")
from maif.streaming import MAIFStreamReader, MAIFStreamWriter
print("  ✓ MAIFStreamReader, MAIFStreamWriter")

# Check simple API
print("Checking simple API...")
from maif_api import MAIF, create_maif, load_maif
print("  ✓ MAIF, create_maif, load_maif")

print("\n✅ All components verified successfully!")
```

## Platform-Specific Notes

### macOS

```bash
# Install with Homebrew dependencies if needed
brew install cmake libomp

# Then install MAIF
pip install -e ".[full]"
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

# Then install MAIF
pip install -e ".[full]"
```

### Windows

```bash
# Install Visual C++ Build Tools if needed
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then install MAIF
pip install -e ".[full]"
```

## Virtual Environment (Recommended)

Using a virtual environment is strongly recommended:

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Install MAIF
pip install -e ".[full]"
```

## Troubleshooting

### Common Issues

**Import Error: Module not found**
```bash
# Ensure you're in the project directory and installed correctly
cd maifscratch-1
pip install -e .
```

**Cryptography build errors**
```bash
# Install build dependencies
pip install --upgrade pip setuptools wheel
pip install cryptography
```

**NumPy version conflicts**
```bash
# Reinstall NumPy with compatible version
pip install numpy>=1.21.0
```

**Sentence-transformers issues**
```bash
# Install PyTorch first
pip install torch
pip install sentence-transformers
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/vineethsai/maifscratch-1/issues)
2. Ensure Python version is 3.8+
3. Try installing in a fresh virtual environment

## Next Steps

- **[Quick Start →](/guide/quick-start)** - Create your first MAIF file
- **[Getting Started →](/guide/getting-started)** - Learn the basics
- **[API Reference →](/api/)** - Detailed API documentation
