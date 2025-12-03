"""
Setup script for MAIF library and SDK.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version from __init__.py
def get_version():
    version_file = os.path.join("maif", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "2.0.0"

setup(
    name="maif",
    version=get_version(),
    author="MAIF Development Team",
    author_email="dev@maif.ai",
    description="Multimodal Artifact File Format - Production-ready AI-native container with trustworthy AI capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maif-ai/maif",
    packages=find_packages(include=["maif", "maif.*"]),
    py_modules=["maif_api"],  # Include the simple API module
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Archiving",
        "Topic :: Software Development :: Version Control",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "cryptography>=3.4.8",
        "pydantic>=2.0.0",
        "structlog>=23.0.0",
        "tenacity>=8.0.0",
    ],
    extras_require={
        "full": [
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "opencv-python>=4.5.0",
            "aiofiles>=0.8.0",
            "brotli>=1.0.9",
            "zstandard>=0.18.0",
            "xxhash>=3.0.0",
            "msgpack>=1.0.4",
            "jsonschema>=4.0.0",
            "click>=8.0.0",
            "tqdm>=4.64.0",
            "psutil>=5.8.0",
            "networkx>=2.6.0",
            "matplotlib>=3.4.0",
            "pillow>=8.3.0",
            "scipy>=1.7.0",
            "lz4>=4.0.0",
            "numba>=0.56.0",
            "prometheus-client>=0.16.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "pytest-asyncio>=0.21.0",
            "black>=21.0.0",
            "mypy>=1.0.0",
            "flake8>=3.9.0",
            "pre-commit>=2.15.0",
        ],
        "compression": ["brotli>=1.0.9", "zstandard>=0.18.0", "lz4>=4.0.0"],
        "async": ["aiofiles>=0.8.0"],
        "cli": ["click>=8.0.0", "tqdm>=4.64.0"],
        "validation": ["jsonschema>=4.0.0"],
        "performance": ["xxhash>=3.0.0", "msgpack>=1.0.4", "psutil>=5.8.0", "numba>=0.56.0"],
        "ml": ["sentence-transformers>=2.2.0", "faiss-cpu>=1.7.0", "scipy>=1.7.0"],
        "vision": ["opencv-python>=4.5.0", "pillow>=8.3.0"],
        "monitoring": ["prometheus-client>=0.16.0"],
    },
    entry_points={
        "console_scripts": [
            "maif=maif.cli:main",
            "maif-create=maif.cli:create_maif",
            "maif-verify=maif.cli:verify_maif",
            "maif-analyze=maif.cli:analyze_maif",
            "maif-extract=maif.cli:extract_content",
        ],
    },
    include_package_data=True,
    package_data={
        "maif": ["*.json", "*.yaml", "schemas/*.json", "templates/*.yaml"],
    },
    keywords="ai, multimodal, security, provenance, forensics, trustworthy-ai, file-format, compression, streaming, aws, cloud, production",
    project_urls={
        "Bug Reports": "https://github.com/maif-ai/maif/issues",
        "Source": "https://github.com/maif-ai/maif",
        "Documentation": "https://maif.readthedocs.io/",
        "Changelog": "https://github.com/maif-ai/maif/blob/main/CHANGELOG.md",
    },
)