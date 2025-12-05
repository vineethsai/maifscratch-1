"""
MAIF Integration Module

Contains integration features:
- Framework adapters (LangChain, LlamaIndex, etc.)
- Enhanced processing
"""

try:
    from .integration import MAIFConverter
except ImportError:
    MAIFConverter = None

MAIFIntegration = None  # Legacy alias, not implemented

try:
    from .integration_enhanced import (
        EnhancedMAIFProcessor,
        ConversionResult,
        EnhancedMAIF,
    )
except ImportError:
    EnhancedMAIFProcessor = None
    ConversionResult = None
    EnhancedMAIF = None

try:
    from .framework_adapters import (
        LangChainAdapter,
        LlamaIndexAdapter,
        AutoGenAdapter,
    )
except ImportError:
    LangChainAdapter = None
    LlamaIndexAdapter = None
    AutoGenAdapter = None

__all__ = [
    "MAIFIntegration",
    "MAIFConverter",
    "EnhancedMAIFProcessor",
    "ConversionResult",
    "EnhancedMAIF",
    "LangChainAdapter",
    "LlamaIndexAdapter",
    "AutoGenAdapter",
]

