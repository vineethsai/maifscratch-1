# MAIFEncoder & MAIFDecoder

The `MAIFEncoder` and `MAIFDecoder` classes provide low-level access to MAIF binary format operations. These are the foundation upon which the higher-level SDK is built.

::: tip When to Use
Use these classes when you need:
- Direct control over block structure
- Append-on-write operations
- Custom block types
- Integration with existing systems

For most use cases, prefer the high-level `MAIFClient` and `Artifact` classes.
:::

## MAIFEncoder

The encoder creates MAIF files by adding blocks of various types.

### Quick Start

```python
from maif import MAIFEncoder

# Create encoder
encoder = MAIFEncoder(agent_id="my-agent")

# Add blocks
encoder.add_text_block("Hello, World!")
encoder.add_binary_block(b"binary data", block_type="data")

# Save to file
encoder.save("output.maif")
```

### Constructor

```python
class MAIFEncoder:
    def __init__(
        self,
        agent_id: Optional[str] = None,
        existing_maif_path: Optional[str] = None,
        existing_manifest_path: Optional[str] = None,
        enable_privacy: bool = False,
        privacy_engine: Optional[PrivacyEngine] = None,
        use_aws: bool = False,
        aws_bucket: Optional[str] = None,
        aws_prefix: str = "maif/"
    ):
        """
        Initialize MAIF encoder.

        Args:
            agent_id: Unique agent identifier (auto-generated if not provided)
            existing_maif_path: Path to existing MAIF file for append operations
            existing_manifest_path: Path to existing manifest for append operations
            enable_privacy: Enable privacy features (encryption, anonymization)
            privacy_engine: Custom privacy engine instance
            use_aws: Use AWS unified storage backend
            aws_bucket: S3 bucket for AWS storage
            aws_prefix: S3 key prefix for AWS storage
        """
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `blocks` | `List[MAIFBlock]` | List of all blocks in the encoder |
| `agent_id` | `str` | Agent identifier |
| `version_history` | `Dict[str, List[MAIFVersion]]` | Version history by block ID |
| `block_registry` | `Dict[str, List[MAIFBlock]]` | Block versions by block ID |

### Methods

#### add_text_block

Add a text block with optional privacy controls.

```python
def add_text_block(
    self,
    text: str,
    metadata: Optional[Dict] = None,
    update_block_id: Optional[str] = None,
    privacy_policy: Optional[PrivacyPolicy] = None,
    anonymize: bool = False,
    privacy_level: Optional[PrivacyLevel] = None,
    encryption_mode: Optional[EncryptionMode] = None
) -> str:
    """
    Add or update a text block.

    Args:
        text: Text content to add
        metadata: Optional metadata dictionary
        update_block_id: If provided, updates existing block
        privacy_policy: Privacy policy to apply
        anonymize: Apply anonymization to text
        privacy_level: Privacy level for the block
        encryption_mode: Encryption mode for the block

    Returns:
        Block ID
    """
```

**Example:**

```python
from maif import MAIFEncoder, PrivacyLevel, EncryptionMode

encoder = MAIFEncoder(agent_id="agent-1", enable_privacy=True)

# Simple text block
block_id = encoder.add_text_block("Hello, World!")

# Text with metadata
block_id = encoder.add_text_block(
    "Sensitive information",
    metadata={"category": "personal"},
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM
)

# Anonymized text
block_id = encoder.add_text_block(
    "User John Doe lives at 123 Main St",
    anonymize=True
)
```

#### add_binary_block

Add a binary data block.

```python
def add_binary_block(
    self,
    data: bytes,
    block_type: str,
    metadata: Optional[Dict] = None,
    update_block_id: Optional[str] = None,
    privacy_policy: Optional[PrivacyPolicy] = None
) -> str:
    """
    Add or update a binary block.

    Args:
        data: Binary data to add
        block_type: Type of block (e.g., "data", "image", "video")
        metadata: Optional metadata dictionary
        update_block_id: If provided, updates existing block
        privacy_policy: Privacy policy to apply

    Returns:
        Block ID
    """
```

**Example:**

```python
# Add image data
with open("image.png", "rb") as f:
    block_id = encoder.add_binary_block(
        f.read(),
        block_type="image",
        metadata={"format": "png", "width": 1920, "height": 1080}
    )

# Add JSON data
import json
block_id = encoder.add_binary_block(
    json.dumps({"key": "value"}).encode(),
    block_type="data",
    metadata={"format": "json"}
)
```

#### add_embeddings_block

Add semantic embeddings.

```python
def add_embeddings_block(
    self,
    embeddings: List[List[float]],
    metadata: Optional[Dict] = None,
    update_block_id: Optional[str] = None,
    privacy_policy: Optional[PrivacyPolicy] = None
) -> str:
    """
    Add or update an embeddings block.

    Args:
        embeddings: List of embedding vectors
        metadata: Optional metadata dictionary
        update_block_id: If provided, updates existing block
        privacy_policy: Privacy policy to apply

    Returns:
        Block ID
    """
```

**Example:**

```python
# Add embeddings from your model
embeddings = [
    [0.1, 0.2, 0.3, ...],  # 768-dimensional vector
    [0.4, 0.5, 0.6, ...],
]

block_id = encoder.add_embeddings_block(
    embeddings,
    metadata={"model": "sentence-transformers", "dimensions": 768}
)
```

#### add_video_block

Add video data with optional metadata extraction.

```python
def add_video_block(
    self,
    video_data: bytes,
    metadata: Optional[Dict] = None,
    update_block_id: Optional[str] = None,
    privacy_policy: Optional[PrivacyPolicy] = None,
    extract_metadata: bool = True,
    enable_semantic_analysis: bool = True
) -> str:
    """
    Add or update a video block.

    Args:
        video_data: Raw video bytes
        metadata: Optional metadata dictionary
        update_block_id: If provided, updates existing block
        privacy_policy: Privacy policy to apply
        extract_metadata: Extract video metadata (duration, resolution, etc.)
        enable_semantic_analysis: Enable semantic analysis of video content

    Returns:
        Block ID
    """
```

#### update_text_block

Update an existing text block (creates a new version).

```python
def update_text_block(
    self,
    block_id: str,
    text: str,
    metadata: Optional[Dict] = None,
    privacy_policy: Optional[PrivacyPolicy] = None,
    anonymize: bool = False
) -> str:
    """
    Update an existing text block.

    Args:
        block_id: ID of block to update
        text: New text content
        metadata: Optional metadata dictionary
        privacy_policy: Privacy policy to apply
        anonymize: Apply anonymization

    Returns:
        Block ID (same as input)
    """
```

#### save

Save the encoded data to files.

```python
def save(
    self,
    maif_path: str,
    manifest_path: Optional[str] = None
) -> Tuple[str, str]:
    """
    Save the MAIF file and manifest.

    Args:
        maif_path: Path for the binary MAIF file
        manifest_path: Path for the JSON manifest (defaults to maif_path + '.manifest.json')

    Returns:
        Tuple of (maif_path, manifest_path)
    """
```

**Example:**

```python
encoder = MAIFEncoder(agent_id="agent-1")
encoder.add_text_block("Hello")
encoder.add_text_block("World")

# Save to files
maif_path, manifest_path = encoder.save("output.maif")
# Creates: output.maif and output.maif.manifest.json
```

## MAIFDecoder

The decoder reads MAIF files and provides access to blocks and their content.

### Quick Start

```python
from maif import MAIFDecoder

# Load a MAIF file
decoder = MAIFDecoder("data.maif", "data.maif.manifest.json")

# Access blocks
for block in decoder.blocks:
    print(f"Block {block.block_id}: {block.block_type}")
    
# Read block data
data = decoder.read_block(decoder.blocks[0])
```

### Constructor

```python
class MAIFDecoder:
    def __init__(
        self,
        maif_path: str,
        manifest_path: Optional[str] = None,
        privacy_engine: Optional[PrivacyEngine] = None,
        requesting_agent: Optional[str] = None,
        preload_semantic: bool = False
    ):
        """
        Initialize MAIF decoder.

        Args:
            maif_path: Path to the MAIF binary file
            manifest_path: Path to the manifest JSON file
            privacy_engine: Privacy engine for decryption
            requesting_agent: Agent ID for access control
            preload_semantic: Preload semantic indices

        Raises:
            FileNotFoundError: If MAIF file doesn't exist
        """
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `blocks` | `List[MAIFBlock]` | List of all blocks in the file |
| `manifest` | `Dict` | Parsed manifest dictionary |
| `version_history` | `Dict[str, List[MAIFVersion]]` | Version history by block ID |
| `block_registry` | `Dict[str, List[MAIFBlock]]` | Block versions by block ID |

### Methods

#### read_block

Read data from a specific block.

```python
def read_block(self, block: MAIFBlock) -> bytes:
    """
    Read data from a block.

    Args:
        block: The block to read

    Returns:
        Block data bytes (decrypted if necessary)
    """
```

**Example:**

```python
decoder = MAIFDecoder("data.maif", "data.maif.manifest.json")

# Read all text blocks
for block in decoder.blocks:
    if block.block_type == "text":
        data = decoder.read_block(block)
        print(data.decode('utf-8'))
```

#### get_block_by_id

Get a specific block by its ID.

```python
def get_block_by_id(self, block_id: str) -> Optional[MAIFBlock]:
    """
    Get a block by its ID.

    Args:
        block_id: The block ID to find

    Returns:
        MAIFBlock if found, None otherwise
    """
```

#### get_block_versions

Get all versions of a block.

```python
def get_block_versions(self, block_id: str) -> List[MAIFBlock]:
    """
    Get all versions of a block.

    Args:
        block_id: The block ID

    Returns:
        List of block versions (oldest to newest)
    """
```

#### get_version_history

Get the version history for a block.

```python
def get_version_history(self, block_id: str) -> List[MAIFVersion]:
    """
    Get version history for a block.

    Args:
        block_id: The block ID

    Returns:
        List of MAIFVersion entries
    """
```

## MAIFBlock

The `MAIFBlock` dataclass represents a single block in a MAIF file.

```python
@dataclass
class MAIFBlock:
    block_type: str
    offset: int = 0
    size: int = 0
    hash_value: str = ""
    version: int = 1
    previous_hash: Optional[str] = None
    block_id: Optional[str] = None
    metadata: Optional[Dict] = None
    data: Optional[bytes] = None
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `block_type` | `str` | Type of block (e.g., "text", "data", "image") |
| `offset` | `int` | Byte offset in the MAIF file |
| `size` | `int` | Size of the block in bytes |
| `hash_value` | `str` | SHA-256 hash of block data |
| `version` | `int` | Version number (starts at 1) |
| `previous_hash` | `str` | Hash of previous version (for updates) |
| `block_id` | `str` | Unique block identifier (UUID) |
| `metadata` | `Dict` | Arbitrary metadata dictionary |
| `data` | `bytes` | Block data (may be lazy-loaded) |
| `hash` | `str` | Alias for `hash_value` |

## Block Types

MAIF supports the following standard block types:

```python
from maif.block_types import BlockType

BlockType.TEXT_DATA      # "text" - UTF-8 text content
BlockType.BINARY_DATA    # "binr" - Binary data
BlockType.IMAGE_DATA     # "imag" - Image data
BlockType.VIDEO_DATA     # "vide" - Video data
BlockType.AUDIO_DATA     # "audi" - Audio data
BlockType.EMBEDDING      # "embd" - Vector embeddings
BlockType.METADATA       # "meta" - Metadata block
BlockType.CROSS_MODAL    # "xmod" - Cross-modal associations
BlockType.KNOWLEDGE_GRAPH # "kgra" - Knowledge graph data
BlockType.SECURITY       # "secu" - Security information
BlockType.PROVENANCE     # "prov" - Provenance chain
BlockType.ACCESS_CONTROL # "actl" - Access control rules
BlockType.LIFECYCLE      # "life" - Lifecycle information
```

## Complete Example

```python
from maif import MAIFEncoder, MAIFDecoder
from maif.privacy import PrivacyPolicy, PrivacyLevel, EncryptionMode

# Create a MAIF file with mixed content
encoder = MAIFEncoder(agent_id="demo-agent", enable_privacy=True)

# Add text content
doc_id = encoder.add_text_block(
    "This is a document about AI safety.",
    metadata={"title": "AI Safety", "category": "research"}
)

# Update the document
encoder.update_text_block(
    doc_id,
    "This is an updated document about AI safety and alignment.",
    metadata={"title": "AI Safety", "category": "research", "updated": True}
)

# Add encrypted sensitive data
encoder.add_text_block(
    "Secret research notes",
    privacy_level=PrivacyLevel.RESTRICTED,
    encryption_mode=EncryptionMode.AES_GCM
)

# Add embeddings
encoder.add_embeddings_block(
    [[0.1, 0.2, 0.3] * 256],  # 768-dimensional embedding
    metadata={"model": "ada-002"}
)

# Save
encoder.save("research.maif")

# Load and read
decoder = MAIFDecoder("research.maif", "research.maif.manifest.json")

print(f"Total blocks: {len(decoder.blocks)}")

# Read each block
for block in decoder.blocks:
    print(f"\nBlock ID: {block.block_id}")
    print(f"  Type: {block.block_type}")
    print(f"  Version: {block.version}")
    print(f"  Hash: {block.hash_value[:16]}...")
    
# Get version history
history = decoder.get_version_history(doc_id)
print(f"\nDocument has {len(history)} versions:")
for v in history:
    print(f"  v{v.version}: {v.operation} by {v.agent_id}")
```
