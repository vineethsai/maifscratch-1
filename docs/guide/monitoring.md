# Monitoring & Observability

::: danger DEPRECATED
This page is deprecated. For the latest documentation, please visit **[DeepWiki](https://deepwiki.com/vineethsai/maif)**.
:::

This guide covers monitoring MAIF applications in production, including health checks, metrics collection, and observability patterns.

## Overview

MAIF includes built-in components for:

- **Health Checks**: Verify system health
- **Rate Limiting**: Control request throughput
- **Metrics**: Collect and aggregate metrics
- **Cost Tracking**: Monitor resource usage

## Built-in Components

### Health Checker

MAIF provides a `HealthChecker` class for monitoring system health:

```python
from maif import HealthChecker

# Create a health checker
health_checker = HealthChecker()

# Perform health check
status = health_checker.check()
print(f"Health status: {status}")

# Check specific components
components = health_checker.check_components()
for component, status in components.items():
    print(f"{component}: {status}")
```

### Rate Limiter

Control request throughput with the built-in rate limiter:

```python
from maif import RateLimiter

# Create a rate limiter (100 requests per second)
rate_limiter = RateLimiter(requests_per_second=100)

def handle_request(request):
    if rate_limiter.allow():
        # Process the request
        process(request)
    else:
        # Rate limited
        return "Too many requests", 429
```

### Metrics Aggregator

Collect and aggregate metrics:

```python
from maif import MetricsAggregator

# Create metrics aggregator
metrics = MetricsAggregator()

# Record metrics
metrics.record("requests_total", 1)
metrics.record("response_time_ms", 45.2)
metrics.record("artifacts_created", 1)

# Get aggregated metrics
summary = metrics.get_summary()
print(f"Total requests: {summary['requests_total']}")
print(f"Avg response time: {summary['response_time_ms_avg']}ms")
```

### Cost Tracker

Track resource usage and costs:

```python
from maif import CostTracker

# Create cost tracker
cost_tracker = CostTracker()

# Track operations
cost_tracker.record_operation("embedding_generation", tokens=1000)
cost_tracker.record_operation("storage_write", bytes=1024000)
cost_tracker.record_operation("api_call", count=1)

# Get cost summary
costs = cost_tracker.get_summary()
print(f"Total cost: ${costs['total']:.4f}")
```

## MAIF File Monitoring

### Monitor Artifact Integrity

```python
from maif_api import load_maif
import os

def check_artifact_health(artifact_path: str) -> dict:
    """Check health of a MAIF artifact."""
    status = {
        "path": artifact_path,
        "exists": os.path.exists(artifact_path),
        "readable": False,
        "integrity": False,
        "size_bytes": 0
    }
    
    if not status["exists"]:
        return status
    
    status["size_bytes"] = os.path.getsize(artifact_path)
    
    try:
        artifact = load_maif(artifact_path)
        status["readable"] = True
        status["integrity"] = artifact.verify()
        status["content_count"] = len(artifact.get_content_list())
    except Exception as e:
        status["error"] = str(e)
    
    return status

# Check artifact health
health = check_artifact_health("my_artifact.maif")
print(f"Artifact healthy: {health['integrity']}")
```

### Monitor Multiple Artifacts

```python
import os
from maif_api import load_maif

class ArtifactMonitor:
    """Monitor a directory of MAIF artifacts."""
    
    def __init__(self, directory: str):
        self.directory = directory
        self.last_check = {}
    
    def scan(self) -> list:
        """Scan and check all artifacts."""
        results = []
        
        for filename in os.listdir(self.directory):
            if filename.endswith('.maif'):
                path = os.path.join(self.directory, filename)
                status = self._check_file(path)
                results.append(status)
        
        return results
    
    def _check_file(self, path: str) -> dict:
        """Check a single artifact file."""
        status = {
            "path": path,
            "name": os.path.basename(path),
            "healthy": False,
            "size_bytes": os.path.getsize(path),
            "modified": os.path.getmtime(path)
        }
        
        try:
            artifact = load_maif(path)
            status["healthy"] = artifact.verify()
            status["blocks"] = len(artifact.get_content_list())
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def get_unhealthy(self) -> list:
        """Get list of unhealthy artifacts."""
        results = self.scan()
        return [r for r in results if not r.get("healthy", False)]

# Usage
monitor = ArtifactMonitor("./artifacts")
unhealthy = monitor.get_unhealthy()
if unhealthy:
    print(f"Warning: {len(unhealthy)} unhealthy artifacts found")
```

## Integration with Standard Tools

### Python Logging

```python
import logging
from maif_api import create_maif

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('maif_app')

def create_monitored_artifact(name: str, content: str):
    """Create artifact with logging."""
    logger.info(f"Creating artifact: {name}")
    
    try:
        artifact = create_maif(name)
        artifact.add_text(content)
        artifact.save(f"{name}.maif")
        
        logger.info(f"Artifact created successfully: {name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create artifact {name}: {e}")
        return False
```

### Prometheus Metrics

Export metrics to Prometheus:

```python
# requirements: prometheus_client

from prometheus_client import Counter, Histogram, start_http_server
from maif_api import create_maif, load_maif
import time

# Define metrics
ARTIFACTS_CREATED = Counter(
    'maif_artifacts_created_total',
    'Total number of artifacts created',
    ['agent_id']
)

ARTIFACT_OPERATIONS = Histogram(
    'maif_operation_duration_seconds',
    'Duration of MAIF operations',
    ['operation']
)

ARTIFACT_SIZE = Histogram(
    'maif_artifact_size_bytes',
    'Size of MAIF artifacts'
)

def create_artifact_with_metrics(agent_id: str, content: str, path: str):
    """Create artifact with Prometheus metrics."""
    start = time.time()
    
    artifact = create_maif(agent_id)
    artifact.add_text(content)
    artifact.save(path)
    
    # Record metrics
    duration = time.time() - start
    ARTIFACTS_CREATED.labels(agent_id=agent_id).inc()
    ARTIFACT_OPERATIONS.labels(operation='create').observe(duration)
    
    import os
    ARTIFACT_SIZE.observe(os.path.getsize(path))
    
    return artifact

# Start metrics server
start_http_server(8000)
print("Metrics available at http://localhost:8000/metrics")
```

### Structured JSON Logging

```python
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON log formatter for MAIF."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, 'artifact_id'):
            log_entry['artifact_id'] = record.artifact_id
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        
        return json.dumps(log_entry)

# Configure JSON logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())

logger = logging.getLogger('maif_json')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Log with extra context
logger.info(
    "Artifact created",
    extra={
        'artifact_id': 'artifact_123',
        'operation': 'create',
        'duration_ms': 45.2
    }
)
```

## Health Check Endpoint

Create a simple health check endpoint:

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from maif_api import load_maif
import os

class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks."""
    
    def do_GET(self):
        if self.path == '/health':
            status = self.check_health()
            
            self.send_response(200 if status['healthy'] else 503)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
    
    def check_health(self) -> dict:
        """Check overall system health."""
        status = {
            'healthy': True,
            'checks': {}
        }
        
        # Check if artifacts directory exists
        artifacts_dir = os.environ.get('ARTIFACTS_DIR', './artifacts')
        status['checks']['artifacts_dir'] = os.path.isdir(artifacts_dir)
        
        # Check sample artifact if it exists
        sample_path = os.path.join(artifacts_dir, 'health_check.maif')
        if os.path.exists(sample_path):
            try:
                artifact = load_maif(sample_path)
                status['checks']['sample_artifact'] = artifact.verify()
            except:
                status['checks']['sample_artifact'] = False
        
        # Overall health
        status['healthy'] = all(status['checks'].values())
        
        return status

# Run health check server
def run_health_server(port: int = 8080):
    server = HTTPServer(('', port), HealthHandler)
    print(f"Health check server running on port {port}")
    server.serve_forever()

# run_health_server()  # Uncomment to run
```

## Monitoring Patterns

### Operation Timing

```python
import time
from contextlib import contextmanager

@contextmanager
def timed_operation(name: str, logger=None):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        duration = (time.time() - start) * 1000  # ms
        message = f"{name} completed in {duration:.2f}ms"
        if logger:
            logger.info(message)
        else:
            print(message)

# Usage
with timed_operation("artifact_creation"):
    artifact = create_maif("timed-artifact")
    artifact.add_text("Some content")
    artifact.save("timed.maif")
```

### Error Tracking

```python
from collections import defaultdict
from datetime import datetime

class ErrorTracker:
    """Track and aggregate errors."""
    
    def __init__(self):
        self.errors = defaultdict(list)
    
    def record(self, error_type: str, message: str, context: dict = None):
        self.errors[error_type].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "context": context or {}
        })
    
    def get_summary(self) -> dict:
        return {
            error_type: len(errors)
            for error_type, errors in self.errors.items()
        }
    
    def get_recent(self, error_type: str = None, limit: int = 10) -> list:
        if error_type:
            return self.errors[error_type][-limit:]
        
        all_errors = []
        for errors in self.errors.values():
            all_errors.extend(errors)
        return sorted(all_errors, key=lambda x: x['timestamp'])[-limit:]

# Usage
tracker = ErrorTracker()

try:
    artifact = load_maif("nonexistent.maif")
except Exception as e:
    tracker.record("load_error", str(e), {"path": "nonexistent.maif"})

print(tracker.get_summary())
```

## Best Practices

1. **Monitor artifact integrity** - Regular integrity checks prevent data corruption
2. **Use structured logging** - JSON logs are easier to analyze
3. **Track operation timing** - Identify performance bottlenecks
4. **Set up alerts** - Get notified of failures immediately
5. **Keep metrics lightweight** - Don't impact application performance

## Next Steps

- **[Performance →](/guide/performance)** - Optimize based on monitoring data
- **[Architecture →](/guide/architecture)** - System design patterns
- **[API Reference →](/api/)** - Complete documentation
