"""
Production Error Handling for MAIF
==================================

Comprehensive error handling with categorization, retry logic, and reporting.
"""

import sys
import traceback
from typing import Optional, Dict, Any, Callable, Type, Union, List
from functools import wraps
import time
from enum import Enum
from dataclasses import dataclass
import json
import os
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import AWS SDK
try:
    import boto3
    from .aws_config import get_aws_config
    from .compliance_logging import EnhancedComplianceLogger

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    RESOURCE_NOT_FOUND = "resource_not_found"
    CONFLICT = "conflict"
    DATA_CORRUPTION = "data_corruption"
    CONFIGURATION = "configuration"
    INTERNAL = "internal"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Context information for errors."""

    operation: str
    component: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MAIFError(Exception):
    """Base exception for all MAIF errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        retry_allowed: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.cause = cause
        self.retry_allowed = retry_allowed
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "retry_allowed": self.retry_allowed,
            "timestamp": self.timestamp,
            "context": {
                "operation": self.context.operation if self.context else None,
                "component": self.context.component if self.context else None,
                "user_id": self.context.user_id if self.context else None,
                "request_id": self.context.request_id if self.context else None,
                "resource_id": self.context.resource_id if self.context else None,
                "metadata": self.context.metadata if self.context else None,
            }
            if self.context
            else None,
            "cause": str(self.cause) if self.cause else None,
            "traceback": traceback.format_exc() if self.cause else None,
        }


# Specific error types
class ValidationError(MAIFError):
    """Validation error."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs,
        )
        self.field = field


class AuthenticationError(MAIFError):
    """Authentication error."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class AuthorizationError(MAIFError):
    """Authorization error."""

    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class NetworkError(MAIFError):
    """Network-related error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            retry_allowed=True,
            **kwargs,
        )


class TimeoutError(MAIFError):
    """Timeout error."""

    def __init__(self, message: str, timeout_seconds: float, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            retry_allowed=True,
            **kwargs,
        )
        self.timeout_seconds = timeout_seconds


class RateLimitError(MAIFError):
    """Rate limit exceeded error."""

    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.LOW,
            retry_allowed=True,
            **kwargs,
        )
        self.retry_after = retry_after


class ResourceNotFoundError(MAIFError):
    """Resource not found error."""

    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.LOW,
            **kwargs,
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(MAIFError):
    """Resource conflict error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFLICT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class DataCorruptionError(MAIFError):
    """Data corruption error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_CORRUPTION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )


class ConfigurationError(MAIFError):
    """Configuration error."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        self.config_key = config_key


class ExternalServiceError(MAIFError):
    """External service error."""

    def __init__(self, service_name: str, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            retry_allowed=True,
            **kwargs,
        )
        self.service_name = service_name


def handle_error(
    error: Exception,
    context: Optional[ErrorContext] = None,
    reraise: bool = True,
    log_level: str = "error",
) -> Optional[MAIFError]:
    """
    Handle an error with proper logging and conversion.

    Args:
        error: The error to handle
        context: Error context information
        reraise: Whether to re-raise the error
        log_level: Logging level to use

    Returns:
        MAIFError instance if not re-raised
    """
    # Convert to MAIFError if needed
    if isinstance(error, MAIFError):
        maif_error = error
    else:
        maif_error = MAIFError(message=str(error), context=context, cause=error)

    # Log the error
    log_func = getattr(logger, log_level)
    log_func("Error occurred", **maif_error.to_dict())

    # Send to monitoring/alerting if critical
    if maif_error.severity == ErrorSeverity.CRITICAL:
        _send_critical_alert(maif_error)

    if reraise:
        raise maif_error
    return maif_error


def _send_critical_alert(error: MAIFError):
    """Send critical error alert through multiple channels."""
    error_dict = error.to_dict()
    logger.critical("CRITICAL ERROR ALERT", error=error_dict)

    # AWS SNS Integration
    if AWS_AVAILABLE and os.environ.get("AWS_SNS_ALERT_TOPIC_ARN"):
        try:
            # Use centralized AWS config if available
            if hasattr(boto3, "get_aws_config"):
                aws_config = get_aws_config()
                sns_client = aws_config.get_client("sns")
            else:
                sns_client = boto3.client("sns")

            topic_arn = os.environ.get("AWS_SNS_ALERT_TOPIC_ARN")
            if not topic_arn:
                logger.warning("AWS_SNS_ALERT_TOPIC_ARN not set")
                return

            sns_client.publish(
                TopicArn=topic_arn,
                Subject=f"MAIF Critical Error: {error.error_code}",
                Message=json.dumps(
                    {
                        "default": json.dumps(error_dict),
                        "email": f"Critical error in MAIF:\n\n"
                        f"Error Code: {error.error_code}\n"
                        f"Severity: {error.severity.value}\n"
                        f"Component: {error.component}\n"
                        f"Operation: {error.operation}\n"
                        f"Message: {error.message}\n"
                        f"Details: {json.dumps(error.details, indent=2)}",
                    }
                ),
                MessageStructure="json",
            )
            logger.info(f"Alert sent via AWS SNS to {topic_arn}")
        except Exception as sns_error:
            logger.error(f"Failed to send SNS alert: {sns_error}")

    # PagerDuty Integration
    pagerduty_key = os.environ.get("PAGERDUTY_INTEGRATION_KEY")
    if pagerduty_key:
        try:
            pagerduty_url = "https://events.pagerduty.com/v2/enqueue"

            payload = {
                "routing_key": pagerduty_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"MAIF Critical Error: {error.error_code}",
                    "severity": "critical"
                    if error.severity == ErrorSeverity.CRITICAL
                    else "error",
                    "source": f"MAIF/{error.component}",
                    "custom_details": error_dict,
                },
                "client": "MAIF Error Handler",
                "client_url": "https://github.com/your-org/maif",
            }

            response = requests.post(
                pagerduty_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Alert sent via PagerDuty")
        except Exception as pd_error:
            logger.error(f"Failed to send PagerDuty alert: {pd_error}")

    # Slack Webhook Integration
    slack_webhook_url = os.environ.get("SLACK_ALERT_WEBHOOK_URL")
    if slack_webhook_url:
        try:
            slack_payload = {
                "text": f"🚨 MAIF Critical Error Alert",
                "attachments": [
                    {
                        "color": "danger",
                        "title": f"Error: {error.error_code}",
                        "fields": [
                            {
                                "title": "Severity",
                                "value": error.severity.value,
                                "short": True,
                            },
                            {
                                "title": "Component",
                                "value": error.component,
                                "short": True,
                            },
                            {
                                "title": "Operation",
                                "value": error.operation,
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": error.timestamp.isoformat(),
                                "short": True,
                            },
                            {
                                "title": "Message",
                                "value": error.message,
                                "short": False,
                            },
                        ],
                        "footer": "MAIF Error Handler",
                        "ts": int(error.timestamp.timestamp()),
                    }
                ],
            }

            response = requests.post(
                slack_webhook_url,
                json=slack_payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Alert sent via Slack")
        except Exception as slack_error:
            logger.error(f"Failed to send Slack alert: {slack_error}")


def error_handler(
    operation: str,
    component: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    reraise: bool = True,
):
    """
    Decorator for automatic error handling.

    Args:
        operation: Name of the operation
        component: Component name
        severity: Default error severity
        reraise: Whether to re-raise exceptions
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                component=component,
                metadata={"args": str(args), "kwargs": str(kwargs)},
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(e, context=context, reraise=reraise)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                component=component,
                metadata={"args": str(args), "kwargs": str(kwargs)},
            )

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handle_error(e, context=context, reraise=reraise)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def create_retry_decorator(
    max_attempts: int = 3,
    wait_multiplier: float = 1.0,
    wait_max: float = 60.0,
    retry_on: Optional[List[Type[Exception]]] = None,
):
    """
    Create a retry decorator with customizable parameters.

    Args:
        max_attempts: Maximum number of retry attempts
        wait_multiplier: Exponential backoff multiplier
        wait_max: Maximum wait time between retries
        retry_on: List of exception types to retry on
    """
    if retry_on is None:
        retry_on = [NetworkError, TimeoutError, RateLimitError, ExternalServiceError]

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_multiplier, max=wait_max),
        retry=retry_if_exception_type(tuple(retry_on)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


# Production retry decorator
production_retry = create_retry_decorator()


def validate_input(
    data: Any, schema: Dict[str, Any], raise_on_error: bool = True
) -> Union[bool, List[str]]:
    """
    Validate input data against a schema.

    Args:
        data: Data to validate
        schema: Validation schema
        raise_on_error: Whether to raise ValidationError on failure

    Returns:
        True if valid, list of errors if not raising
    """
    errors = []

    # Implement schema validation
    if isinstance(schema, dict) and schema.get("type") == "json_schema":
        # JSON Schema validation
        try:
            import jsonschema

            validator = jsonschema.Draft7Validator(schema.get("schema", {}))
            for error in validator.iter_errors(data):
                errors.append(
                    f"{'.'.join(str(p) for p in error.path)}: {error.message}"
                )
        except ImportError:
            errors.append("jsonschema library not installed for JSON schema validation")
        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")

    elif isinstance(schema, dict) and schema.get("type") == "dataclass":
        # Dataclass validation
        dataclass_type = schema.get("dataclass")
        if dataclass_type:
            try:
                from dataclasses import fields

                if hasattr(dataclass_type, "__dataclass_fields__"):
                    for field in fields(dataclass_type):
                        if (
                            field.name not in data
                            and field.default is field.default_factory
                        ):
                            errors.append(f"Missing required field: {field.name}")
                        elif field.name in data:
                            # Type checking
                            expected_type = field.type
                            actual_value = data[field.name]
                            if not isinstance(actual_value, expected_type):
                                errors.append(
                                    f"{field.name}: expected {expected_type.__name__}, got {type(actual_value).__name__}"
                                )
            except Exception as e:
                errors.append(f"Dataclass validation error: {str(e)}")

    elif isinstance(schema, dict):
        # Simple dict-based validation
        for key, rules in schema.items():
            if isinstance(rules, dict):
                # Check required fields
                if rules.get("required", False) and key not in data:
                    errors.append(f"Missing required field: {key}")

                # Check type
                if key in data and "type" in rules:
                    expected_type = rules["type"]
                    actual_value = data.get(key)
                    if expected_type == "string" and not isinstance(actual_value, str):
                        errors.append(
                            f"{key}: expected string, got {type(actual_value).__name__}"
                        )
                    elif expected_type == "number" and not isinstance(
                        actual_value, (int, float)
                    ):
                        errors.append(
                            f"{key}: expected number, got {type(actual_value).__name__}"
                        )
                    elif expected_type == "boolean" and not isinstance(
                        actual_value, bool
                    ):
                        errors.append(
                            f"{key}: expected boolean, got {type(actual_value).__name__}"
                        )
                    elif expected_type == "array" and not isinstance(
                        actual_value, list
                    ):
                        errors.append(
                            f"{key}: expected array, got {type(actual_value).__name__}"
                        )
                    elif expected_type == "object" and not isinstance(
                        actual_value, dict
                    ):
                        errors.append(
                            f"{key}: expected object, got {type(actual_value).__name__}"
                        )

                # Check min/max for numbers
                if key in data and isinstance(data[key], (int, float)):
                    if "min" in rules and data[key] < rules["min"]:
                        errors.append(
                            f"{key}: value {data[key]} is less than minimum {rules['min']}"
                        )
                    if "max" in rules and data[key] > rules["max"]:
                        errors.append(
                            f"{key}: value {data[key]} is greater than maximum {rules['max']}"
                        )

                # Check length for strings and arrays
                if key in data and isinstance(data[key], (str, list)):
                    if "minLength" in rules and len(data[key]) < rules["minLength"]:
                        errors.append(
                            f"{key}: length {len(data[key])} is less than minimum {rules['minLength']}"
                        )
                    if "maxLength" in rules and len(data[key]) > rules["maxLength"]:
                        errors.append(
                            f"{key}: length {len(data[key])} is greater than maximum {rules['maxLength']}"
                        )

    if errors and raise_on_error:
        raise ValidationError(
            f"Validation failed: {', '.join(errors)}",
            context=ErrorContext(operation="validate_input", component="validation"),
        )

    return True if not errors else errors


# Circuit breaker implementation
class CircuitBreaker:
    """Circuit breaker for handling repeated failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise ExternalServiceError(
                    "Circuit breaker is open", "Service temporarily unavailable"
                )

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    "Circuit breaker opened",
                    failure_count=self.failure_count,
                    service=func.__name__,
                )

            raise


import asyncio
