"""
Production Logging Configuration for MAIF
========================================

Structured logging setup with support for multiple outputs and formats.
"""

import sys
import logging
import structlog
from typing import Optional, List, Dict, Any
import json
from datetime import datetime
import traceback


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    add_timestamp: bool = True,
    add_caller_info: bool = True,
    log_file: Optional[str] = None,
):
    """
    Configure structured logging for production use.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json" or "console")
        add_timestamp: Include timestamp in logs
        add_caller_info: Include file/function/line info
        log_file: Optional file path for log output
    """

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure processors
    processors: List[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if add_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="iso"))

    if add_caller_info:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
        )

    # Add custom processors
    processors.extend(
        [
            add_service_context,
            add_aws_context,
            sanitize_sensitive_data,
        ]
    )

    # Configure renderer based on format
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(file_handler)


def add_service_context(logger, method_name, event_dict):
    """Add service context to all log entries."""
    from .config import get_config

    try:
        config = get_config()
        event_dict["service"] = config.app_name
        event_dict["environment"] = config.environment
    except Exception:
        pass

    return event_dict


def add_aws_context(logger, method_name, event_dict):
    """Add AWS context when available."""
    import os

    # Add AWS metadata if available
    if "AWS_REGION" in os.environ:
        event_dict["aws_region"] = os.environ["AWS_REGION"]
    if "AWS_LAMBDA_FUNCTION_NAME" in os.environ:
        event_dict["lambda_function"] = os.environ["AWS_LAMBDA_FUNCTION_NAME"]
    if "AWS_LAMBDA_REQUEST_ID" in os.environ:
        event_dict["request_id"] = os.environ["AWS_LAMBDA_REQUEST_ID"]

    return event_dict


def sanitize_sensitive_data(logger, method_name, event_dict):
    """Sanitize sensitive data from logs."""
    sensitive_keys = [
        "password",
        "token",
        "api_key",
        "secret",
        "credential",
        "private_key",
        "aws_access_key",
        "aws_secret_key",
    ]

    def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for key, value in d.items():
            if any(sk in key.lower() for sk in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    sanitize_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    return sanitize_dict(event_dict)


class ProductionLogger:
    """Production-ready logger with structured logging and context management."""

    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self._context = {}

    def bind(self, **kwargs) -> "ProductionLogger":
        """Bind context variables to logger."""
        self._context.update(kwargs)
        self.logger = self.logger.bind(**self._context)
        return self

    def unbind(self, *keys) -> "ProductionLogger":
        """Remove context variables from logger."""
        for key in keys:
            self._context.pop(key, None)
        self.logger = structlog.get_logger(self.logger._logger.name).bind(
            **self._context
        )
        return self

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception."""
        if exception:
            kwargs["exception"] = str(exception)
            kwargs["traceback"] = traceback.format_exc()
        self.logger.error(message, **kwargs)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception."""
        if exception:
            kwargs["exception"] = str(exception)
            kwargs["traceback"] = traceback.format_exc()
        self.logger.critical(message, **kwargs)

    def log_operation(self, operation: str, duration: float, success: bool, **kwargs):
        """Log operation with standard fields."""
        self.info(
            f"Operation {operation} completed",
            operation=operation,
            duration_ms=duration * 1000,
            success=success,
            **kwargs,
        )

    def log_aws_api_call(
        self,
        service: str,
        operation: str,
        duration: float,
        success: bool,
        error: Optional[str] = None,
        **kwargs,
    ):
        """Log AWS API call with standard fields."""
        event = {
            "aws_service": service,
            "aws_operation": operation,
            "duration_ms": duration * 1000,
            "success": success,
        }
        if error:
            event["error"] = error
        event.update(kwargs)

        if success:
            self.info(f"AWS API call: {service}.{operation}", **event)
        else:
            self.error(f"AWS API call failed: {service}.{operation}", **event)


def get_logger(name: str) -> ProductionLogger:
    """Get a production logger instance."""
    return ProductionLogger(name)


# Lambda logging configuration
def configure_lambda_logging():
    """Configure logging for AWS Lambda environment."""
    setup_logging(
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        log_format="json",
        add_timestamp=False,  # Lambda adds timestamp
        add_caller_info=True,
    )


# ECS/Fargate logging configuration
def configure_ecs_logging():
    """Configure logging for ECS/Fargate environment."""
    setup_logging(
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        log_format="json",
        add_timestamp=True,
        add_caller_info=True,
    )


# Development logging configuration
def configure_development_logging():
    """Configure logging for development environment."""
    setup_logging(
        log_level="DEBUG",
        log_format="console",
        add_timestamp=True,
        add_caller_info=True,
    )


# Initialize logging based on environment
import os

if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    configure_lambda_logging()
elif os.environ.get("ECS_CONTAINER_METADATA_URI"):
    configure_ecs_logging()
elif os.environ.get("MAIF_ENVIRONMENT") == "development":
    configure_development_logging()
else:
    # Default production logging
    setup_logging()
