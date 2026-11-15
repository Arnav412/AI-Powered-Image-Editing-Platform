"""Typed exceptions for PromptCanvas.

Provides structured error handling across storage, validation, services, and AI providers.
"""
from __future__ import annotations


class PromptCanvasError(ValueError):
    """Base exception for all PromptCanvas errors."""


class ImageValidationError(PromptCanvasError):
    """Raised when an uploaded file is not a valid or supported image."""


class ImageNotFoundError(PromptCanvasError):
    """Raised when an image ID does not exist in storage."""


class VersionNotFoundError(PromptCanvasError):
    """Raised when a version ID does not exist in storage."""


class StorageError(PromptCanvasError):
    """Raised when a database or filesystem operation fails."""


class SearchError(PromptCanvasError):
    """Raised when an error occurs during semantic or lexical search."""
