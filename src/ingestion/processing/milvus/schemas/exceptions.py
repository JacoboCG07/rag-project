"""
Custom exceptions for Milvus schemas
"""

class SchemaNotFoundError(Exception):
    """Error raised when no schema is registered for the given name."""
    pass

