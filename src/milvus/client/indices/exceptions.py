"""
Custom exceptions for Milvus indices
"""


class IndexNotFoundError(Exception):
    """Error raised when no index is registered for the given name."""
    pass

