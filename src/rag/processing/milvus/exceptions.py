"""
Custom exceptions for Milvus operations
"""


class MilvusConnectionError(Exception):
    """Error when establishing connection with Milvus."""
    pass


class MilvusCollectionError(Exception):
    """Error when working with Milvus collections."""
    pass


class MilvusInsertError(Exception):
    """Error when inserting data into Milvus."""
    pass
