"""
Metadata Module
Extrae y construye filtros de metadata para búsquedas en Milvus
"""
from .extractor import MetadataExtractor
from .filter_builder import MetadataFilterBuilder

__all__ = [
    "MetadataExtractor",
    "MetadataFilterBuilder"
]

