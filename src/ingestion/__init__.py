"""
Ingestion: Module for document ingestion and indexing
"""

from .extractors import (
    BaseDocumentExtractor,
    PDFExtractor,
    TXTExtractor,
    DocumentExtractorFactory,
    DocumentExtractionManager
)

# Lazy import to avoid loading IngestionPipeline (which requires API keys) when only extractors are needed
def __getattr__(name):
    if name == 'IngestionPipeline':
        from .ingestion_pipeline import IngestionPipeline
        return IngestionPipeline
    # Backward compatibility: RAGPipeline -> IngestionPipeline
    if name == 'RAGPipeline':
        from .ingestion_pipeline import IngestionPipeline
        return IngestionPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'BaseDocumentExtractor',
    'PDFExtractor',
    'TXTExtractor',
    'DocumentExtractorFactory',
    'DocumentExtractionManager',
    'IngestionPipeline',
    'RAGPipeline'  # Backward compatibility alias
]
