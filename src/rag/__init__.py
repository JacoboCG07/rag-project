"""
RAG: Module for Retrieval Augmented Generation
"""

from .extractors import (
    BaseDocumentExtractor,
    PDFExtractor,
    TXTExtractor,
    DocumentExtractorFactory,
    DocumentExtractionManager
)

# Lazy import to avoid loading RAGPipeline (which requires API keys) when only extractors are needed
def __getattr__(name):
    if name == 'RAGPipeline':
        from .rag_pipeline import RAGPipeline
        return RAGPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'BaseDocumentExtractor',
    'PDFExtractor',
    'TXTExtractor',
    'DocumentExtractorFactory',
    'DocumentExtractionManager',
    'RAGPipeline'
]
