"""
Configuration classes for RAG Pipeline
Uses Pydantic for type validation and configuration management
"""
from typing import Optional, Callable, Any, List
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

from llms import OpenAITextModel
from rag.processing.embeddings.openai_embedder import OpenAIEmbedder
from rag.processing.summarizer.llm_summarizer import LLMSummarizer

load_dotenv()


class MilvusConfig(BaseModel):
    """Configuration for Milvus connection"""
    
    dbname: str = os.getenv("MILVUS_DB_NAME", "default")
    alias: str = "default"
    uri: Optional[str] = os.getenv("MILVUS_URI", None)
    token: Optional[str] = os.getenv("MILVUS_TOKEN", None)
    host: Optional[str] = os.getenv("MILVUS_HOST", "localhost")
    port: Optional[str] = os.getenv("MILVUS_PORT", "19530")


class RAGPipelineConfig(BaseModel):
    """Configuration for RAG Pipeline (document processing)"""
    
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    collection_name_documents: str = os.getenv("MILVUS_COLLECTION_NAME_DOCUMENTS", "documents")
    collection_name_summaries: str = os.getenv("MILVUS_COLLECTION_NAME_SUMMARIES", "summaries")
    
    # Chunking configuration
    chunk_size: int = Field(default=2000, ge=100, le=10_000, description="Maximum size of each chunk in characters")
    chunk_overlap: int = Field(default=200, ge=10, le=1000, description="Number of characters to overlap between chunks")
    detect_chapters: bool = Field(default=True, description="Whether to detect chapters in documents")
    
    # Processing options
    extract_images: bool = Field(default=False, description="Whether to extract and process images from PDFs")
    
    # Functions (will be set separately, not in config)
    embedder = OpenAIEmbedder(model="text-embedding-ada-002", count_tokens=False)
    generate_embeddings_func = embedder.generate_embedding

    # Embedding dimensions
    embedding_dim = embedder.get_dimensions()

    # Summary configuration
    text_model = OpenAITextModel(model="gpt-4o")
    summarizer = LLMSummarizer(text_model=text_model, max_tokens=1_500, temperature=0.3)
    generate_summary_func = summarizer.generate_summary