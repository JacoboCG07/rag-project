"""
Configuration classes for RAG Pipeline
Uses Pydantic for type validation and configuration management
"""
from typing import Optional, Callable, Any, List, Tuple
from pydantic import BaseModel, Field, model_validator, ConfigDict
import os
from dotenv import load_dotenv

from llms import OpenAITextModel, OpenAIVisionModel
from rag.processing.embeddings.openai_embedder import OpenAIEmbedder
from rag.processing.summarizer.llm_summarizer import LLMSummarizer
from rag.processing.describer.llm_image_describer import LLMImageDescriber

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
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    collection_name_documents: str = os.getenv("MILVUS_COLLECTION_NAME_DOCUMENTS", "documents")
    collection_name_summaries: str = os.getenv("MILVUS_COLLECTION_NAME_SUMMARIES", "summaries")
    
    # Chunking configuration
    chunk_size: int = Field(default=2000, ge=100, le=10_000, description="Maximum size of each chunk in characters")
    chunk_overlap: int = Field(default=0, ge=10, le=1000, description="Number of characters to overlap between chunks")
    detect_chapters: bool = Field(default=True, description="Whether to detect chapters in documents")
    
    # Processing options
    extract_images: bool = Field(default=False, description="Whether to extract and process images from PDFs")
    
    # Embedding configuration
    embedder: OpenAIEmbedder = Field(default_factory=lambda: OpenAIEmbedder(model="text-embedding-ada-002", count_tokens=False))
    generate_embeddings_func: Optional[Callable[[str], Tuple[List[float], Optional[int]]]] = Field(default=None, exclude=True)
    embedding_dim: int = Field(default=1536)  # Default for text-embedding-ada-002

    # Summary configuration
    text_model: OpenAITextModel = Field(default_factory=lambda: OpenAITextModel(model="gpt-4o"))
    summarizer: Optional[LLMSummarizer] = Field(default=None)
    generate_summary_func: Optional[Callable[[str], str]] = Field(default=None, exclude=True)
    
    # Image description configuration
    vision_model: OpenAIVisionModel = Field(default_factory=lambda: OpenAIVisionModel(model="gpt-4o"))
    image_describer: Optional[LLMImageDescriber] = Field(default=None)
    describe_image_func: Optional[Callable[..., str]] = Field(default=None, exclude=True)
    
    @model_validator(mode='after')
    def initialize_functions(self):
        """Initialize functions after model creation"""
        # Set functions after initialization
        if self.generate_embeddings_func is None:
            # Create a wrapper function that accepts positional arguments
            # since generate_embedding requires keyword-only arguments
            def embedding_wrapper(text: str) -> Tuple[List[float], Optional[int]]:
                return self.embedder.generate_embedding(text=text)
            self.generate_embeddings_func = embedding_wrapper
        if self.summarizer is None:
            self.summarizer = LLMSummarizer(text_model=self.text_model, max_tokens=1_500, temperature=0.3)
        if self.generate_summary_func is None:
            self.generate_summary_func = self.summarizer.generate_summary
        if self.image_describer is None:
            self.image_describer = LLMImageDescriber(vision_model=self.vision_model, max_tokens=1_000, temperature=0.3)
        if self.describe_image_func is None:
            self.describe_image_func = self.image_describer.describe_image
        if self.embedding_dim == 1536:  # Only update if still default
            self.embedding_dim = self.embedder.get_dimensions()
        return self