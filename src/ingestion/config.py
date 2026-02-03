"""
Configuration classes for Ingestion Pipeline
Uses Pydantic for type validation and configuration management
"""
import os
from dotenv import load_dotenv
from src.utils import get_logger
from src.llms import OpenAITextModel, OpenAIVisionModel
from typing import Optional, Callable, Any, List, Tuple
from src.llms.embeddings.openai_embedder import OpenAIEmbedder
from pydantic import BaseModel, Field, model_validator, ConfigDict
from src.ingestion.processing.summarizer.llm_summarizer import LLMSummarizer
from src.ingestion.processing.describer.llm_image_describer import LLMImageDescriber

load_dotenv()

class MilvusConfig(BaseModel):
    """Configuration for Milvus connection"""
    
    dbname: str = os.getenv("MILVUS_DB_NAME", "default")
    alias: str = "default"
    uri: Optional[str] = os.getenv("MILVUS_URI", None)
    token: Optional[str] = os.getenv("MILVUS_TOKEN", None)
    host: Optional[str] = os.getenv("MILVUS_HOST", "localhost")
    port: Optional[str] = os.getenv("MILVUS_PORT", "19530")


class IngestionPipelineConfig(BaseModel):
    """Configuration for Ingestion Pipeline (document processing)"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Milvus configuration
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    collection_name: str = Field(
        default="default_collection",
        description="Collection name. Each collection will have two partitions: 'documents' and 'summaries'"
    )
    
    # Chunking configuration
    chunk_size: int = Field(default=2000, ge=100, le=10_000, description="Maximum size of each chunk in characters")
    chunk_overlap: int = Field(default=0, ge=10, le=1000, description="Number of characters to overlap between chunks")
    detect_chapters: bool = Field(default=True, description="Whether to detect chapters in documents")
    
    # Embedding configuration
    embedder: OpenAIEmbedder = Field(default_factory=lambda: OpenAIEmbedder(model="text-embedding-ada-002", count_tokens=False))
    generate_embeddings_func: Optional[Callable[[str], Tuple[List[float], Optional[int]]]] = Field(default=None, exclude=True)

    # Summary configuration
    text_model: OpenAITextModel = Field(default_factory=lambda: OpenAITextModel(model="gpt-4o"))
    summarizer: Optional[LLMSummarizer] = Field(default=None)
    generate_summary_func: Optional[Callable[[str], str]] = Field(default=None, exclude=True)
    
    # Processing options
    extract_images: bool = Field(default=False, description="Whether to extract and process images from PDFs")

    # Image description configuration
    vision_model: OpenAIVisionModel = Field(default_factory=lambda: OpenAIVisionModel(model="gpt-4o"))
    image_describer: Optional[LLMImageDescriber] = Field(default=None)
    describe_image_func: Optional[Callable[..., str]] = Field(default=None, exclude=True)
    
    @model_validator(mode='after')
    def initialize_functions(self):
        """Initialize functions after model creation"""
        logger = get_logger(__name__)
        
        logger.debug("Initializing IngestionPipelineConfig functions")
        
        # Set functions after initialization
        if self.generate_embeddings_func is None:
            self.generate_embeddings_func = self.embedder.generate_embedding
        if self.summarizer is None:
            self.summarizer = LLMSummarizer(text_model=self.text_model, max_tokens=1_500, temperature=0.3)
        if self.generate_summary_func is None:
            self.generate_summary_func = self.summarizer.generate_summary
        if self.image_describer is None:
            self.image_describer = LLMImageDescriber(vision_model=self.vision_model, max_tokens=1_000, temperature=0.3)
        if self.describe_image_func is None:
            self.describe_image_func = self.image_describer.describe_image
        
        logger.info(
            "IngestionPipelineConfig initialized successfully",
            extra={
                "collection_name": self.collection_name,
                "embedding_dim": self.embedder.dimensions,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "has_summarizer": self.summarizer is not None,
                "has_image_describer": self.image_describer is not None
            }
        )
        
        return self