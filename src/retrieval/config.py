"""
Configuration classes for Search Pipeline
Uses Pydantic for type validation and configuration management
"""

import os
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from src.utils import get_logger
from src.llms import BaseEmbedder, BaseTextModel, create_embedder, create_text_model
from pydantic import BaseModel, Field, model_validator, ConfigDict
from src.milvus.partitions import partition_documents, partition_summaries

load_dotenv()

class SearchType(str, Enum):
    """Types of search available"""
    SIMPLE = "simple"  # Direct search in Milvus
    WITH_SELECTION = "with_selection"  # Document selection + search in selected documents
    WITH_SELECTION_AND_METADATA = "with_selection_and_metadata"  # Document selection + search with metadata filters


class MilvusConfig(BaseModel):
    """Configuration for Milvus connection"""
    
    alias: str = "default"
    uri: Optional[str] = os.getenv("MILVUS_URI", None)
    dbname: str = os.getenv("MILVUS_DB_NAME", "default")
    token: Optional[str] = os.getenv("MILVUS_TOKEN", None)
    port: Optional[str] = os.getenv("MILVUS_PORT", "19530")
    host: Optional[str] = os.getenv("MILVUS_HOST", "localhost")


class SearchPipelineConfig(BaseModel):
    """Configuration for Search Pipeline"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Search type
    search_type: SearchType = Field(
        default=SearchType.SIMPLE,
        description="Type of search: 'simple' (direct search), 'with_selection' (selection + search), 'with_selection_and_metadata' (selection + search with metadata)"
    )
    
    # Milvus configuration
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    collection_name: str = Field(
        default="default_collection",
        description="Nombre de la colección. Particiones solo vía MILVUS_PARTITION_NAME_* (.env)"
    )
    
    # Embedding configuration (para convertir query a embedding internamente)
    embedder: BaseEmbedder = Field(
        default_factory=lambda: create_embedder(count_tokens=False),
        description="Embedder para convertir la query a vector"
    )
    
    # Search parameters
    search_limit: int = Field(
        default=10,
        ge=5,
        le=25,
        description="Maximum number of results to return"
    )
    
    # Document selection configuration (only used when search_type == WITH_SELECTION)
    text_model: Optional[BaseTextModel] = Field(
        default=None,
        description="LLM model for document selection (required when search_type == 'with_selection')"
    )
    chooser_max_tokens: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum tokens for document chooser LLM"
    )
    chooser_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Temperature for document chooser LLM"
    )
    
    @model_validator(mode='after')
    def validate_config(self):
        """Validate configuration based on search type"""
        logger = get_logger(__name__)
        
        # If search type requires document selection, text_model is required
        if self.search_type in [SearchType.WITH_SELECTION, SearchType.WITH_SELECTION_AND_METADATA]:
            if self.text_model is None:
                # Try to create a default text model for the configured provider
                try:
                    self.text_model = create_text_model()
                    logger.info("Created default text model for document selection")
                except Exception as e:
                    logger.error(
                        f"text_model is required for search type '{self.search_type.value}'. Error creating default: {str(e)}"
                    )
                    raise ValueError(
                        f"text_model is required for search type '{self.search_type.value}'. "
                        "Please provide a BaseTextModel instance."
                    )
        
        logger.info(
            "SearchPipelineConfig initialized",
            extra={
                "search_type": self.search_type.value,
                "collection_name": self.collection_name,
                "partition_documents": partition_documents(),
                "partition_summaries": partition_summaries(),
                "search_limit": self.search_limit,
                "has_text_model": self.text_model is not None
            }
        )
        
        return self