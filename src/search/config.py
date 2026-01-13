"""
Configuration classes for Search Pipeline
Uses Pydantic for type validation and configuration management
"""
from typing import Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, model_validator
import os
from dotenv import load_dotenv

from llms.text import BaseTextModel, OpenAITextModel
from src.utils import get_logger

load_dotenv()


class SearchType(str, Enum):
    """Types of search available"""
    NORMAL = "normal"  # Direct search in Milvus
    WITH_SELECTION = "with_selection"  # Document selection + search in selected documents


class MilvusConfig(BaseModel):
    """Configuration for Milvus connection"""
    
    dbname: str = os.getenv("MILVUS_DB_NAME", "default")
    alias: str = "default"
    uri: Optional[str] = os.getenv("MILVUS_URI", None)
    token: Optional[str] = os.getenv("MILVUS_TOKEN", None)
    host: Optional[str] = os.getenv("MILVUS_HOST", "localhost")
    port: Optional[str] = os.getenv("MILVUS_PORT", "19530")


class SearchPipelineConfig(BaseModel):
    """Configuration for Search Pipeline"""
    
    # Search type
    search_type: SearchType = Field(
        default=SearchType.NORMAL,
        description="Type of search to perform: 'normal' for direct search, 'with_selection' for selection + search"
    )
    
    # Milvus configuration
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    collection_name_documents: str = os.getenv("MILVUS_COLLECTION_NAME_DOCUMENTS", "documents")
    collection_name_summaries: str = os.getenv("MILVUS_COLLECTION_NAME_SUMMARIES", "summaries")
    
    # Search parameters
    search_limit: int = Field(
        default=10,
        ge=1,
        le=100,
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
        
        # If search type is WITH_SELECTION, text_model is required
        if self.search_type == SearchType.WITH_SELECTION:
            if self.text_model is None:
                # Try to create a default OpenAI model if available
                try:
                    self.text_model = OpenAITextModel(model="gpt-4o-mini")
                    logger.info("Created default OpenAI text model for document selection")
                except Exception as e:
                    logger.error(
                        f"text_model is required when search_type='with_selection'. Error creating default: {str(e)}"
                    )
                    raise ValueError(
                        "text_model is required when search_type='with_selection'. "
                        "Please provide a BaseTextModel instance."
                    )
        
        logger.info(
            "SearchPipelineConfig initialized",
            extra={
                "search_type": self.search_type.value,
                "collection_documents": self.collection_name_documents,
                "collection_summaries": self.collection_name_summaries,
                "search_limit": self.search_limit,
                "has_text_model": self.text_model is not None
            }
        )
        
        return self

