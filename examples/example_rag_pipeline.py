"""
Example usage of RAG Pipeline with configuration
"""
from src.ingestion.config import IngestionPipelineConfig, SearchPipelineConfig, MilvusConfig
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.llms.embeddings.openai_embedder import OpenAIEmbedder

# Example 1: Using default configuration (from environment variables)
def example_default_config():
    """Example using default configuration from environment variables"""
    
    # Create default config
    config = IngestionPipelineConfig()
    
    # Initialize embedder
    embedder = OpenAIEmbedder()
    
    # Create pipeline
    pipeline = IngestionPipeline(
        config=config,
        generate_embeddings_func=lambda text: embedder.generate_embedding(text=text)[0]
    )
    
    # Process a file
    success, message, info = pipeline.process_single_file(
        file_path="path/to/document.pdf"
    )
    
    print(f"Success: {success}")
    print(f"Message: {message}")
    print(f"Info: {info}")


# Example 2: Custom configuration
def example_custom_config():
    """Example using custom configuration"""
    
    # Create custom Milvus config
    milvus_config = MilvusConfig(
        dbname="my_database",
        host="localhost",
        port="19530"
    )
    
    # Create custom Ingestion config
    config = IngestionPipelineConfig(
        milvus=milvus_config,
        collection_name_documents="my_documents",
        collection_name_summaries="my_summaries",
        embedding_dim=1536,
        chunk_size=1500,
        chunk_overlap=150,
        extract_images=True,
        generate_summary=True
    )
    
    # Initialize embedder
    embedder = OpenAIEmbedder()
    
    # Create pipeline with summary function
    def generate_summary(text: str) -> str:
        # Your summary generation logic here
        return f"Summary of: {text[:100]}..."
    
    pipeline = IngestionPipeline(
        config=config,
        generate_embeddings_func=lambda text: embedder.generate_embedding(text=text)[0],
        generate_summary_func=generate_summary
    )
    
    # Process a file (will use config defaults)
    success, message, info = pipeline.process_single_file(
        file_path="path/to/document.pdf"
    )
    
    # Or override config defaults
    success, message, info = pipeline.process_single_file(
        file_path="path/to/document.pdf",
        extract_images=False,  # Override config
        generate_summary=True   # Override config
    )


# Example 3: Search Pipeline Configuration
def example_search_config():
    """Example of Search Pipeline configuration"""
    
    # Create search config
    search_config = SearchPipelineConfig(
        collection_name="documents",
        default_limit=20,
        metric_type="COSINE",
        nprobe=16,
        embedding_model="text-embedding-3-small"
    )
    
    # Use this config with your search implementation
    # (You would integrate this with MilvusSearcher)
    print(f"Search config: {search_config}")


if __name__ == "__main__":
    # Run examples
    example_default_config()
    example_custom_config()
    example_search_config()

