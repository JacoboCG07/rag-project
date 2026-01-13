"""
Example usage of SearchPipeline

This example shows how to use the SearchPipeline with both search types:
1. Normal search: Direct vector search in Milvus
2. Search with selection: Document selection + search in selected documents
"""
from llms.text import OpenAITextModel
from src.search.config import SearchPipelineConfig, SearchType
from src.search.pipeline import SearchPipeline
from rag.processing.embeddings.openai_embedder import OpenAIEmbedder


def example_normal_search():
    """Example: Normal search in Milvus"""
    print("=" * 60)
    print("EXAMPLE 1: Normal Search")
    print("=" * 60)
    
    # Create configuration for normal search
    config = SearchPipelineConfig(
        search_type=SearchType.NORMAL,
        collection_name_documents="documents",
        search_limit=10
    )
    
    # Create pipeline
    with SearchPipeline(config=config) as pipeline:
        # Generate query embedding (example)
        embedder = OpenAIEmbedder(model="text-embedding-ada-002")
        query_text = "How to install the system?"
        query_embedding, _ = embedder.generate_embedding(text=query_text)
        
        # Perform search
        results = pipeline.search(
            query_embedding=query_embedding,
            partition_names=None,  # Search in all partitions
            filter_expr=None  # No additional filters
        )
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   File: {result.get('file_name', 'N/A')}")
            print(f"   Text: {result.get('text', '')[:100]}...")
    
    print("\n" + "=" * 60 + "\n")


def example_search_with_selection():
    """Example: Search with document selection"""
    print("=" * 60)
    print("EXAMPLE 2: Search with Document Selection")
    print("=" * 60)
    
    # Create LLM model for document selection
    text_model = OpenAITextModel(
        api_key="your-api-key",  # Replace with your API key
        model="gpt-4o-mini"
    )
    
    # Create configuration for search with selection
    config = SearchPipelineConfig(
        search_type=SearchType.WITH_SELECTION,
        collection_name_documents="documents",
        collection_name_summaries="summaries",
        text_model=text_model,
        search_limit=10,
        chooser_max_tokens=500,
        chooser_temperature=0.2
    )
    
    # Create pipeline
    with SearchPipeline(config=config) as pipeline:
        # Generate query embedding
        embedder = OpenAIEmbedder(model="text-embedding-ada-002")
        user_query = "Necesito documentación sobre instalación del sistema"
        query_embedding, _ = embedder.generate_embedding(text=user_query)
        
        # Perform search with selection
        # The pipeline will:
        # 1. Select relevant documents using LLM
        # 2. Search only in selected documents
        results = pipeline.search(
            query_embedding=query_embedding,
            user_query=user_query,  # Required for WITH_SELECTION mode
            partition_names=None,
            filter_expr=None
        )
        
        print(f"\nFound {len(results)} results from selected documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   File: {result.get('file_name', 'N/A')}")
            print(f"   File ID: {result.get('file_id', 'N/A')}")
            print(f"   Text: {result.get('text', '')[:100]}...")
    
    print("\n" + "=" * 60 + "\n")


def example_search_with_filters():
    """Example: Normal search with filters"""
    print("=" * 60)
    print("EXAMPLE 3: Normal Search with Filters")
    print("=" * 60)
    
    config = SearchPipelineConfig(
        search_type=SearchType.NORMAL,
        collection_name_documents="documents",
        search_limit=5
    )
    
    with SearchPipeline(config=config) as pipeline:
        embedder = OpenAIEmbedder(model="text-embedding-ada-002")
        query_text = "API documentation"
        query_embedding, _ = embedder.generate_embedding(text=query_text)
        
        # Search with filter expression
        results = pipeline.search(
            query_embedding=query_embedding,
            filter_expr='type_file == "PDF"'  # Only search in PDFs
        )
        
        print(f"\nFound {len(results)} PDF results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('file_name', 'N/A')} (Score: {result['score']:.4f})")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # Uncomment the example you want to run
    
    # Example 1: Normal search
    # example_normal_search()
    
    # Example 2: Search with document selection
    # example_search_with_selection()
    
    # Example 3: Search with filters
    # example_search_with_filters()
    
    print("Uncomment an example to run it!")

