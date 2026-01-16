"""
Example usage of SearchPipeline with different strategies

This example shows how to use the SearchPipeline with three search types:
1. Simple search: Direct vector search in Milvus
2. Search with selection: Document selection + search in selected documents
3. Search with selection and metadata: Document selection + search with metadata filters
"""
from llms.text import OpenAITextModel
from src.search.config import SearchPipelineConfig, SearchType
from src.search.pipeline import SearchPipeline
from rag.processing.embeddings.openai_embedder import OpenAIEmbedder


def example_simple_search():
    """Example 1: Simple search in Milvus"""
    print("=" * 60)
    print("EXAMPLE 1: Simple Search (Direct Vector Search)")
    print("=" * 60)
    
    # Create configuration for simple search
    config = SearchPipelineConfig(
        search_type=SearchType.SIMPLE,
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
    """Example 2: Search with document selection"""
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


def example_search_with_selection_and_metadata():
    """Example 3: Search with document selection and metadata filters"""
    print("=" * 60)
    print("EXAMPLE 3: Search with Document Selection + Metadata Filters")
    print("=" * 60)
    
    # Create LLM model for document selection
    text_model = OpenAITextModel(
        api_key="your-api-key",  # Replace with your API key
        model="gpt-4o-mini"
    )
    
    # Create configuration for search with selection and metadata
    config = SearchPipelineConfig(
        search_type=SearchType.WITH_SELECTION_AND_METADATA,
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
        user_query = "Documentación de instalación en PDF"
        query_embedding, _ = embedder.generate_embedding(text=user_query)
        
        # Perform search with selection and metadata
        # The pipeline will:
        # 1. Select relevant documents using LLM
        # 2. Search in selected documents AND apply metadata filters
        results = pipeline.search(
            query_embedding=query_embedding,
            user_query=user_query,  # Required for WITH_SELECTION_AND_METADATA mode
            filter_expr='type_file == "PDF"'  # Only search in PDFs
        )
        
        print(f"\nFound {len(results)} PDF results from selected documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   File: {result.get('file_name', 'N/A')}")
            print(f"   File ID: {result.get('file_id', 'N/A')}")
            print(f"   Type: {result.get('type_file', 'N/A')}")
            print(f"   Text: {result.get('text', '')[:100]}...")
    
    print("\n" + "=" * 60 + "\n")


def example_comparison():
    """Example 4: Comparison of all three strategies"""
    print("=" * 80)
    print("EXAMPLE 4: Comparison of Search Strategies")
    print("=" * 80)
    
    # Query setup
    embedder = OpenAIEmbedder(model="text-embedding-ada-002")
    user_query = "System installation guide"
    query_embedding, _ = embedder.generate_embedding(text=user_query)
    
    # Strategy 1: Simple Search
    print("\n1. SIMPLE SEARCH")
    print("-" * 80)
    config1 = SearchPipelineConfig(
        search_type=SearchType.SIMPLE,
        search_limit=5
    )
    with SearchPipeline(config=config1) as pipeline:
        results1 = pipeline.search(query_embedding=query_embedding)
        print(f"Results: {len(results1)}")
    
    # Strategy 2: With Selection
    print("\n2. WITH DOCUMENT SELECTION")
    print("-" * 80)
    text_model = OpenAITextModel(model="gpt-4o-mini")
    config2 = SearchPipelineConfig(
        search_type=SearchType.WITH_SELECTION,
        text_model=text_model,
        search_limit=5
    )
    with SearchPipeline(config=config2) as pipeline:
        results2 = pipeline.search(
            query_embedding=query_embedding,
            user_query=user_query
        )
        print(f"Results: {len(results2)}")
    
    # Strategy 3: With Selection and Metadata
    print("\n3. WITH DOCUMENT SELECTION + METADATA")
    print("-" * 80)
    config3 = SearchPipelineConfig(
        search_type=SearchType.WITH_SELECTION_AND_METADATA,
        text_model=text_model,
        search_limit=5
    )
    with SearchPipeline(config=config3) as pipeline:
        results3 = pipeline.search(
            query_embedding=query_embedding,
            user_query=user_query,
            filter_expr='type_file == "PDF"'
        )
        print(f"Results: {len(results3)}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Uncomment the example you want to run
    
    # Example 1: Simple search (direct vector search)
    # example_simple_search()
    
    # Example 2: Search with document selection
    # example_search_with_selection()
    
    # Example 3: Search with selection and metadata filters
    # example_search_with_selection_and_metadata()
    
    # Example 4: Compare all strategies
    # example_comparison()
    
    print("Uncomment an example to run it!")