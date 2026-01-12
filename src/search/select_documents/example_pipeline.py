"""
Example usage of document selector
"""

import os
from dotenv import load_dotenv
from llms.text import OpenAITextModel
from src.search.select_documents import DocumentSelector
from src.utils import get_logger

load_dotenv()
logger = get_logger(__name__)


def example_simple_selector():
    """Basic selector example."""
    
    logger.info("=== Example 1: Simple Selector ===")
    
    # Configuration
    dbname = os.getenv("MILVUS_DB_NAME", "rag_db")
    collection_name = "summaries_collection"
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize model
    text_model = OpenAITextModel(
        api_key=openai_api_key,
        model="gpt-4o-mini"
    )
    
    # Use selector
    with DocumentSelector(
        dbname=dbname,
        collection_name=collection_name,
        text_model=text_model,
        uri=uri
    ) as selector:
        
        # Execute selection
        user_query = "I need documentation about system installation"
        selected_ids = selector.run(user_query)
        
        print("\n" + "="*80)
        print("RESULT:")
        print("="*80)
        print(f"Query: {user_query}")
        print(f"Selected documents: {selected_ids}")
        print(f"Total: {len(selected_ids)} documents")
        print("="*80 + "\n")


def example_with_details():
    """Example with complete details."""
    
    logger.info("=== Example 2: Selector with Details ===")
    
    # Configuration
    dbname = os.getenv("MILVUS_DB_NAME", "rag_db")
    collection_name = "summaries_collection"
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    text_model = OpenAITextModel(
        api_key=openai_api_key,
        model="gpt-4o-mini"
    )
    
    with DocumentSelector(
        dbname=dbname,
        collection_name=collection_name,
        text_model=text_model,
        uri=uri
    ) as selector:
        
        # Execute with details
        user_query = "I'm looking for user manuals and practical guides"
        selected_docs = selector.run_with_details(user_query)
        
        print("\n" + "="*80)
        print("SELECTED DOCUMENTS WITH DETAILS:")
        print("="*80 + "\n")
        
        for i, doc in enumerate(selected_docs, 1):
            print(f"{i}. 📄 {doc.get('file_name', 'No name')}")
            print(f"   ID: {doc.get('file_id', 'N/A')}")
            print(f"   Type: {doc.get('type_file', 'N/A')}")
            print(f"   Pages: {doc.get('total_pages', 'N/A')}")
            print(f"   Description: {doc.get('text', 'N/A')[:100]}...")
            print()
        
        print("="*80 + "\n")


def example_multiple_queries():
    """Example with multiple queries."""
    
    logger.info("=== Example 3: Multiple Queries ===")
    
    dbname = os.getenv("MILVUS_DB_NAME", "rag_db")
    collection_name = "summaries_collection"
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    text_model = OpenAITextModel(
        api_key=openai_api_key,
        model="gpt-4o-mini"
    )
    
    queries = [
        "Documents about technical architecture",
        "User guides for beginners",
        "Configuration and maintenance manuals"
    ]
    
    with DocumentSelector(
        dbname=dbname,
        collection_name=collection_name,
        text_model=text_model,
        uri=uri
    ) as selector:
        
        print("\n" + "="*80)
        print("MULTIPLE QUERIES:")
        print("="*80 + "\n")
        
        for query in queries:
            selected = selector.run(query)
            print(f"📋 {query}")
            print(f"   → {len(selected)} documents: {', '.join(selected[:3])}...")
            print()
        
        print("="*80 + "\n")


def main():
    """Executes all examples."""
    
    try:
        logger.info("Starting selector examples\n")
        
        example_simple_selector()
        print("\n" + "-"*80 + "\n")
        
        example_with_details()
        print("\n" + "-"*80 + "\n")
        
        example_multiple_queries()
        
        logger.info("\n✅ Examples completed")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
