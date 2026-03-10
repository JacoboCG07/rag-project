"""
Example: Metadata Extraction and Filter Building

Demonstrates how to use MetadataExtractor and MetadataFilterBuilder
to extract metadata from user queries and build Milvus filters.
"""
from llms.text import OpenAITextModel
from src.retrieval.metadata import MetadataExtractor, MetadataFilterBuilder


def example_basic_extraction():
    """
    Example 1: Basic metadata extraction.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Metadata Extraction")
    print("=" * 80)
    
    # Setup
    text_model = OpenAITextModel(
        api_key="your-api-key",
        model="gpt-4o-mini"
    )
    
    extractor = MetadataExtractor(text_model)
    
    # User query
    user_query = "Buscar en las páginas 1 a 5 del manual de instalación"
    
    # Markdown documents (simulated)
    markdown_documents = """
## Documento manual_instalacion:

- "file_id": "doc_001",
- "file_name": "manual_instalacion.pdf",
- "file_type": "PDF",
- "pages": "50",
- "chapters": "5",
- "full_images": "10",
- "text": "Manual completo de instalación del sistema."

---

## Documento guia_usuario:

- "file_id": "doc_002",
- "file_name": "guia_usuario.pdf",
- "file_type": "PDF",
- "pages": "120",
- "chapters": "12",
- "full_images": "45",
- "text": "Guía de usuario completa."
"""
    
    # Documents info
    documents_info = [
        {
            "file_id": "doc_001",
            "file_name": "manual_instalacion.pdf",
            "file_type": "PDF",
            "pages": "50",
            "chapters": "5",
            "full_images": "10",
            "text": "Manual completo de instalación del sistema."
        },
        {
            "file_id": "doc_002",
            "file_name": "guia_usuario.pdf",
            "file_type": "PDF",
            "pages": "120",
            "chapters": "12",
            "full_images": "45",
            "text": "Guía de usuario completa."
        }
    ]
    
    # Extract metadata
    metadata_dict = extractor.extract(
        user_query=user_query,
        markdown_documents=markdown_documents,
        documents_info=documents_info
    )
    
    print("\n📋 Metadata Extraída:")
    for file_id, metadata in metadata_dict.items():
        print(f"\n  {file_id}:")
        print(f"    pages: {metadata.get('pages')}")
        print(f"    chapters: {metadata.get('chapters')}")
        print(f"    search_image: {metadata.get('search_image')}")
        print(f"    num_image: {metadata.get('num_image')}")
        print(f"    file_type: {metadata.get('file_type')}")
    
    print("\n" + "=" * 80 + "\n")
    return metadata_dict


def example_filter_building(metadata_dict):
    """
    Example 2: Building filters from extracted metadata.
    """
    print("=" * 80)
    print("EXAMPLE 2: Building Milvus Filters")
    print("=" * 80)
    
    builder = MetadataFilterBuilder()
    
    # Single document filter
    print("\n🔧 Filtro para un documento:")
    for file_id, metadata in metadata_dict.items():
        filter_expr = builder.build_filter_for_document(file_id, metadata)
        print(f"\n  {file_id}:")
        print(f"  {filter_expr}")
    
    # Combined filter
    print("\n🔧 Filtro combinado:")
    combined_filter = builder.build_combined_filter(metadata_dict)
    print(f"\n  {combined_filter}")
    
    # Validate filter
    print("\n✅ Validación:")
    is_valid = builder.validate_filter(combined_filter)
    print(f"  Filter is valid: {is_valid}")
    
    print("\n" + "=" * 80 + "\n")
    return combined_filter


def example_with_images():
    """
    Example 3: Extracting metadata for image search.
    """
    print("=" * 80)
    print("EXAMPLE 3: Image Search Metadata")
    print("=" * 80)
    
    text_model = OpenAITextModel(
        api_key="your-api-key",
        model="gpt-4o-mini"
    )
    
    extractor = MetadataExtractor(text_model)
    builder = MetadataFilterBuilder()
    
    # Query looking for images
    user_query = "Mostrar diagramas de arquitectura en las imágenes 1 y 2"
    
    markdown_documents = """
## Documento architecture:

- "file_id": "doc_003",
- "file_name": "architecture_diagrams.pdf",
- "file_type": "PDF",
- "pages": "30",
- "chapters": "3",
- "full_images": "15",
- "text": "Diagramas de arquitectura del sistema."
"""
    
    documents_info = [
        {
            "file_id": "doc_003",
            "file_name": "architecture_diagrams.pdf",
            "file_type": "PDF",
            "pages": "30",
            "chapters": "3",
            "full_images": "15",
            "text": "Diagramas de arquitectura del sistema."
        }
    ]
    
    # Extract
    metadata_dict = extractor.extract(
        user_query=user_query,
        markdown_documents=markdown_documents,
        documents_info=documents_info
    )
    
    print("\n📋 Metadata (con imágenes):")
    for file_id, metadata in metadata_dict.items():
        print(f"\n  {file_id}:")
        print(f"    search_image: {metadata.get('search_image')}")
        print(f"    num_image: {metadata.get('num_image')}")
    
    # Build filter
    combined_filter = builder.build_combined_filter(metadata_dict)
    print(f"\n🔧 Filter: {combined_filter}")
    
    print("\n" + "=" * 80 + "\n")


def example_with_chapters():
    """
    Example 4: Extracting metadata for chapter search.
    """
    print("=" * 80)
    print("EXAMPLE 4: Chapter Search Metadata")
    print("=" * 80)
    
    text_model = OpenAITextModel(
        api_key="your-api-key",
        model="gpt-4o-mini"
    )
    
    extractor = MetadataExtractor(text_model)
    builder = MetadataFilterBuilder()
    
    # Query with chapters
    user_query = "Información del capítulo 2 y 3 sobre instalación"
    
    markdown_documents = """
## Documento manual:

- "file_id": "doc_001",
- "file_name": "manual_instalacion.pdf",
- "file_type": "PDF",
- "pages": "50",
- "chapters": "5",
- "full_images": "10",
- "text": "Manual de instalación. Capítulos: 1-Intro, 2-Requisitos, 3-Instalación, 4-Config, 5-Troubleshooting"
"""
    
    documents_info = [
        {
            "file_id": "doc_001",
            "file_name": "manual_instalacion.pdf",
            "file_type": "PDF",
            "pages": "50",
            "chapters": "5",
            "full_images": "10",
            "text": "Manual de instalación. Capítulos: 1-Intro, 2-Requisitos, 3-Instalación, 4-Config, 5-Troubleshooting"
        }
    ]
    
    # Extract
    metadata_dict = extractor.extract(
        user_query=user_query,
        markdown_documents=markdown_documents,
        documents_info=documents_info
    )
    
    print("\n📋 Metadata (con capítulos):")
    for file_id, metadata in metadata_dict.items():
        print(f"\n  {file_id}:")
        print(f"    chapters: {metadata.get('chapters')}")
    
    # Build filter
    combined_filter = builder.build_combined_filter(metadata_dict)
    print(f"\n🔧 Filter: {combined_filter}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Run examples
    print("\n🚀 Metadata Extraction Examples\n")
    
    # Uncomment to run examples
    
    # Example 1: Basic extraction
    # metadata_dict = example_basic_extraction()
    
    # Example 2: Filter building
    # example_filter_building(metadata_dict)
    
    # Example 3: Image search
    # example_with_images()
    
    # Example 4: Chapter search
    # example_with_chapters()
    
    print("✅ Uncomment examples to run them!")

