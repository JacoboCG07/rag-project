# Ejemplo 2: Libro con Capítulos (búsqueda con metadatos)

## 📋 Descripción

Este ejemplo demuestra el uso de la estrategia **WITH_SELECTION_AND_METADATA** (`document_selector_metadata_search`) para búsquedas precisas en documentos estructurados como libros (capítulos, páginas).

## 🎯 Escenario

Tienes un libro con múltiples capítulos y necesitas hacer preguntas específicas sobre secciones concretas:

### Documentos
- `book_sample.pdf`: Libro con capítulos numerados, páginas y posiblemente imágenes

### Preguntas ejemplo
- "¿Qué dice el capítulo 3 sobre metodologías de desarrollo?"
- "Busca información en las páginas 10 a 20 sobre arquitectura de software"
- "¿Qué conceptos se explican en el capítulo 1?"
- "Muéstrame información del capítulo 5 sobre testing y pruebas"
- "¿Qué imágenes hay en el capítulo 2?"

## 🔍 Cómo funciona

La estrategia **WITH_SELECTION_AND_METADATA** funciona en varios pasos:

1. **Selección de documentos**: El LLM identifica qué documentos son relevantes para la pregunta.
2. **Extracción de metadatos**: El LLM analiza la pregunta y extrae metadatos (capítulos, páginas, búsqueda de imágenes, tipo de archivo).
3. **Construcción de filtros**: Se generan expresiones de filtro de Milvus para buscar solo en las secciones indicadas.
4. **Búsqueda vectorial filtrada**: Búsqueda semántica solo en los chunks que cumplen los filtros.

## 🚀 Cómo ejecutar

### 1. Preparar el documento

Coloca el libro PDF en la carpeta `data/`:

```
2_book_chapters/
├── data/
│   └── book_sample.pdf
├── upload_documents.py
├── run_retrieval.py
├── run_chatbot.py
└── README.md
```

### 2. Iniciar Milvus

```bash
# Desde la raíz del proyecto
docker-compose up -d
```

### 3. Subir e indexar el libro

**⚠️ IMPORTANTE: Debes ejecutar este paso primero**

```bash
cd examples/2_book_chapters
python upload_documents.py
```

Este script:
- ✅ Verifica que el archivo existe
- ✅ Procesa el libro (extracción, chunking con metadatos de capítulos y páginas)
- ✅ Indexa en Milvus (particiones `documents` y `summaries`)
- ✅ Genera resumen del libro

### 4. Ejecutar las búsquedas

Una vez indexado el libro:

```bash
python run_retrieval.py
```

### 5. Ejecutar el chatbot (opcional)

Para respuestas generadas por el LLM usando búsqueda con metadatos:

```bash
python run_chatbot.py
```

## 📊 Salida esperada

```
================================================================================
EJEMPLO 2: BÚSQUEDA EN LIBRO CON CAPÍTULOS Y METADATOS
================================================================================

Documentos en el sistema:
  - book_sample.pdf: Libro con múltiples capítulos

================================================================================

────────────────────────────────────────────────────────────────────────────────
CONSULTA 1: ¿Qué dice el capítulo 3 sobre metodologías de desarrollo?
────────────────────────────────────────────────────────────────────────────────

✓ Encontrados 4 resultados:

1. Documento: book_sample.pdf
   Score: 0.9456
   Páginas: 45-48
   Capítulos: Chapter 3
   Texto: Las metodologías ágiles como Scrum y Kanban han revolucionado...
...
```

## ⚙️ Configuración

El ejemplo usa la siguiente configuración:

```python
SearchPipelineConfig(
    search_type=SearchType.WITH_SELECTION_AND_METADATA,
    collection_name="book_chapters",
    text_model=OpenAITextModel(model="gpt-4o-mini"),
    search_limit=10,
    chooser_max_tokens=500,
    chooser_temperature=0.2,
)
# La colección book_chapters tiene dos particiones: 'documents' y 'summaries'
```

## 💡 Ventajas de WITH_SELECTION_AND_METADATA

- ✅ **Precisión**: Busca exactamente en capítulos/páginas indicados
- ✅ **Lenguaje natural**: Entiende "capítulo 3", "páginas 10-20", etc.
- ✅ **Eficiencia**: Solo busca en chunks relevantes
- ✅ **Flexibilidad**: Combina filtros (capítulos + páginas + tipo)
- ✅ **Imágenes**: Puede filtrar por chunks con imágenes

## 🔗 Ver también

- [Ejemplo 1: El Principito](../1_principito/) - Búsqueda simple y chatbot
- [Ejemplo 4: Reclutamiento con CVs](../4_cv_recruitment/) - Búsqueda con selección de documentos
- [Documentación del pipeline](../../src/retrieval/)
