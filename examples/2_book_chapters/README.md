# Ejemplo 2: Búsqueda en Libro con Capítulos y Metadatos

## 📚 Descripción

Este ejemplo demuestra el uso de la estrategia **WITH_SELECTION_AND_METADATA** (`document_selector_metadata_search`) para búsquedas precisas en documentos estructurados como libros.

## 🎯 Escenario

Tienes un libro con múltiples capítulos y necesitas hacer preguntas específicas sobre secciones concretas:

### Documentos
- `book_sample.pdf`: Libro con capítulos numerados, páginas y posiblemente imágenes

### Preguntas Ejemplo
- "¿Qué dice el capítulo 3 sobre metodologías de desarrollo?"
- "Busca información en las páginas 10 a 20 sobre arquitectura de software"
- "¿Qué conceptos se explican en el capítulo 1?"
- "Muéstrame información del capítulo 5 sobre testing y pruebas"
- "¿Qué imágenes hay en el capítulo 2?"

## 🔍 Cómo Funciona

La estrategia **WITH_SELECTION_AND_METADATA** funciona en cuatro pasos:

1. **Selección de Documentos**: El LLM identifica qué documentos son relevantes para la pregunta.

2. **Extracción de Metadatos**: El LLM analiza la pregunta y extrae metadatos estructurados:
   - Capítulos mencionados (ej: "capítulo 3" → `["3"]`)
   - Páginas específicas (ej: "páginas 10-20" → `[10, 11, ..., 20]`)
   - Búsqueda de imágenes (ej: "¿qué imágenes...?" → `search_image: true`)
   - Tipo de archivo (ej: "en el PDF" → `type_file: "PDF"`)

3. **Construcción de Filtros**: Genera expresiones de filtro de Milvus:
   ```python
   'file_id == "book_123" and chapters in ["3"] and pages in ["10","11",...,"20"]'
   ```

4. **Búsqueda Vectorial Filtrada**: Realiza búsqueda semántica solo en los chunks que cumplen los filtros.

## 🚀 Cómo Ejecutar

### 1. Preparar el Documento

Coloca el libro PDF en la carpeta `data/`:
```
2_book_chapters/
├── data/
│   └── book_sample.pdf
├── upload_documents.py
├── run_example.py
└── README.md
```

**Importante**: El libro debe tener estructura de capítulos y páginas.

### 2. Iniciar Milvus

Asegúrate de que Milvus está corriendo:
```bash
# Desde la raíz del proyecto
docker-compose up -d
```

### 3. Subir e Indexar el Libro

**⚠️ IMPORTANTE: Debes ejecutar este paso primero**

```bash
cd examples/2_book_chapters
python upload_documents.py
```

Este script:
- ✅ Verifica que el archivo existe
- ✅ Procesa el libro (extracción con metadatos de capítulos y páginas)
- ✅ Genera chunks con información estructurada
- ✅ Indexa en Milvus con metadatos
- ✅ Genera resumen del libro

Los chunks se indexan con campos como:
- `chapters`: "Chapter 1", "Capítulo 3", etc.
- `pages`: "1-5", "10", etc.
- `type_file`: "PDF"

### 4. Ejecutar las Búsquedas

Una vez indexado el libro, ejecuta el ejemplo:

```bash
python run_example.py
```

## 📊 Salida Esperada

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

2. Documento: book_sample.pdf
   Score: 0.9201
   Páginas: 48-50
   Capítulos: Chapter 3
   Texto: El desarrollo iterativo permite adaptarse a cambios...

...
```

## ⚙️ Configuración

El ejemplo usa la siguiente configuración:

```python
SearchPipelineConfig(
    search_type=SearchType.WITH_SELECTION_AND_METADATA,
    collection_name_documents="documents",
    collection_name_summaries="summaries",
    text_model=OpenAITextModel(model="gpt-4o-mini"),
    search_limit=10,
    chooser_max_tokens=500,
    chooser_temperature=0.2
)
```

## 💡 Ventajas de WITH_SELECTION_AND_METADATA

- ✅ **Precisión Quirúrgica**: Busca exactamente donde el usuario pide
- ✅ **Comprensión de Lenguaje Natural**: Entiende "capítulo 3", "páginas 10-20", etc.
- ✅ **Eficiencia**: Solo busca en chunks relevantes
- ✅ **Flexibilidad**: Combina múltiples filtros (capítulos + páginas + tipo)
- ✅ **Búsqueda de Imágenes**: Puede filtrar por chunks con imágenes

## 📝 Formato de Metadatos Requerido

Para que este ejemplo funcione, los chunks en Milvus deben tener:

```python
{
    "file_id": "unique_id",
    "file_name": "book_sample.pdf",
    "chapters": "Chapter 3",  # o "Capítulo 3", "Cap. 3", etc.
    "pages": "45-48",         # o "45", "45,46,47,48", etc.
    "type_file": "PDF",
    "text": "contenido del chunk...",
    "embedding": [0.1, 0.2, ...]
}
```

## 🔗 Ver También

- [Ejemplo 1: CVs y Reclutamiento](../1_cv_recruitment/) - Búsqueda con selección
- [Ejemplo 3: Manual Técnico](../3_technical_manual/) - Búsqueda simple
- [Documentación de Metadatos](../../src/search/metadata/)

