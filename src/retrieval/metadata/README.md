# Metadata Module

Módulo para extracción y construcción de filtros de metadata para búsquedas en Milvus.

---

## 🎯 Objetivo

Este módulo permite:
1. **Extraer metadata** del query del usuario usando un LLM
2. **Construir filtros** de Milvus desde la metadata extraída

---

## 📦 Componentes

### 1. `MetadataExtractor`

Extrae metadata relevante de documentos usando un LLM.

```python
from src.retrieval.metadata import MetadataExtractor
from llms.text import OpenAITextModel

text_model = OpenAITextModel(model="gpt-4o-mini")
extractor = MetadataExtractor(text_model)

metadata_dict = extractor.extract(
    user_query="Buscar en páginas 1-5 del manual",
    markdown_documents=markdown,
    documents_info=summaries
)

# Resultado:
# {
#     "doc_001": {
#         "pages": [1, 2, 3, 4, 5],
#         "chapters": null,
#         "search_image": False,
#         "num_image": null,
#         "file_type": "PDF"
#     }
# }
```

### 2. `MetadataFilterBuilder`

Construye expresiones de filtro de Milvus desde metadata extraída.

```python
from src.retrieval.metadata import MetadataFilterBuilder

builder = MetadataFilterBuilder()

# Filtro para un documento
filter_expr = builder.build_filter_for_document(
    file_id="doc_001",
    metadata={
        "pages": [1, 2, 3],
        "chapters": null,
        "search_image": False,
        "num_image": null,
        "file_type": "PDF"
    }
)

# Resultado: 'file_id == "doc_001" and pages in ["1", "2", "3"] and file_type == "PDF"'

# Filtro combinado para múltiples documentos
combined_filter = builder.build_combined_filter(metadata_dict)
```

---

## 🔄 Flujo Completo

```
User Query: "Buscar en páginas 1-5 del manual de instalación"
    +
Markdown Documents + Documents Info
    ↓
┌──────────────────────────────┐
│  MetadataExtractor (LLM)     │
│  - Analiza query + docs      │
│  - Extrae metadata relevante │
└──────────────────────────────┘
    ↓
{
  "doc_001": {
    "pages": [1, 2, 3, 4, 5],
    "type_file": "PDF",
    ...
  }
}
    ↓
┌──────────────────────────────┐
│  MetadataFilterBuilder       │
│  - Construye expresión       │
└──────────────────────────────┘
    ↓
'file_id == "doc_001" and pages in ["1","2","3","4","5"] and file_type == "PDF"'
    ↓
Se usa en Milvus search
```

---

## 📊 Metadata Soportada

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `pages` | List[int] o null | Números de página específicos |
| `chapters` | List[str] o null | Capítulos mencionados |
| `search_image` | boolean | Si busca imágenes |
| `num_image` | List[int] o null | Números de imagen específicos |
| `file_type` | string | Tipo de archivo (PDF, TXT, etc.) |

---

## 💻 Ejemplo Completo

Ver [`examples/metadata_extraction_example.py`](examples/metadata_extraction_example.py)

---

## 🔧 Configuración

```python
# Personalizar extractor
extractor = MetadataExtractor(
    text_model=text_model,
    max_tokens=500,      # Tokens para respuesta LLM
    temperature=0.2       # Temperatura (0.0-1.0)
)
```

---

## 🧩 Integración con Estrategias

```python
# En DocumentSelectorMetadataSearchStrategy

from src.retrieval.metadata import MetadataExtractor, MetadataFilterBuilder

# 1. Extraer metadata
metadata_dict = metadata_extractor.extract(
    user_query=user_query,
    markdown_documents=markdown,
    documents_info=summaries
)

# 2. Construir filtro
filter_builder = MetadataFilterBuilder()
combined_filter = filter_builder.build_combined_filter(metadata_dict)

# 3. Buscar con filtro
results = searcher.search(
    query_embedding=embedding,
    filter_expr=combined_filter
)
```

---

## 📝 Prompt

El prompt para el LLM se encuentra en [`prompt.md`](prompt.md) y define:
- Instrucciones de extracción
- Formato de respuesta JSON
- Ejemplos de uso
- Reglas y validaciones

---

## 🧪 TODO

- [ ] Implementar `_build_prompt()` en extractor
- [ ] Implementar `_parse_response()` completo
- [ ] Validaciones de metadata
- [ ] Tests unitarios
- [ ] Tests funcionales con LLM real
- [ ] Manejo de errores robusto

---

**Última actualización**: 2026-01-14

