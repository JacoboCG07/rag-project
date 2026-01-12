# 🔍 Sistema de Selección de Documentos RAG

Sistema completo de selección de documentos para RAG que combina recuperación de resúmenes, generación de markdown estructurado y selección inteligente mediante LLM.

## 📋 Descripción General

El sistema funciona como una **pipeline de 3 etapas**:

```
1. Recuperación de Resúmenes → 2. Generación de Markdown → 3. Selección con LLM
     (Milvus)                      (Formato legible)            (IA inteligente)
```

## 🏗️ Arquitectura

```
src/search/
├── select_documents/           # Pipeline principal
│   ├── summary_retriever.py    # Recupera resúmenes de Milvus
│   ├── markdown_generator.py   # Genera markdown formateado
│   ├── select_documents.py     # Orquestador de la pipeline
│   └── example_pipeline.py     # Ejemplos de uso
│
└── choose_documents/           # Selección con LLM
    ├── llm_document_chooser.py # Selector inteligente con IA
    └── document_chooser_prompt.md # Prompt para el LLM
```

## 🚀 Uso Rápido

### Opción 1: Pipeline Completa (Recomendado)

```python
from llms.text import OpenAITextModel
from src.search.select_documents import DocumentSelector

# Configurar modelo LLM
text_model = OpenAITextModel(
    api_key="tu-api-key",
    model="gpt-4o-mini"
)

# Inicializar selector con LLM
with DocumentSelector(
    dbname="rag_db",
    collection_name="summaries_collection",
    uri="http://localhost:19530",
    text_model=text_model
) as selector:
    
    # Ejecutar pipeline completa en un solo paso
    selected_ids = selector.select_documents(
        user_query="Necesito documentación sobre instalación"
    )
    
    print(f"Documentos seleccionados: {selected_ids}")
```

### Opción 2: Pipeline Paso a Paso

```python
with DocumentSelector(
    dbname="rag_db",
    collection_name="summaries_collection",
    uri="http://localhost:19530",
    text_model=text_model
) as selector:
    
    # Paso 1: Obtener resúmenes
    summaries = selector.get_summaries()
    print(f"Total documentos: {len(summaries)}")
    
    # Paso 2: Generar markdown
    markdown = selector.generate_markdown(summaries)
    print(markdown)
    
    # Paso 3: Seleccionar con LLM
    selected_ids = selector.select_documents(
        user_query="Busco guías técnicas",
        summaries=summaries,
        markdown=markdown
    )
    print(f"Seleccionados: {selected_ids}")
```

### Opción 3: Con Detalles Completos

```python
# Obtener información completa de documentos seleccionados
selected_documents = selector.select_documents_with_details(
    user_query="Manuales de usuario"
)

for doc in selected_documents:
    print(f"📄 {doc['file_name']}")
    print(f"   ID: {doc['file_id']}")
    print(f"   Tipo: {doc['type_file']}")
    print(f"   Páginas: {doc['total_pages']}")
    print(f"   Descripción: {doc['text']}")
```

### Opción 4: Sin LLM (Solo Recuperación)

```python
# Si no necesitas selección inteligente
with DocumentSelector(
    dbname="rag_db",
    collection_name="summaries_collection",
    uri="http://localhost:19530"
    # NO proporcionamos text_model
) as selector:
    
    # Solo recuperar y mostrar
    summaries = selector.get_summaries()
    markdown = selector.generate_markdown(summaries)
    print(markdown)
```

## 📦 Componentes Principales

### 1. DocumentSelector

**Orquestador principal de la pipeline.**

```python
selector = DocumentSelector(
    dbname="rag_db",                    # Base de datos Milvus
    collection_name="summaries",        # Colección de resúmenes
    text_model=text_model,              # Modelo LLM (opcional)
    uri="http://localhost:19530",       # URI de Milvus
    chooser_max_tokens=500,             # Tokens para selección
    chooser_temperature=0.2             # Temperatura del LLM
)
```

**Métodos principales:**
- `get_summaries()`: Obtiene resúmenes de Milvus
- `generate_markdown()`: Genera markdown formateado
- `select_documents(user_query)`: Selecciona documentos con LLM
- `select_documents_with_details(user_query)`: Selecciona y devuelve detalles completos

### 2. SummaryRetriever

**Recupera resúmenes desde Milvus.**

```python
retriever = SummaryRetriever(
    dbname="rag_db",
    collection_name="summaries_collection",
    uri="http://localhost:19530"
)

summaries = retriever.get_all_summaries()
# Devuelve lista de diccionarios con:
# - file_id, file_name, type_file
# - total_pages, total_chapters, total_num_image
# - text (descripción)
```

### 3. MarkdownGenerator

**Genera markdown estructurado y legible.**

```python
generator = MarkdownGenerator()

# Para un documento
markdown = generator.generate_document_markdown(summary)

# Para todos los documentos
markdown = generator.generate_all_documents_markdown(summaries)
```

**Formato generado:**

```markdown
## 📄 manual_usuario_v1.pdf

- **Tipo:** PDF  
- **Páginas:** 128  
- **Capítulos:** 12  
- **Imágenes:** 54  

**Descripción:**  
Manual de usuario diseñado para explicar...
```

### 4. LLMDocumentChooser

**Selecciona documentos usando inteligencia artificial.**

```python
from src.search.choose_documents import LLMDocumentChooser

chooser = LLMDocumentChooser(
    text_model=text_model,
    max_tokens=500,
    temperature=0.2
)

selected_ids = chooser.choose_documents(
    markdown_descriptions=markdown,
    user_query="Busco guías de instalación",
    summaries=summaries
)
```

## 🔧 Configuración

### Variables de Entorno

```env
# Milvus
MILVUS_DB_NAME=rag_db
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=your_token  # Opcional

# OpenAI (si usas OpenAI)
OPENAI_API_KEY=sk-...
```

### Dependencias

```python
from llms.text import OpenAITextModel, BaseTextModel
from src.search.select_documents import DocumentSelector
from src.search.choose_documents import LLMDocumentChooser
```

## 📊 Flujo de Datos

```
Usuario: "Necesito documentación de instalación"
    ↓
┌─────────────────────────────────────────┐
│  1. SummaryRetriever                    │
│     ↓                                   │
│  Conecta a Milvus                       │
│  Query: "id >= 0"                       │
│     ↓                                   │
│  Recupera: 50 documentos                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. MarkdownGenerator                   │
│     ↓                                   │
│  Formatea cada documento:               │
│  ## 📄 doc1.pdf                         │
│  - Tipo: PDF                            │
│  - Páginas: 100                         │
│  - Descripción: ...                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. LLMDocumentChooser                  │
│     ↓                                   │
│  Envía a LLM:                           │
│  - Consulta usuario                     │
│  - Markdown de documentos               │
│     ↓                                   │
│  LLM analiza y responde:                │
│  "doc_123, doc_456, doc_789"            │
│     ↓                                   │
│  Parsea y valida IDs                    │
└─────────────────────────────────────────┘
    ↓
Resultado: ["doc_123", "doc_456", "doc_789"]
```

## 🎯 Casos de Uso

### 1. Búsqueda RAG Básica

```python
# Seleccionar documentos relevantes para una búsqueda
selected_ids = selector.select_documents(
    "¿Cómo instalar el sistema?"
)

# Usar los IDs para búsqueda vectorial en la colección principal
# ...realizar búsqueda RAG en documentos seleccionados...
```

### 2. Filtrado Pre-RAG

```python
# Reducir espacio de búsqueda antes de RAG
all_summaries = selector.get_summaries()
# 1000 documentos disponibles

selected = selector.select_documents_with_details(
    "Documentación técnica de APIs"
)
# 5 documentos relevantes seleccionados

# Ahora buscar solo en estos 5 documentos
```

### 3. Recomendación de Documentos

```python
# Recomendar documentos al usuario
recommended = selector.select_documents_with_details(
    "Soy nuevo, ¿qué debo leer primero?"
)

print("📚 Documentos recomendados:")
for doc in recommended:
    print(f"- {doc['file_name']}: {doc['text'][:100]}...")
```

### 4. Exploración Interactiva

```python
# Mostrar todos los documentos disponibles
markdown = selector.generate_markdown()
print(markdown)

# Usuario ve las opciones y refina su búsqueda
selected = selector.select_documents(
    "De estos, solo los manuales de usuario"
)
```

## 🔍 Ejemplos Avanzados

### Filtrado Manual + Selección LLM

```python
# Combinar filtrado manual con selección inteligente
summaries = selector.get_summaries()

# Filtrar solo PDFs grandes
large_pdfs = [
    s for s in summaries 
    if s['type_file'] == 'PDF' and int(s['total_pages']) > 50
]

# Seleccionar los más relevantes de estos PDFs
selected = selector.select_documents(
    user_query="Guías completas de administración",
    summaries=large_pdfs
)
```

### Pipeline Personalizada

```python
from src.search.select_documents import SummaryRetriever, MarkdownGenerator
from src.search.choose_documents import LLMDocumentChooser

# Construir pipeline personalizada
retriever = SummaryRetriever(...)
generator = MarkdownGenerator()
chooser = LLMDocumentChooser(text_model=model)

# Ejecutar con lógica personalizada
summaries = retriever.get_all_summaries()

# Filtro personalizado
filtered = custom_filter(summaries)

# Markdown personalizado
markdown = generator.generate_all_documents_markdown(filtered)

# Selección
selected = chooser.choose_documents(markdown, query, filtered)
```

## ⚙️ Parámetros de Configuración

### DocumentSelector

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `dbname` | str | - | Nombre de BD Milvus (requerido) |
| `collection_name` | str | - | Colección de resúmenes (requerido) |
| `text_model` | BaseTextModel | None | Modelo LLM para selección |
| `uri` | str | None | URI de conexión Milvus |
| `chooser_max_tokens` | int | 500 | Tokens máximos para chooser |
| `chooser_temperature` | float | 0.2 | Temperatura del LLM |

### LLMDocumentChooser

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `text_model` | BaseTextModel | - | Modelo LLM (requerido) |
| `max_tokens` | int | 500 | Tokens máximos |
| `temperature` | float | 0.2 | Temperatura (0.0-1.0) |

## 📝 Logging

Todos los componentes generan logs detallados:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Logs generados:
# INFO: Inicializando DocumentSelector
# INFO: Obteniendo resúmenes de documentos
# INFO: Se obtuvieron 50 resúmenes
# INFO: Generando markdown de documentos
# DEBUG: Paso 1/3: Obteniendo resúmenes
# DEBUG: Paso 2/3: Generando markdown
# DEBUG: Paso 3/3: Seleccionando documentos con LLM
# INFO: Pipeline completado - 3 documentos seleccionados
```

## 🐛 Manejo de Errores

```python
try:
    selected = selector.select_documents(user_query)
except ValueError as e:
    # Error de validación (ej: sin text_model)
    print(f"Error de configuración: {e}")
except Exception as e:
    # Error de ejecución (ej: Milvus no disponible)
    print(f"Error de ejecución: {e}")
```

## 🧪 Testing

Ejecuta los ejemplos:

```bash
python src/search/select_documents/example_pipeline.py
```

## 📚 Referencias

- **Milvus**: Base de datos vectorial
- **LLMs**: OpenAI GPT-4, Anthropic Claude, etc.
- **RAG**: Retrieval Augmented Generation

## 🎓 Notas Importantes

1. **Temperature**: Usa valores bajos (0.1-0.3) para selecciones más consistentes
2. **Max Tokens**: 500 es suficiente para respuestas de IDs
3. **Context Window**: El markdown completo debe caber en el contexto del LLM
4. **Validación**: Los IDs devueltos por el LLM se validan contra los disponibles
5. **Performance**: La pipeline es rápida (~2-3 segundos end-to-end)

## 🔮 Roadmap

- [ ] Soporte para selección multi-criterio
- [ ] Cache de selecciones frecuentes
- [ ] Ranking de relevancia con scores
- [ ] Selección en streaming
- [ ] Interfaz web interactiva

