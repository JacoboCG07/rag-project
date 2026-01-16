# 🏗️ Arquitectura Técnica - Estrategias de Búsqueda

Documentación técnica sobre la implementación del sistema de búsqueda con patrón Strategy.

---

## 🎨 Patrón de Diseño

**Strategy Pattern** - Cada estrategia encapsula un algoritmo de búsqueda diferente.

```
SearchPipeline (Context)
    │
    └─── SearchStrategy (Abstract)
            ├─── SimpleSearchStrategy
            ├─── DocumentSelectorSearchStrategy
            └─── DocumentSelectorMetadataSearchStrategy
```

---

## 📐 Arquitectura General

```
┌──────────────────────────────────────────┐
│         SearchPipeline                   │
│  (Orquestador principal)                 │
│                                          │
│  - Recibe configuración                  │
│  - Crea estrategia apropiada (factory)   │
│  - Delega búsqueda a estrategia          │
└──────────────────────────────────────────┘
              │
              ├─► Strategy 1: SimpleSearchStrategy
              ├─► Strategy 2: DocumentSelectorSearchStrategy
              └─► Strategy 3: DocumentSelectorMetadataSearchStrategy
```

---

## 🔍 Estrategia 1: SimpleSearchStrategy

### Diagrama de Flujo

```
query_embedding
    ↓
MilvusSearcher.connect()
    ↓
collection.search(
    data=query_embedding,
    limit=10,
    anns_field="text_embedding"
)
    ↓
results (ordenados por score)
    ↓
MilvusSearcher.disconnect()
    ↓
return results
```

### Componentes

```python
SimpleSearchStrategy
├── self.searcher: MilvusSearcher
│   └── Conexión directa a Milvus
└── self.config: SearchPipelineConfig
    └── Parámetros (limit, collection_name, etc.)
```

### Métricas Técnicas

```
Latencia:
├─ Generar embedding:     20-50ms
├─ Búsqueda Milvus:       30-150ms
└─ Total:                 50-200ms

Throughput:               1000+ búsquedas/s
CPU:                      Bajo (~5% por búsqueda)
RAM:                      Bajo (~10MB por búsqueda)
Red:                      1-5KB por búsqueda
```

---

## 🧠 Estrategia 2: DocumentSelectorSearchStrategy

### Diagrama de Flujo

```
user_query + query_embedding
    │
    ├─── PASO 1: Selección de Documentos
    │    │
    │    └─► DocumentSelector.run(user_query)
    │         │
    │         ├─► SummaryRetriever.get_all_summaries()
    │         │   └─ Milvus "summaries" collection
    │         │
    │         ├─► MarkdownGenerator.generate_markdown()
    │         │   └─ Formato estructurado para LLM
    │         │
    │         └─► LLMDocumentChooser.choose_documents()
    │             ├─ Prompt + markdown → LLM
    │             └─ Parse response → ["doc1", "doc3"]
    │
    └─── PASO 2: Búsqueda en Seleccionados
         │
         └─► MilvusSearcher.search_by_partition()
             ├─ For each selected_file_id:
             │  └─ search(partition_name=file_id)
             │
             ├─ Consolidar resultados
             ├─ Ordenar por score
             └─ Limitar a search_limit
```

### Componentes

```python
DocumentSelectorSearchStrategy
├── self.searcher: MilvusSearcher
│   └── Búsqueda en colección "documents"
│
└── self.document_selector: DocumentSelector
    ├── retriever: SummaryRetriever
    │   └── Conexión a colección "summaries"
    │
    ├── markdown_generator: MarkdownGenerator
    │   └── Formateador de resúmenes
    │
    └── chooser: LLMDocumentChooser
        └── text_model: BaseTextModel (OpenAI, Anthropic, etc.)
```

### Métricas Técnicas

```
Latencia (desglose):
├─ Recuperar resúmenes:    100-200ms
├─ Generar markdown:       10-50ms
├─ Llamada LLM:           1000-2000ms
├─ Parse respuesta:        10-20ms
├─ Generar embedding:      20-50ms
├─ Búsqueda particiones:  200-500ms
└─ Total:                 1500-3000ms

Costos por búsqueda:
├─ GPT-4o-mini:           $0.001
├─ GPT-4o:                $0.005
└─ Claude Sonnet:         $0.003

Throughput:               10-50 búsquedas/s (limitado por LLM)
CPU:                      Medio (~15% por búsqueda)
RAM:                      Medio (~50MB por búsqueda)
Red:                      10-50KB por búsqueda (+ LLM API)
```

---

## 🔌 Interfaces y Contratos

### SearchStrategy (Abstract Base Class)

```python
class SearchStrategy(ABC):
    def __init__(self, config: SearchPipelineConfig)
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        user_query: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]
    
    @abstractmethod
    def close(self) -> None
```

### Estructura de Resultado

```python
Result = {
    "id": int,                    # ID del chunk en Milvus
    "score": float,               # Score de similitud (0-1)
    "text": str,                  # Contenido del chunk
    "file_id": str,               # ID del documento padre
    "file_name": str,             # Nombre del archivo
    "source_id": str,             # ID de fuente
    "pages": str,                 # Páginas (ej: "1-5")
    "chapters": str,              # Capítulos
    "type_file": str              # Tipo (PDF, TXT, etc.)
}
```

---

## 🔄 Factory Pattern

```python
# SearchPipeline._create_strategy()

def _create_strategy(self, config: SearchPipelineConfig) -> SearchStrategy:
    """Factory method para crear la estrategia apropiada"""
    
    if config.search_type == SearchType.SIMPLE:
        return SimpleSearchStrategy(config)
    
    elif config.search_type == SearchType.WITH_SELECTION:
        return DocumentSelectorSearchStrategy(config)
    
    elif config.search_type == SearchType.WITH_SELECTION_AND_METADATA:
        return DocumentSelectorMetadataSearchStrategy(config)
    
    else:
        raise ValueError(f"Unknown search_type: {config.search_type}")
```

---

## 📦 Dependencias

```
SimpleSearchStrategy
└── MilvusSearcher
    └── pymilvus

DocumentSelectorSearchStrategy
├── MilvusSearcher
│   └── pymilvus
└── DocumentSelector
    ├── SummaryRetriever
    │   └── pymilvus
    ├── MarkdownGenerator
    └── LLMDocumentChooser
        └── BaseTextModel (llms.text)
            ├── openai (OpenAI)
            └── anthropic (Anthropic)
```

---

## 🗄️ Esquema de Datos en Milvus

### Colección "documents" (chunks)

```python
Schema:
├── id: INT64 (primary key)
├── text_embedding: FLOAT_VECTOR[1536]  # OpenAI ada-002
├── text: VARCHAR
├── file_id: VARCHAR
├── file_name: VARCHAR
├── source_id: VARCHAR
├── pages: VARCHAR
├── chapters: VARCHAR
└── type_file: VARCHAR

Partitions:
└── Por file_id (ej: "doc_123", "doc_456")
```

### Colección "summaries" (resúmenes)

```python
Schema:
├── id: INT64 (primary key)
├── file_id: VARCHAR (único por documento)
├── file_name: VARCHAR
├── text: VARCHAR (resumen del documento)
├── type_file: VARCHAR
├── total_pages: INT
├── total_chapters: INT
└── total_num_image: INT
```

---

## 🔧 Configuración

```python
SearchPipelineConfig:
├── search_type: SearchType
│   ├── SIMPLE
│   ├── WITH_SELECTION
│   └── WITH_SELECTION_AND_METADATA
│
├── milvus: MilvusConfig
│   ├── dbname: str
│   ├── alias: str
│   ├── uri: Optional[str]
│   ├── token: Optional[str]
│   ├── host: Optional[str]
│   └── port: Optional[str]
│
├── collection_name_documents: str
├── collection_name_summaries: str
├── search_limit: int (default: 10)
│
└── text_model: Optional[BaseTextModel]
    └── Requerido para WITH_SELECTION
```

---

## 🧪 Testing

```python
# Unit tests
tests/unit_tests/search/strategies/
├── test_simple_search.py
├── test_document_selector_search.py
└── test_base_strategy.py

# Functional tests
tests/functional_tests/search/
├── test_search_pipeline.py
└── test_document_selection.py

# Integration tests
tests/integration_tests/
└── test_end_to_end_search.py
```

---

## 📊 Logging

```python
# Estructura de logs

INFO: Initializing SimpleSearchStrategy
DEBUG: Connecting to Milvus (db=rag_db, collection=documents)
DEBUG: Executing search (limit=10, filter=None)
INFO: Search completed (results=8, time=150ms)
DEBUG: Disconnecting from Milvus

INFO: Initializing DocumentSelectorSearchStrategy
DEBUG: Initializing DocumentSelector
INFO: Step 1: Selecting relevant documents
DEBUG: Retrieved 50 summaries (time=120ms)
DEBUG: Generated markdown (size=15KB)
INFO: LLM selection (model=gpt-4o-mini)
INFO: Documents selected: 3 (doc1, doc3, doc5)
INFO: Step 2: Searching in selected documents
DEBUG: Searching partition doc1 (found=8)
DEBUG: Searching partition doc3 (found=5)
DEBUG: Searching partition doc5 (found=3)
INFO: Search completed (total=16, final=10, time=2.3s)
```

---

## 🔒 Consideraciones de Seguridad

### SimpleSearchStrategy
- ✅ No envía datos a APIs externas
- ✅ Todo local (Milvus)
- ⚠️ Validar filter_expr para evitar injection

### DocumentSelectorSearchStrategy
- ⚠️ Resúmenes se envían al LLM (considerar datos sensibles)
- ✅ Opción: Usar LLM local (Ollama, llama.cpp)
- ✅ Opción: Filtrar información sensible de resúmenes
- ⚠️ Rate limiting en API del LLM

---

## 🚀 Optimizaciones

### SimpleSearchStrategy
```python
# Cache de embeddings
# Connection pooling en Milvus
# Batch queries
```

### DocumentSelectorSearchStrategy
```python
# Cache de selecciones frecuentes
# Parallel partition search
# Stream LLM responses
# Comprimir markdown enviado al LLM
```

---

## 📚 Referencias

- [Milvus Documentation](https://milvus.io/docs)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

---

**Última actualización**: 2026-01-14