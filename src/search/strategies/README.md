# Estrategias de Búsqueda

Sistema de búsqueda con múltiples estrategias implementadas usando el patrón Strategy.

---

## 📚 Documentación

### 🎯 [**GUIA_USO.md**](GUIA_USO.md) - Documentación Funcional
**Para usuarios que quieren decidir qué estrategia usar**

- ✅ Ventajas y desventajas de cada estrategia
- 🎯 Cuándo usar cada una (casos de uso)
- 📊 Comparación práctica
- 💰 Comparación de costos
- ⚡ Comparación de velocidad
- 🎨 Árbol de decisión
- 💻 Ejemplos de código

**→ [Ir a Guía de Uso](GUIA_USO.md)**

---

### 🏗️ [**ARQUITECTURA.md**](ARQUITECTURA.md) - Documentación Técnica
**Para desarrolladores que quieren entender cómo funciona**

- 🎨 Patrón Strategy explicado
- 📐 Diagramas de arquitectura
- 🔄 Flujos de datos internos
- 📊 Métricas de rendimiento
- 🔌 Interfaces y contratos
- 🗄️ Esquemas de Milvus
- 🧪 Testing
- 🔒 Consideraciones de seguridad

**→ [Ir a Arquitectura](ARQUITECTURA.md)**

---

## 🚀 Quick Start

### Búsqueda Simple
```python
from src.search import SearchPipeline, SearchPipelineConfig, SearchType

config = SearchPipelineConfig(search_type=SearchType.SIMPLE)
with SearchPipeline(config=config) as pipeline:
    results = pipeline.search(query_embedding=embedding)
```

### Búsqueda con Selección
```python
from llms.text import OpenAITextModel

config = SearchPipelineConfig(
    search_type=SearchType.WITH_SELECTION,
    text_model=OpenAITextModel(model="gpt-4o-mini")
)

with SearchPipeline(config=config) as pipeline:
    results = pipeline.search(
        query_embedding=embedding,
        user_query="Tu consulta aquí"
    )
```

---

## 📊 Comparación Rápida

| Característica | Simple | Con Selección |
|---------------|--------|---------------|
| Velocidad | ⚡⚡⚡ | ⚡⚡ |
| Precisión | ⭐⭐ | ⭐⭐⭐⭐ |
| Costo | Gratis | ~$0.001/búsqueda |
| Mejor para | < 100 docs | > 100 docs |

[Ver comparación completa →](GUIA_USO.md#comparación-rápida)

---

## 🗂️ Archivos

```
strategies/
├── README.md                          ← Este archivo (índice)
├── GUIA_USO.md                       ← Documentación funcional
├── ARQUITECTURA.md                    ← Documentación técnica
│
├── base.py                           ← Clase abstracta SearchStrategy
├── simple_search.py                  ← SimpleSearchStrategy
├── document_selector_search.py       ← DocumentSelectorSearchStrategy
├── document_selector_metadata_search.py
└── __init__.py
```

---

## 📖 Ver También

- [Ejemplos completos](../examples/search_pipeline_example.py)
- [Configuración](../config.py)
- [DocumentSelector](../document_selection/)

---

**Última actualización**: 2026-01-14

