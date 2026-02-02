# Ejemplo 3: Búsqueda Simple en Manual Técnico

## 📖 Descripción

Este ejemplo demuestra el uso de la estrategia **SIMPLE** (`simple_search`) para búsquedas directas en documentación técnica sin necesidad de selección inteligente de documentos.

## 🎯 Escenario

Tienes un manual técnico o documentación y necesitas buscar información específica de forma rápida y directa:

### Documentos
- `manual.pdf`: Manual técnico, guía de usuario, o documentación de API

### Preguntas Ejemplo
- "¿Cómo instalar el sistema?"
- "¿Cuáles son los requisitos del sistema?"
- "Explica la configuración inicial"
- "¿Cómo configurar las variables de entorno?"
- "¿Cómo solucionar errores comunes?"
- "Documentación de la API REST"

## 🔍 Cómo Funciona

La estrategia **SIMPLE** es la más directa:

1. **Búsqueda Vectorial Directa**: Convierte tu pregunta en un embedding y busca los chunks más similares en Milvus.

2. **Sin Selección Previa**: No usa LLM para seleccionar documentos, busca directamente en toda la colección (o en los filtros que proporciones).

3. **Rápida y Eficiente**: Ideal cuando:
   - Tienes un solo documento
   - Todos los documentos son relevantes
   - No necesitas filtrado inteligente
   - Quieres máxima velocidad

## 🚀 Cómo Ejecutar

### 1. Preparar el Documento

Coloca el manual PDF en la carpeta `data/`:
```
3_technical_manual/
├── data/
│   └── manual.pdf
├── upload_documents.py
├── run_example.py
└── README.md
```

### 2. Iniciar Milvus

Asegúrate de que Milvus está corriendo:
```bash
# Desde la raíz del proyecto
docker-compose up -d
```

### 3. Subir e Indexar el Manual

**⚠️ IMPORTANTE: Debes ejecutar este paso primero**

```bash
cd examples/3_technical_manual
python upload_documents.py
```

Este script:
- ✅ Verifica que el archivo existe
- ✅ Procesa el manual (extracción, chunking, embeddings)
- ✅ Indexa los chunks en Milvus
- ✅ Genera resumen del documento

### 4. Ejecutar las Búsquedas

Una vez indexado el manual, ejecuta el ejemplo:

```bash
python run_example.py
```

## 📊 Salida Esperada

```
================================================================================
EJEMPLO 3: BÚSQUEDA SIMPLE EN MANUAL TÉCNICO
================================================================================

Documentos en el sistema:
  - manual.pdf: Manual técnico o documentación

================================================================================

────────────────────────────────────────────────────────────────────────────────
CONSULTA 1: ¿Cómo instalar el sistema?
────────────────────────────────────────────────────────────────────────────────

✓ Encontrados 8 resultados:

1. Documento: manual.pdf
   Score: 0.9567
   Páginas: 5-7
   Texto: Para instalar el sistema, primero asegúrese de tener Python 3.8+
          y Docker instalados. Clone el repositorio y ejecute...

2. Documento: manual.pdf
   Score: 0.9234
   Páginas: 8-10
   Texto: Los requisitos del sistema incluyen: 4GB RAM mínimo, 10GB espacio
          en disco, conexión a internet para descargar dependencias...

...
```

## ⚙️ Configuración

El ejemplo usa la configuración más simple:

```python
SearchPipelineConfig(
    search_type=SearchType.SIMPLE,
    collection_name_documents="documents",
    search_limit=10
)
```

**Nota**: No requiere `text_model` porque no hay selección de documentos con LLM.

## 💡 Ventajas de SIMPLE

- ✅ **Velocidad**: La estrategia más rápida, sin overhead de LLM
- ✅ **Simplicidad**: Configuración mínima
- ✅ **Costo**: No consume tokens de LLM para selección
- ✅ **Directa**: Búsqueda vectorial pura
- ✅ **Flexible**: Puedes añadir filtros manualmente si lo necesitas

## 🔧 Búsqueda con Filtros Manuales (Opcional)

Aunque SIMPLE no usa LLM para selección, puedes proporcionar filtros manualmente:

```python
# Buscar solo en PDFs
results = pipeline.search(
    query_embedding=embedding,
    filter_expr='type_file == "PDF"'
)

# Buscar en un documento específico
results = pipeline.search(
    query_embedding=embedding,
    filter_expr='file_id == "manual_123"'
)

# Buscar en particiones específicas
results = pipeline.search(
    query_embedding=embedding,
    partition_names=["technical_docs"]
)
```

## 📊 Comparación con Otras Estrategias

| Característica | SIMPLE | WITH_SELECTION | WITH_SELECTION_AND_METADATA |
|----------------|--------|----------------|----------------------------|
| Velocidad | ⚡⚡⚡ Muy rápida | ⚡⚡ Rápida | ⚡ Normal |
| Precisión | ⭐⭐ Buena | ⭐⭐⭐ Muy buena | ⭐⭐⭐⭐ Excelente |
| Costo LLM | 💰 Solo embeddings | 💰💰 + Selección | 💰💰💰 + Selección + Metadata |
| Complejidad | 🟢 Baja | 🟡 Media | 🔴 Alta |
| Mejor para | 1 documento | Múltiples docs | Docs estructurados |

## 🎯 Cuándo Usar SIMPLE

**Usa SIMPLE cuando:**
- ✅ Tienes un solo documento o pocos documentos
- ✅ Todos los documentos son relevantes para las búsquedas
- ✅ Necesitas máxima velocidad
- ✅ Quieres minimizar costos de LLM
- ✅ La documentación es homogénea

**NO uses SIMPLE cuando:**
- ❌ Tienes muchos documentos de diferentes temas
- ❌ Necesitas filtrar por metadatos complejos (capítulos, páginas)
- ❌ Quieres que el sistema entienda contexto de la pregunta
- ❌ Los documentos tienen estructuras diferentes

## 🔗 Ver También

- [Ejemplo 1: CVs y Reclutamiento](../1_cv_recruitment/) - Búsqueda con selección
- [Ejemplo 2: Libro con Capítulos](../2_book_chapters/) - Búsqueda con metadatos
- [Documentación del Pipeline](../../src/search/)

