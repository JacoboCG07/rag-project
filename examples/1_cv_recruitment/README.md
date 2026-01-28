# Ejemplo 1: Sistema de Reclutamiento con CVs

## 📋 Descripción

Este ejemplo demuestra el uso de la estrategia **WITH_SELECTION** (`document_selector_search`) en un caso de uso real de reclutamiento.

## 🎯 Escenario

Tienes una propuesta de trabajo y varios CVs de candidatos. Necesitas hacer preguntas para encontrar al mejor candidato:

### Documentos
- `job_proposal.pdf`: Propuesta de trabajo con descripción del puesto, requisitos técnicos, experiencia requerida, etc.
- `cv_candidate_1.pdf`: CV del Candidato 1
- `cv_candidate_2.pdf`: CV del Candidato 2
- `cv_candidate_3.pdf`: CV del Candidato 3

### Preguntas Ejemplo
- "¿Qué candidato tiene experiencia en Python y desarrollo backend?"
- "¿Quién cumple mejor con los requisitos técnicos de la propuesta de trabajo?"
- "¿Qué candidato tiene más años de experiencia profesional?"
- "¿Algún candidato tiene experiencia con bases de datos vectoriales o Milvus?"
- "¿Qué formación académica tienen los candidatos?"

## 🔍 Cómo Funciona

La estrategia **WITH_SELECTION** funciona en dos pasos:

1. **Selección de Documentos**: El LLM analiza tu pregunta y los resúmenes de todos los documentos, seleccionando cuáles son relevantes (por ejemplo, solo los CVs si preguntas sobre experiencia).

2. **Búsqueda Vectorial**: Realiza búsqueda semántica solo en los documentos seleccionados, mejorando la precisión y reduciendo ruido.

## 🚀 Cómo Ejecutar

### 1. Preparar los Documentos

Coloca los 4 archivos PDF en la carpeta `data/`:
```
1_cv_recruitment/
├── data/
│   ├── job_proposal.pdf
│   ├── cv_candidate_1.pdf
│   ├── cv_candidate_2.pdf
│   └── cv_candidate_3.pdf
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

### 3. Subir e Indexar los Documentos

**⚠️ IMPORTANTE: Debes ejecutar este paso primero**

```bash
cd examples/1_cv_recruitment
python upload_documents.py
```

Este script:
- ✅ Verifica que los archivos existen
- ✅ Procesa cada documento (extracción, chunking, embeddings)
- ✅ Indexa los chunks en Milvus
- ✅ Genera resúmenes para la colección de summaries

### 4. Ejecutar las Búsquedas

Una vez indexados los documentos, ejecuta el ejemplo:

```bash
python run_example.py
```

## 📊 Salida Esperada

```
================================================================================
EJEMPLO 1: SISTEMA DE RECLUTAMIENTO CON CVs
================================================================================

Documentos en el sistema:
  - job_proposal.pdf: Propuesta de trabajo con requisitos
  - cv_candidate_1.pdf: CV del Candidato 1
  - cv_candidate_2.pdf: CV del Candidato 2
  - cv_candidate_3.pdf: CV del Candidato 3

================================================================================

────────────────────────────────────────────────────────────────────────────────
CONSULTA 1: ¿Qué candidato tiene experiencia en Python y desarrollo backend?
────────────────────────────────────────────────────────────────────────────────

✓ Encontrados 5 resultados:

1. Documento: cv_candidate_2.pdf
   Score: 0.9234
   Páginas: 1-2
   Texto: Experiencia profesional: Senior Backend Developer en TechCorp...

2. Documento: cv_candidate_1.pdf
   Score: 0.8876
   Páginas: 1
   Texto: Habilidades técnicas: Python, Django, FastAPI, PostgreSQL...

...
```

## ⚙️ Configuración

El ejemplo usa la siguiente configuración:

```python
SearchPipelineConfig(
    search_type=SearchType.WITH_SELECTION,
    collection_name_documents="documents",
    collection_name_summaries="summaries",
    text_model=OpenAITextModel(model="gpt-4o-mini"),
    search_limit=10,
    chooser_max_tokens=500,
    chooser_temperature=0.2
)
```

## 💡 Ventajas de WITH_SELECTION

- ✅ **Precisión**: Solo busca en documentos relevantes
- ✅ **Eficiencia**: Reduce el espacio de búsqueda
- ✅ **Inteligencia**: El LLM entiende el contexto de la pregunta
- ✅ **Flexibilidad**: Funciona con cualquier tipo de documento

## 🔗 Ver También

- [Ejemplo 2: Libro con Capítulos](../2_book_chapters/) - Búsqueda con metadatos
- [Ejemplo 3: Manual Técnico](../3_technical_manual/) - Búsqueda simple
- [Documentación del Pipeline](../../src/search/)

