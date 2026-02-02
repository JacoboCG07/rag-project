# 📚 Ejemplos de Uso - RAG Project

Esta carpeta contiene ejemplos prácticos que demuestran las tres estrategias de búsqueda del sistema RAG.

## 🎯 Descripción General

Cada ejemplo muestra un caso de uso real con documentos específicos y consultas típicas. Los ejemplos están diseñados para ser ejecutados de forma independiente y demostrar las capacidades únicas de cada estrategia.

## 📂 Estructura de Ejemplos

```
examples/
├── README.md (este archivo)
├── 1_cv_recruitment/          # Ejemplo 1: Reclutamiento con CVs
│   ├── data/
│   │   ├── job_proposal.pdf
│   │   ├── cv_candidate_1.pdf
│   │   ├── cv_candidate_2.pdf
│   │   └── cv_candidate_3.pdf
│   ├── run_example.py
│   └── README.md
├── 2_book_chapters/           # Ejemplo 2: Libro con capítulos
│   ├── data/
│   │   └── book_sample.pdf
│   ├── run_example.py
│   └── README.md
└── 3_technical_manual/        # Ejemplo 3: Manual técnico
    ├── data/
    │   └── manual.pdf
    ├── run_example.py
    └── README.md
```

## 🚀 Los Tres Ejemplos

### 1️⃣ Reclutamiento con CVs - Estrategia WITH_SELECTION

**Caso de uso**: Sistema de reclutamiento que busca en CVs y propuestas de trabajo.

**Estrategia**: `WITH_SELECTION` (document_selector_search)

**Documentos**:
- 1 propuesta de trabajo
- 3 CVs de candidatos

**Cómo funciona**:
1. El LLM analiza la pregunta y selecciona qué documentos son relevantes
2. Busca solo en los documentos seleccionados

**Ejemplo de pregunta**: 
> "¿Qué candidato tiene experiencia en Python y desarrollo backend?"

**Ventajas**:
- ✅ Selección inteligente de documentos relevantes
- ✅ Reduce ruido en los resultados
- ✅ Ideal para múltiples documentos de diferentes tipos

[Ver ejemplo completo →](./1_cv_recruitment/)

---

### 2️⃣ Libro con Capítulos - Estrategia WITH_SELECTION_AND_METADATA

**Caso de uso**: Búsqueda precisa en libros con estructura de capítulos y páginas.

**Estrategia**: `WITH_SELECTION_AND_METADATA` (document_selector_metadata_search)

**Documentos**:
- 1 libro con múltiples capítulos

**Cómo funciona**:
1. Selecciona el documento relevante
2. Extrae metadatos de la pregunta (capítulos, páginas, imágenes)
3. Construye filtros precisos de Milvus
4. Busca solo en las secciones específicas

**Ejemplo de pregunta**: 
> "¿Qué dice el capítulo 3 sobre metodologías de desarrollo?"

**Ventajas**:
- ✅ Precisión quirúrgica en búsquedas
- ✅ Entiende lenguaje natural ("capítulo 3", "páginas 10-20")
- ✅ Ideal para documentos estructurados
- ✅ Puede buscar imágenes específicas

[Ver ejemplo completo →](./2_book_chapters/)

---

### 3️⃣ Manual Técnico - Estrategia SIMPLE

**Caso de uso**: Búsqueda directa en documentación técnica.

**Estrategia**: `SIMPLE` (simple_search)

**Documentos**:
- 1 manual técnico o documentación

**Cómo funciona**:
1. Búsqueda vectorial directa en Milvus
2. Sin selección previa de documentos
3. Máxima velocidad y simplicidad

**Ejemplo de pregunta**: 
> "¿Cómo instalar el sistema?"

**Ventajas**:
- ✅ Máxima velocidad
- ✅ Configuración mínima
- ✅ Menor costo (no usa LLM para selección)
- ✅ Ideal para un solo documento o documentación homogénea

[Ver ejemplo completo →](./3_technical_manual/)

---

## 📊 Comparación de Estrategias

| Característica | SIMPLE | WITH_SELECTION | WITH_SELECTION_AND_METADATA |
|----------------|--------|----------------|----------------------------|
| **Velocidad** | ⚡⚡⚡ | ⚡⚡ | ⚡ |
| **Precisión** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Complejidad** | 🟢 Baja | 🟡 Media | 🔴 Alta |
| **Costo LLM** | 💰 Mínimo | 💰💰 Medio | 💰💰💰 Alto |
| **Mejor para** | 1 documento | Múltiples docs | Docs estructurados |
| **Selección de docs** | ❌ | ✅ | ✅ |
| **Filtros de metadata** | ❌ | ❌ | ✅ |
| **Ejemplo** | Manual técnico | CVs | Libro con capítulos |

## 🛠️ Requisitos Previos

Antes de ejecutar cualquier ejemplo, asegúrate de tener:

### 1. Servicios Corriendo

```bash
# Iniciar Milvus y otros servicios
docker-compose up -d
```

### 2. Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto (ver `.env.example`):

```bash
# OpenAI
OPENAI_API_KEY=tu-api-key-aqui

# Milvus
MILVUS_URI=http://localhost:19530
MILVUS_DB_NAME=default
MILVUS_COLLECTION_NAME_DOCUMENTS=documents
MILVUS_COLLECTION_NAME_SUMMARIES=summaries
```

### 3. Dependencias Instaladas

```bash
pip install -r requirements.txt
```

## 🚀 Cómo Ejecutar los Ejemplos

Cada ejemplo tiene dos scripts:
- `upload_documents.py`: Sube e indexa los documentos en Milvus (**ejecutar primero**)
- `run_example.py`: Ejecuta las búsquedas de ejemplo

### Flujo Completo para Cada Ejemplo

```bash
# 1. Ir al directorio del ejemplo
cd examples/1_cv_recruitment  # o 2_book_chapters, o 3_technical_manual

# 2. PRIMERO: Subir e indexar los documentos
python upload_documents.py

# 3. DESPUÉS: Ejecutar las búsquedas
python run_example.py
```

### Ejemplo 1: CVs

```bash
cd examples/1_cv_recruitment

# Paso 1: Indexar documentos
python upload_documents.py

# Paso 2: Ejecutar búsquedas
python run_example.py
```

### Ejemplo 2: Libro

```bash
cd examples/2_book_chapters

# Paso 1: Indexar libro
python upload_documents.py

# Paso 2: Ejecutar búsquedas
python run_example.py
```

### Ejemplo 3: Manual

```bash
cd examples/3_technical_manual

# Paso 1: Indexar manual
python upload_documents.py

# Paso 2: Ejecutar búsquedas
python run_example.py
```

## 📝 Preparar tus Propios Documentos

Cada ejemplo incluye una carpeta `data/` donde debes colocar los documentos:

### Ejemplo 1 - CVs:
```bash
examples/1_cv_recruitment/data/
├── job_proposal.pdf
├── cv_candidate_1.pdf
├── cv_candidate_2.pdf
└── cv_candidate_3.pdf
```

### Ejemplo 2 - Libro:
```bash
examples/2_book_chapters/data/
└── book_sample.pdf
```

### Ejemplo 3 - Manual:
```bash
examples/3_technical_manual/data/
└── manual.pdf
```

## 🎓 Flujo de Trabajo Completo

1. **Preparar documentos**: Coloca los PDFs en las carpetas `data/` de cada ejemplo
2. **Iniciar servicios**: `docker-compose up -d` (desde la raíz del proyecto)
3. **Configurar entorno**: Crea archivo `.env` con tus API keys (ver `.env.example`)
4. **Indexar documentos**: Ejecuta `python upload_documents.py` en cada ejemplo
5. **Ejecutar búsquedas**: Ejecuta `python run_example.py` en cada ejemplo
6. **Analizar resultados**: Observa cómo cada estrategia maneja las consultas

## 🐛 Solución de Problemas

### Error: "No se encontró el archivo"
- Verifica que los PDFs están en la carpeta `data/` correcta
- Los nombres de archivo deben coincidir exactamente

### Error: "Connection refused" o "Milvus not available"
- Asegúrate de que Milvus está corriendo: `docker-compose ps`
- Verifica la URI en el archivo `.env`

### Error: "Collection not found"
- Los documentos deben estar indexados primero
- Verifica los nombres de colecciones en `.env`

### Error: "OpenAI API key not found"
- Configura `OPENAI_API_KEY` en el archivo `.env`
- Verifica que la API key es válida

### Resultados vacíos o irrelevantes
- Verifica que los documentos están correctamente indexados
- Revisa que los embeddings se generaron correctamente
- Ajusta el `search_limit` en la configuración

## 📖 Documentación Adicional

- [Documentación del Pipeline de Búsqueda](../src/search/)
- [Configuración de Milvus](../src/search/milvus/)
- [Estrategias de Búsqueda](../src/search/strategies/)
- [Modelos de Datos](../src/search/models.py)

## 💡 Consejos

1. **Empieza con SIMPLE**: Es la estrategia más fácil de entender y configurar
2. **Usa WITH_SELECTION**: Cuando tengas múltiples documentos de diferentes temas
3. **Usa WITH_SELECTION_AND_METADATA**: Para documentos estructurados (libros, manuales con capítulos)
4. **Experimenta con las queries**: Prueba diferentes formas de hacer la misma pregunta
5. **Revisa los logs**: Usa los logs para entender qué está haciendo el sistema

## 🤝 Contribuir

Si tienes ideas para nuevos ejemplos o mejoras:
1. Crea un nuevo directorio en `examples/`
2. Sigue la estructura de los ejemplos existentes
3. Documenta claramente el caso de uso
4. Incluye un README detallado

## 📧 Soporte

Si tienes problemas ejecutando los ejemplos:
1. Revisa la sección de solución de problemas
2. Verifica los logs del sistema
3. Consulta la documentación del proyecto
4. Abre un issue en GitHub

---

**¡Disfruta explorando las capacidades del sistema RAG!** 🚀

