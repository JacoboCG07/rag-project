# Prueba 1: El Principito con imágenes

Ejemplo de RAG sobre **El Principito** de Antoine de Saint-Exupéry. Usa un PDF que incluye texto e imágenes, y demuestra todo el flujo: ingestión, retrieval y chatbot.

## De qué trata

Se indexa `principito.pdf` en Milvus con **extracción de imágenes** activada (`extract_images=True`). El texto se divide en chunks y las imágenes se describen con un modelo de visión. Todo queda en la colección `principito_imagenes`, que tiene dos particiones: `documents` (chunks de texto + descriptores de imágenes) y `summaries` (resúmenes de cada documento).

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `upload_documents.py` | Ingesta: procesa `principito.pdf`, extrae texto e imágenes, genera embeddings, resúmenes y descripciones, y los indexa en la colección `principito_imagenes`. |
| `run_retrieval.py` | Búsqueda: consultas vectoriales sobre la colección. Muestra los chunks más relevantes para cada pregunta sin generar respuestas. |
| `run_chatbot.py` | Chatbot RAG: retrieval + LLM. Recupera contexto y genera respuestas naturales usando el texto e imágenes indexados. |

## Orden de ejecución

1. **Subir documentos** (primero, solo una vez):
   ```bash
   python examples/prueba_1/upload_documents.py
   ```

2. **Búsqueda** (solo chunks):
   ```bash
   python examples/prueba_1/run_retrieval.py
   ```

3. **Chatbot** (retrieval + generación):
   ```bash
   python examples/prueba_1/run_chatbot.py
   ```

## Requisitos

- Milvus en marcha (`docker-compose up -d`)
- `.env` con `OPENAI_API_KEY` y configuración de Milvus
- PDF en `data/principito.pdf`

## Visualizar la base vectorial (Attu)

Una vez tengas levantado el servicio con `docker-compose up -d` puedes usar **Attu**, la interfaz web para explorar Milvus y ver las colecciones (por ejemplo `principito_imagenes`), particiones y datos indexados.

**URL:** http://localhost:8000

En Attu, conecta a Milvus con `localhost:19530` (configuración por defecto del `docker-compose`).
