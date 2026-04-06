# Prueba 1: El Principito con imágenes

Ejemplo de RAG sobre **El Principito**. Los tres scripts llaman por HTTP a los servicios del proyecto (ingesta, retrieval, chatbot).

## De qué trata

Se sube `data/principito.pdf` a la colección **`principito_imagenes`**

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `upload_documents.py` | POST multipart a `/ingest` (servicio puerto **8015**). (Sube el documento)|
| `run_retrieval.py` | POST JSON a `/search` (servicio puerto **8007**). (Realiza una búsqueda)|
| `run_chatbot.py` | POST JSON a `/chat` (servicio puerto **8018**). (Realiza una búsqueda y contesta con la información extraída)|

Las URLs base están definidas como constantes al inicio de cada script (`localhost` y los puertos anteriores).

## Variables de entorno del proyecto

Los servicios cargan el `.env` en la raíz del repo (`rag-project/.env`). Variables previstas:

| Variable | Uso |
|----------|-----|
| `OPENAI_API_KEY` | API de OpenAI (ingesta/chatbot según configuración). |
| `MILVUS_HOST` | Host de Milvus. |
| `MILVUS_PORT` | Puerto de Milvus. |
| `MILVUS_DB_NAME` | Base de datos en Milvus. |
| `MILVUS_PARTITION_NAME_DOCUMENTS` | Partición de documentos. |
| `MILVUS_PARTITION_NAME_SUMMARIES` | Partición de resúmenes. |
| `MONGO_HOST` | Host de MongoDB (logs / metadatos según el servicio). |
| `MONGO_PORT` | Puerto de MongoDB. |
| `MONGO_ROOT_USERNAME` | Usuario root de MongoDB. |
| `MONGO_ROOT_PASSWORD` | Contraseña root de MongoDB. |
| `MONGO_DB_NAME` | Nombre de la base de datos. |
| `MONGO_LOGS_COLLECTION` | Colección de logs. |
| `MONGO_LOG_LEVEL` | Nivel de log para Mongo u operador de logs. |

Los scripts del ejemplo solo necesitan que los tres servicios estén levantados y el `.env` correcto para los contenedores o procesos del backend.

## Orden de ejecución

1. Levantar Milvus y los servicios (o al menos los que vayas a usar):

   ```bash
   docker-compose up -d
   ```

2. **Subir documentos**:

   ```bash
   python examples/1_principito/upload_documents.py
   ```

3. **Búsqueda** (solo chunks):

   ```bash
   python examples/1_principito/run_retrieval.py
   ```

4. **Chatbot**:

   ```bash
   python examples/1_principito/run_chatbot.py
   ```

## Requisitos

- Servicios FastAPI en marcha (o `docker compose` con `ingestion`, `retrieval`, `chatbot`).
- `.env` en la raíz del repo con los valores anteriores (p. ej. `OPENAI_API_KEY` y Milvus/Mongo) para que los servicios arranquen bien.
- PDF de ejemplo en `data/principito.pdf`.

## Attu
 (Interfaz web para ver la base de datos y las distintas colecciones de milvus)
**URL:** http://localhost:8000

Conexión a Milvus: `localhost:19530` desde el host.
