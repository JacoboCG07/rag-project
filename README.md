# RAG Project

Pipeline de **Retrieval Augmented Generation (RAG)**: ingestión de documentos (PDF/TXT), búsqueda vectorial en **Milvus** y respuestas con **OpenAI** (embeddings + LLM). La lógica vive en `src/`; la exposición son tres servicios **FastAPI** (ingesta, retrieval, chatbot) que puedes levantar con Docker.

La arquitectura sigue principios **SOLID** y un diseño **modular** que facilita adaptar e incorporar nuevos modelos de **embeddings** y **LLM**, evolucionar la estrategia de **recuperación** y de **ingesta** según el caso de uso, y ampliar el soporte a **nuevos formatos de documento** sin reescribir el pipeline completo.

El proyecto sigue evolucionando; la API puede cambiar.

---

## Inicio rápido (lo mínimo para probarlo)

Objetivo: tener Milvus + los tres servicios en marcha y ejecutar el ejemplo **El Principito** contra la API HTTP.

### Necesitas

- **Docker** y **Docker Compose** (plugin `docker compose`, no solo el binario antiguo).
- **Python 3.10+** (recomendado 3.11) en tu máquina, solo para lanzar los scripts de ejemplo (usan `httpx`).
- Una **API key de OpenAI**.

### Pasos

1. **Clonar e instalar dependencias** (para poder ejecutar los scripts de `examples/`):

   ```bash
   git clone <url-del-repo>
   cd rag-project
   pip install -r requirements.txt
   ```

2. **Crear `.env` en la raíz del repo** con al menos:

   ```env
   OPENAI_API_KEY=sk-...

   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   MILVUS_DB_NAME=default

   MILVUS_PARTITION_NAME_DOCUMENTS=documents
   MILVUS_PARTITION_NAME_SUMMARIES=summaries
   ```

   Los contenedores de ingesta, retrieval y chatbot **sobrescriben** `MILVUS_HOST` a `standalone` para hablar con Milvus dentro de la red Docker; el bloque anterior sirve también si ejecutas código en el host contra Milvus expuesto en `localhost:19530`.

3. **Levantar el stack** (la primera vez construye imágenes; puede tardar varios minutos):

   ```bash
   docker compose up -d
   ```

   Comprueba estado: `docker compose ps`. Espera a que **standalone** (Milvus) esté healthy antes de probar los servicios RAG.

4. **Probar salud de los servicios** (opcional pero útil):

   ```bash
   curl -s http://localhost:8015/health
   curl -s http://localhost:8007/health
   curl -s http://localhost:8018/health
   ```

5. **Ejecutar el ejemplo guiado** (desde la raíz del repo, en este orden):

   ```bash
   python examples/1_principito/upload_documents.py
   python examples/1_principito/run_retrieval.py
   python examples/1_principito/run_chatbot.py
   ```

   Detalle de puertos, variables y el PDF de ejemplo: [`examples/1_principito/README.md`](examples/1_principito/README.md).

Si algo falla, revisa `docker compose logs` del servicio que falle. Para inspeccionar colecciones en el navegador: [Attu](#attu-milvus-en-el-navegador) en `http://localhost:8000`.

---

## Contenido

- [Inicio rápido (lo mínimo para probarlo)](#inicio-rápido-lo-mínimo-para-probarlo)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Puesta en marcha (detalle)](#puesta-en-marcha-detalle)
- [Variables de entorno](#variables-de-entorno)
- [Servicios Docker](#servicios-docker)
- [APIs HTTP](#apis-http)
- [Flujo de trabajo típico](#flujo-de-trabajo-típico)
- [Ejemplos](#ejemplos)
- [Attu (Milvus en el navegador)](#attu-milvus-en-el-navegador)
- [Tests](#tests)
- [Tecnologías](#tecnologías)
- [Roadmap](#roadmap)

---

## Arquitectura

- **Ingestion**: recibe ficheros multipart, trocea, embebe e inserta en Milvus.
- **Retrieval**: búsqueda semántica sobre una colección (sin generar respuesta larga con el LLM).
- **Chatbot**: retrieval + generación de respuesta con el modelo de chat indicado.

Los nombres de **partición** Milvus (documentos / resúmenes) salen del `.env` (`MILVUS_PARTITION_NAME_*`). El **nombre de colección** y parámetros de búsqueda los envías en ingesta, en `POST /search` y en `POST /chat` (el chatbot usa el mismo cuerpo que `/search`, más `llm_model`).

---

## Requisitos

- Python **3.10+** (recomendado 3.11)
- **Docker** y **Docker Compose**
- API key de **OpenAI** (el diseño permite integrar otros proveedores con cambios acotados)

---

## Puesta en marcha (detalle)

### 1. Clonar e instalar dependencias

```bash
git clone <url-del-repo>
cd rag-project
pip install -r requirements.txt
```

### 2. Configurar `.env`

Crea un `.env` en la raíz. Mínimo para los servicios:

```env
OPENAI_API_KEY=sk-...

MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DB_NAME=default

MILVUS_PARTITION_NAME_DOCUMENTS=documents
MILVUS_PARTITION_NAME_SUMMARIES=summaries
```

Opcional: MongoDB para logs (`MONGO_*`), token Milvus (`MILVUS_TOKEN`), URI (`MILVUS_URI`). En `docker-compose.yml`, los servicios RAG fijan `MILVUS_HOST=standalone` hacia el contenedor Milvus.

### 3. Levantar el stack

```bash
docker compose up -d
```

`docker compose ps` para ver el estado de los contenedores.

### 4. Seguir un ejemplo

El flujo más documentado es **`examples/1_principito/`** (llamadas HTTP a ingesta, retrieval y chatbot). Usa la misma `collection_name` al ingerir que en búsqueda y chat.

---

## Variables de entorno

| Área | Variables | Notas |
|------|-----------|--------|
| OpenAI | `OPENAI_API_KEY` | Obligatoria para embeddings y chat. |
| Milvus | `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_DB_NAME`, `MILVUS_TOKEN`, `MILVUS_URI` | En Compose hacia los contenedores: host `standalone`, puerto `19530`. |
| Particiones | `MILVUS_PARTITION_NAME_DOCUMENTS`, `MILVUS_PARTITION_NAME_SUMMARIES` | Usadas por ingesta y búsqueda. |
| Ingesta (servidor) | `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP`, `INGEST_EMBEDDING_MODEL`, `INGEST_MAX_IMAGES_PER_DOCUMENT` | Configuran el operador de ingesta; no van en cada petición HTTP. |

---

## Servicios Docker

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| Milvus standalone | `19530` | Base vectorial. |
| Attu | `8000` | UI web para Milvus. |
| **ingestion** | `8015` | `POST /ingest`, `GET /health`. |
| **retrieval** | `8007` | `POST /search`, `GET /health`. |
| **chatbot** | `8018` | `POST /chat`, `GET /health`. |
| MongoDB | `27017` | Logs (opcional según configuración). |
| etcd, MinIO | internos | Dependencias de Milvus. |

Reconstruir solo los servicios RAG:

```bash
docker compose build ingestion retrieval chatbot
docker compose up -d ingestion retrieval chatbot
```

---

### Ingesta — `http://localhost:8015`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Comprobación de servicio. |
| POST | `/ingest` | Multipart: fichero + metadatos. |

**Formulario `multipart`**

| Campo | Obligatorio | Descripción |
|--------|-------------|-------------|
| `file` | Sí | PDF o TXT. |
| `collection_name` | Sí | Colección Milvus destino. |
| `extract_images` | Sí | `true` / `false` (también variantes como `1`, `yes`). |
| `job_id` | No | Trazas en logs. |

**Ejemplo**

```bash
curl -s -X POST http://localhost:8015/ingest \
  -F "file=@./documento.pdf" \
  -F "collection_name=mi_coleccion" \
  -F "extract_images=false" \
  -F "job_id=job-001"
```

---

### Retrieval — `http://localhost:8007`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Comprobación de servicio. |
| POST | `/search` | JSON: búsqueda semántica. |

**Cuerpo JSON**

| Campo | Obligatorio | Default / rango |
|--------|-------------|-----------------|
| `query` | Sí | — |
| `collection_name` | Sí | — |
| `search_type` | No | `simple` |
| `search_limit` | No | `10` (entre **5 y 25**) |
| `job_id` | No | — |

Valores de `search_type`: `simple`, `with_selection`, `with_selection_and_metadata`.

El modelo de embeddings de la consulta es el configurado en el pipeline (por defecto en código / entorno); este endpoint **no** expone `embedding_model` ni `filter_expr`.

**Ejemplo**

```bash
curl -s -X POST http://localhost:8007/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Tu pregunta\",\"collection_name\":\"mi_coleccion\",\"search_type\":\"simple\",\"search_limit\":10}"
```

---

### Chatbot — `http://localhost:8018`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Comprobación de servicio. |
| POST | `/chat` | JSON: mismos campos que `/search` más `llm_model`. |

**Cuerpo JSON**

Igual que **`POST /search`**: `query`, `collection_name`, `search_type` (opcional, default `simple`), `search_limit` (opcional, 5–25, default `10`), `job_id` (opcional). Además:

| Campo | Obligatorio | Default |
|--------|-------------|---------|
| `llm_model` | No | `gpt-4o-mini` |

Temperatura y `max_tokens` del LLM siguen valores por defecto en código, no la petición.

**Ejemplo**

```bash
curl -s -X POST http://localhost:8018/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Tu pregunta\",\"collection_name\":\"mi_coleccion\",\"search_type\":\"simple\",\"search_limit\":10,\"llm_model\":\"gpt-4o-mini\",\"job_id\":\"mi-traza\"}"
```

---

## Flujo de trabajo típico

1. **Ingerir** documentos en una `collection_name` concreta (`POST /ingest`).
2. **Buscar** fragmentos relevantes (`POST /search`) o **chatear** con contexto recuperado (`POST /chat`).
3. Opcional: revisar datos en **Attu** (`http://localhost:8000`, conexión a Milvus: `localhost:19530` desde el host).

---

## Ejemplos

Cada carpeta tiene su propio `README.md`. Resumen en [`examples/README.md`](examples/README.md).

| Carpeta | Enfoque |
|---------|---------|
| [`examples/1_principito/`](examples/1_principito/) | Flujo completo vía **APIs HTTP** (recomendado para empezar). |
| [`examples/2_cv_recruitment/`](examples/2_cv_recruitment/) | Reclutamiento con CVs; estrategia **with_selection**; scripts con **pipeline en proceso** (no solo API HTTP). |

---

## Attu (Milvus en el navegador)

- URL: [http://localhost:8000](http://localhost:8000) (tras `docker compose up`).
- Desde tu PC, en la pantalla de conexión de Attu: host **`localhost`**, puerto **`19530`**. (Dentro de Docker, otros servicios usan el hostname `standalone`.)

Sirve para inspeccionar colecciones, particiones, entidades e índices.

---

## Tests

```bash
pytest
```

Hay pruebas bajo `tests/` (incluido smoke de APIs con mocks).

---

## Tecnologías

- Python, FastAPI, Uvicorn, Pydantic  
- Milvus (pymilvus), PyMuPDF, OpenAI SDK  
- Docker Compose, Pytest, HTTPX  

---

## Roadmap

- Afianzar estrategias de búsqueda (`with_selection`, metadatos) y documentación alineada al código.
- Colas de trabajo (p. ej. RabbitMQ) para ingesta y llamadas a modelos bajo carga.
