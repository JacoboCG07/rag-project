# RAG Project

**Retrieval Augmented Generation (RAG)** pipeline: document ingestion (PDF/TXT), vector search in **Milvus**, and responses with **OpenAI** (embeddings + LLM). The logic lives in `src/`; three **FastAPI** services (ingestion, retrieval, chatbot) expose it and can be run with Docker.

The architecture follows **SOLID** principles and a **modular** design that makes it easy to swap in new **embedding** and **LLM** models, evolve the **retrieval** and **ingestion** strategies for different use cases, and add support for **new document formats** without rewriting the full pipeline.

The project is still evolving; the API may change.

---

## Quick start (minimum to try it out)

Goal: get Milvus + the three services running and execute the **CV Recruitment** example against the HTTP API.

### You need

- **Docker** and **Docker Compose**
- **Python 3.10+** (3.11 recommended) on your machine — only needed to run the example scripts (they use `httpx`).
- An **OpenAI API key**.

### Steps

1. **Clone and install dependencies** (to run the scripts in `examples/`):

   ```bash
   git clone https://github.com/JacoboCG07/rag-project.git
   cd rag-project
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   On **Ubuntu, Debian, and similar systems** (including many **WSL** setups), the system Python is often marked as *externally managed* ([PEP 668](https://peps.python.org/pep-0668/)): installing packages with `pip` directly on that interpreter may raise an error or go against the OS package policy. That is why these steps create a **virtual environment** and only run `pip install` inside it. If you use an IDE that already selects the `.venv` interpreter, you may not need to activate it manually in every terminal.

2. **Configure `.env` at the repo root** with at least:

   ```env
   OPENAI_API_KEY=sk-...

   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   MILVUS_DB_NAME=default

   MILVUS_PARTITION_NAME_DOCUMENTS=documents
   MILVUS_PARTITION_NAME_SUMMARIES=summaries
   ```

   The ingestion, retrieval, and chatbot containers **override** `MILVUS_HOST` to `standalone` to talk to Milvus inside the Docker network; the block above also works if you run code on the host against Milvus exposed on `localhost:19530`.

3. **Start the stack** (the first time it builds images; this may take several minutes):

   ```bash
   docker compose up -d
   ```

   Check status: `docker compose ps`. Wait for **standalone** (Milvus) to be healthy before testing the RAG services.

4. **Check service health** (optional but useful):

   ```bash
   curl -s http://localhost:8015/health
   curl -s http://localhost:8007/health
   curl -s http://localhost:8018/health
   ```

5. **Run the guided example** (from the repo root, with the **venv activated**, in this order):

   ```bash
   python examples/cv_recruitment/upload_documents.py
   python examples/cv_recruitment/run_retrieval.py
   python examples/cv_recruitment/run_chatbot.py
   ```

   Port details, variables, and the example PDFs: [`examples/cv_recruitment/README.md`](examples/cv_recruitment/README.md).

If something fails, check `docker compose logs` for the failing service. To inspect collections in the browser: [Attu](#attu-milvus-in-the-browser) at `http://localhost:8000`.

---

## Contents

- [Quick start (minimum to try it out)](#quick-start-minimum-to-try-it-out)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup (detailed)](#setup-detailed)
- [Environment variables](#environment-variables)
- [Docker services](#docker-services)
- [HTTP APIs](#http-apis)
- [Typical workflow](#typical-workflow)
- [Examples](#examples)
- [Attu (Milvus in the browser)](#attu-milvus-in-the-browser)
- [Tests](#tests)
- [Technologies](#technologies)
- [Roadmap](#roadmap)

---

## Architecture

- **Ingestion**: receives multipart files, chunks, embeds, and inserts into Milvus.
- **Retrieval**: semantic search over a collection (without generating a long LLM response).
- **Chatbot**: retrieval + response generation with the specified chat model.

Milvus **partition** names (documents / summaries) come from the `.env` (`MILVUS_PARTITION_NAME_*`). The **collection name** and search parameters are sent in the ingestion request, in `POST /search`, and in `POST /chat` (the chatbot uses the same body as `/search`, plus `llm_model`).

---

## Requirements

- Python **3.10+** (3.11 recommended); for local development, install dependencies inside a **venv**.
- Docker with the Compose plugin
- **OpenAI** API key (the design allows integrating other providers with minimal changes)

---

## Setup (detailed)

### 1. Clone and install dependencies

```bash
git clone https://github.com/JacoboCG07/rag-project.git
cd rag-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

Create or edit the `.env` at the repo root. Minimum for the services:

```env
OPENAI_API_KEY=sk-...

MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DB_NAME=default

MILVUS_PARTITION_NAME_DOCUMENTS=documents
MILVUS_PARTITION_NAME_SUMMARIES=summaries
```

Optional: MongoDB for logs (`MONGO_*`), Milvus token (`MILVUS_TOKEN`), URI (`MILVUS_URI`). The `docker-compose.yml` file defines the stack; when started with `docker compose up`, the RAG services set `MILVUS_HOST=standalone` to point to the Milvus container.

### 3. Start the stack

```bash
docker compose up -d
```

Run `docker compose ps` to see container status.

### 4. Follow an example

The most documented flow is **`examples/cv_recruitment/`** (HTTP calls to ingestion, retrieval, and chatbot). Use the same `collection_name` when ingesting as in search and chat.

---

## Environment variables

| Area | Variables | Notes |
|------|-----------|-------|
| OpenAI | `OPENAI_API_KEY` | Required for embeddings and chat. |
| Milvus | `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_DB_NAME`, `MILVUS_TOKEN`, `MILVUS_URI` | In Compose, towards containers: host `standalone`, port `19530`. |
| Partitions | `MILVUS_PARTITION_NAME_DOCUMENTS`, `MILVUS_PARTITION_NAME_SUMMARIES` | Used by ingestion and retrieval. |
| Ingestion (server) | `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP`, `INGEST_EMBEDDING_MODEL`, `INGEST_MAX_IMAGES_PER_DOCUMENT` | Configure the ingestion operator; not sent per HTTP request. |

---

## Docker services

| Service | Port | Description |
|---------|------|-------------|
| Milvus standalone | `19530` | Vector database. |
| Attu | `8000` | Web UI for Milvus. |
| **ingestion** | `8015` | `POST /ingest`, `GET /health`. |
| **retrieval** | `8007` | `POST /search`, `GET /health`. |
| **chatbot** | `8018` | `POST /chat`, `GET /health`. |
| MongoDB | `27017` | Logs (optional depending on configuration). |
| etcd, MinIO | internal | Milvus dependencies. |

Rebuild only the RAG services:

```bash
docker compose build ingestion retrieval chatbot
docker compose up -d ingestion retrieval chatbot
```

---

### Ingestion — `http://localhost:8015`

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Service health check. |
| POST | `/ingest` | Multipart: file + metadata. |

**`multipart` form**

| Field | Required | Description |
|-------|----------|-------------|
| `file` | Yes | PDF or TXT. |
| `collection_name` | Yes | Target Milvus collection. |
| `extract_images` | Yes | `true` / `false` (also `1`, `yes`). |
| `job_id` | No | Trace identifier for logs. |

**Example**

```bash
curl -s -X POST http://localhost:8015/ingest \
  -F "file=@./document.pdf" \
  -F "collection_name=my_collection" \
  -F "extract_images=false" \
  -F "job_id=job-001"
```

---

### Retrieval — `http://localhost:8007`

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Service health check. |
| POST | `/search` | JSON: semantic search. |

**JSON body**

| Field | Required | Default / range |
|-------|----------|-----------------|
| `query` | Yes | — |
| `collection_name` | Yes | — |
| `search_type` | No | `simple` |
| `search_limit` | No | `10` (between **5 and 25**) |
| `job_id` | No | — |

`search_type` values: `simple`, `with_selection`, `with_selection_and_metadata`.

The query embedding model is the one configured in the pipeline (default in code / environment); this endpoint does **not** expose `embedding_model` or `filter_expr`.

**Example**

```bash
curl -s -X POST http://localhost:8007/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Your question\",\"collection_name\":\"my_collection\",\"search_type\":\"simple\",\"search_limit\":10}"
```

---

### Chatbot — `http://localhost:8018`

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Service health check. |
| POST | `/chat` | JSON: same fields as `/search` plus `llm_model`. |

**JSON body**

Same as **`POST /search`**: `query`, `collection_name`, `search_type` (optional, default `simple`), `search_limit` (optional, 5–25, default `10`), `job_id` (optional). Additionally:

| Field | Required | Default |
|-------|----------|---------|
| `llm_model` | No | `gpt-4o-mini` |

LLM temperature and `max_tokens` use code defaults, not request parameters.

**Example**

```bash
curl -s -X POST http://localhost:8018/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Your question\",\"collection_name\":\"my_collection\",\"search_type\":\"simple\",\"search_limit\":10,\"llm_model\":\"gpt-4o-mini\",\"job_id\":\"my-trace\"}"
```

---

## Typical workflow

1. **Ingest** documents into a specific `collection_name` (`POST /ingest`).
2. **Search** for relevant chunks (`POST /search`) or **chat** with retrieved context (`POST /chat`).
3. Optional: inspect data in **Attu** (`http://localhost:8000`, connect to Milvus: `localhost:19530`).

---

## Examples

Each folder has its own `README.md`. Summary in [`examples/README.md`](examples/README.md).

| Folder | Approach |
|--------|----------|
| [`examples/cv_recruitment/`](examples/cv_recruitment/) | CV recruitment; **with_selection** strategy; HTTP calls to ingestion, retrieval, and chatbot. |

---

## Attu (Milvus in the browser)

- URL: [http://localhost:8000](http://localhost:8000) (after `docker compose up`).
- From your machine, on the Attu connection screen: host **`localhost`**, port **`19530`**. (Inside Docker, other services use the hostname `standalone`.)

Use it to inspect collections, partitions, entities, and indexes.

---

## Tests

With the **venv** activated (same dependencies as `requirements.txt`):

```bash
pytest
```

Tests live under `tests/` (including API smoke tests with mocks).

---

## Technologies

- Python, FastAPI, Uvicorn, Pydantic
- Milvus (pymilvus), PyMuPDF, OpenAI SDK
- Docker Compose, Pytest, HTTPX

---

## Roadmap

- Consolidate search strategies (`with_selection`, metadata) and keep documentation aligned with the code.
- Work queues (e.g. RabbitMQ) for ingestion and model calls under load.