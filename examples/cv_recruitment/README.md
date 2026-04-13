# CV Recruitment System

## Description

This example uses the **`with_selection`** strategy for recruitment: first the LLM selects relevant documents from their summaries, then semantic search is performed only on those documents.

All three scripts **call the project services over HTTP** (ingestion, retrieval, chatbot); the pipeline is not imported in-process.

## Scenario

A job proposal (`job_proposal.pdf`) and four CVs (`cv_candidate_1.pdf` … `cv_candidate_4.pdf`). Goal: ask questions and compare candidates against the job requirements.

### Documents in `data/`

- `job_proposal.pdf` — job description and requirements
- `cv_candidate_1.pdf` … `cv_candidate_4.pdf` — candidate CVs

### Queries in the scripts

The queries run automatically are defined in the code (`queries` list in each script):

- **`run_retrieval.py`**: asks to evaluate the best candidates (score 0–10) based on the job proposal's technical requirements and rank them. (Returns raw chunks only.)
- **`run_chatbot.py`**: same query but the chatbot answers with a full scored ranking.

You can replace those strings with any other question (e.g.: list Python/backend profiles, who meets the requirements, years of experience, education per candidate, etc.).

## How `with_selection` works

1. **Document selection**: the LLM reads the query and each document's summary to decide which ones are relevant.
2. **Vector search**: the query embedding is applied only to the selected documents, reducing noise.

This is especially useful when the collection contains **many documents**: instead of comparing the query against the entire corpus at once (where unrelated chunks tend to mix in), the search first narrows down to the relevant documents and then performs a **more focused** semantic search on that subset. This reduces noise, improves chunk relevance, and search cost scales better as the number of files grows.

## Files and services

| File | Description |
|------|-------------|
| `upload_documents.py` | Multipart POST to `/ingest` (service port **8015**). |
| `run_retrieval.py` | JSON POST to `/search` with `search_type: "with_selection"` (service port **8007**). |
| `run_chatbot.py` | JSON POST to `/chat` with the same strategy (service port **8018**). |

Base URLs are defined as constants at the top of each script (`localhost` and the ports above). All three scripts use the collection **`cv_recruitment`**. Chunk size, embedding model, and selector parameters (e.g. chooser model) are defined by each service's deployment.

## How to run

### 1. Prepare the PDF files

Make sure the files are present in `examples/cv_recruitment/data/`:

```
cv_recruitment/
├── data/
│   ├── job_proposal.pdf
│   ├── cv_candidate_1.pdf
│   ├── cv_candidate_2.pdf
│   ├── cv_candidate_3.pdf
│   └── cv_candidate_4.pdf
├── upload_documents.py
├── run_retrieval.py
├── run_chatbot.py
└── README.md
```

### 2. Environment and services

From the repo root (`rag-project`):

```bash
docker compose up -d
```

The services load the `.env` at the root (`OPENAI_API_KEY`, Milvus, Mongo, etc.). The example scripts only require that **ingestion**, **retrieval**, and **chatbot** are accessible on the ports above; they do not run the pipeline in-process.

### 3. Upload and index (required first)

```bash
python examples/cv_recruitment/upload_documents.py
```

The script checks that the PDFs exist and sends each one to the ingestion service, leaving the **`cv_recruitment`** collection ready (partitions and chunking are defined by the service configuration).

### 4. Retrieval search

```bash
python examples/cv_recruitment/run_retrieval.py
```

### 5. Chatbot (optional)

```bash
python examples/cv_recruitment/run_chatbot.py
```

## Advantages of `with_selection`

- **Precision**: search is scoped to relevant documents only.
- **Efficiency**: less noise than searching the entire corpus at once.
- **Context**: the selector uses document summaries to understand the query.

## See also

- [Example 1: The Little Prince](../1_principito/) — same HTTP API architecture; search type is `simple` in that case.
