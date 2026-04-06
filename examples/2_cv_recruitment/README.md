# Ejemplo 2: Sistema de reclutamiento con CVs

## Descripción

Este ejemplo usa la estrategia **`with_selection`** para reclutamiento: primero se eligen documentos relevantes a partir de resúmenes y luego se hace búsqueda semántica solo sobre ellos.

Igual que el [ejemplo 1 (Principito)](../1_principito/), los tres scripts **llaman por HTTP** a los servicios del proyecto (ingesta, retrieval, chatbot); no importan el pipeline en proceso.

## Escenario

Una propuesta de trabajo (`job_proposal.pdf`) y cuatro CVs (`cv_candidate_1.pdf` … `cv_candidate_4.pdf`). Objetivo: formular preguntas y comparar candidatos frente al puesto.

### Documentos en `data/`

- `job_proposal.pdf` — descripción del puesto y requisitos  
- `cv_candidate_1.pdf` … `cv_candidate_4.pdf` — CVs

### Consultas en los scripts

Las preguntas ejecutadas están en el código (listas `queries` en cada script):

- **`run_retrieval.py`**: una consulta larga que pide evaluar a los mejores candidatos (nota 0–10) según requisitos técnicos de la propuesta y ordenarlos. (Sólo obtendrás los fragmentos)
- **`run_chatbot.py`**: lo mismo pero el chat te responderá con las notas.

Puedes sustituir esas cadenas por otras (por ejemplo: Dime los perfiles Python/backend, quién cumple requisitos, años de experiencia,formación de cada uno, etc).

## Cómo funciona `with_selection`

1. **Selección de documentos**: el LLM mira la pregunta y los resúmenes de los documentos y decide cuáles son relevantes.  
2. **Búsqueda vectorial**: el embedding de la consulta se aplica solo sobre los documentos seleccionados, reduciendo ruido.

Esto es especialmente útil cuando la colección tiene **muchos documentos**: en lugar de comparar la consulta contra todo el corpus a la vez (donde suelen mezclarse fragmentos poco relacionados), primero se acota el conjunto a los que importan para la pregunta y luego la búsqueda semántica es **más concreta** sobre ese subconjunto. Así reduces ruido, mejoras la relevancia de los chunks recuperados y el coste de búsqueda escala mejor al crecer el número de ficheros.

## Archivos y servicios

| Archivo | Descripción |
|---------|-------------|
| `upload_documents.py` | POST multipart a `/ingest` (servicio puerto **8015**). |
| `run_retrieval.py` | POST JSON a `/search` con `search_type: "with_selection"` (servicio puerto **8007**). |
| `run_chatbot.py` | POST JSON a `/chat` con la misma estrategia (servicio puerto **8018**). |

Las URLs base están definidas como constantes al inicio de cada script (`localhost` y los puertos anteriores). La colección usada en los tres es **`cv_recruitment`**. Chunk, embedding y parámetros del selector (p. ej. modelo del chooser) los define el despliegue de cada servicio.

## Cómo ejecutar

### 1. Preparar los PDF

Asegurate de que esten los archivos en `examples/2_cv_recruitment/data/`:

```
2_cv_recruitment/
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

### 2. Entorno y servicios

Desde la raíz del repo (`rag-project`):

```bash
docker-compose up -d
```

Los servicios cargan el `.env` en la raíz (`OPENAI_API_KEY`, Milvus, Mongo, etc.). Los scripts del ejemplo solo necesitan que **ingestion**, **retrieval** y **chatbot** estén accesibles en los puertos indicados; no ejecutan el pipeline en el mismo proceso que el cliente.

### 3. Subir e indexar (obligatorio primero)

```bash
cd examples/2_cv_recruitment
python upload_documents.py
```

El script comprueba que existan los PDF y envía cada uno al servicio de ingesta; queda la colección **`cv_recruitment`** lista (particiones y chunking según la configuración del servicio).

### 4. Búsqueda (retrieval)

```bash
python run_retrieval.py
```

### 5. Chatbot (opcional)

```bash
python run_chatbot.py
```

## Ventajas de `with_selection`

- **Precisión**: se acota la búsqueda a documentos relevantes.  
- **Eficiencia**: menos ruido que buscar en todo el corpus a la vez.  
- **Contexto**: el selector usa resúmenes para entender la pregunta.

## Ver también

- [Ejemplo 1: El Principito](../1_principito/) — misma arquitectura por API HTTP; en ese caso la búsqueda es `simple`.