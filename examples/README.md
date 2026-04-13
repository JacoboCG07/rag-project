# Tutorial

Carpeta guiada con su propio `README` con pasos, requisitos y detalle.

| Carpeta | Qué es |
|---------|--------|
| [cv_recruitment](cv_recruitment/) | Reclutamiento con propuesta de trabajo y varios CVs: estrategia **`WITH_SELECTION`** (selección de documentos + búsqueda vectorial). Los scripts llaman por **HTTP** a los servicios de ingesta, retrieval y chatbot. |

Para configuración del `.env`, orden de ejecución y detalles, sigue el README de la carpeta. En local usa **`docker compose`** (subcomando) y un **venv** de Python antes de `pip install -r requirements.txt` (véase el [README principal](../README.md)).
