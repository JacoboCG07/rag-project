# RAG Project

Proyecto en desarrollo para Retrieval Augmented Generation (RAG) utilizando Milvus como vector database y un LLM para la generación aumentada de contexto.

Este proyecto aún no está finalizado y se encuentra en evolución activa. Su arquitectura y desarrollo se basan en principios sólidos de ingeniería de software, siguiendo metodologías SOLID, Clean Code y buenas prácticas de diseño para asegurar escalabilidad, mantenibilidad y calidad del código.

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.8+
- Docker y Docker Compose
- API Key de OpenAI

## Configuración Inicial

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# OpenAI (OBLIGATORIO)
OPENAI_API_KEY=tu_api_key_de_openai

# Milvus (opcional, valores por defecto)
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DB_NAME=default

# Colecciones (opcional, valores por defecto)
MILVUS_COLLECTION_NAME_DOCUMENTS=documents
MILVUS_COLLECTION_NAME_SUMMARIES=summaries

# MongoDB (opcional, para logs)
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=admin123
```

### 3. Levantar servicios con Docker

```bash
docker-compose up -d
```

Esto inicia:

- **Milvus** (puerto 19530) - Base de datos vectorial
- **Attu** (puerto 8000) - Interfaz web para Milvus
- **MongoDB** (puerto 27017) - Para logs
- Servicios auxiliares (etcd, MinIO)

Verificar que están corriendo:

```bash
docker ps
```

### 4. Subir documentos (Ejemplo)

```bash
cd examples/1_cv_recruitment
python upload_documents.py
```

Este ejemplo procesa CVs y propuestas de trabajo, indexándolos en la partición `cv_recruitment`.

### 5. Realizar búsquedas (Ejemplo)

```bash
cd examples/1_cv_recruitment
python run_example.py
```

## Ejemplos Disponibles

- **Ejemplo 1**: Sistema de Reclutamiento con CVs (`examples/1_cv_recruitment/`)
- **Ejemplo 2**: Libro con Capítulos (`examples/2_book_chapters/`)
- **Ejemplo 3**: Manual Técnico (`examples/3_technical_manual/`)

Cada ejemplo incluye su propio `README.md` con instrucciones detalladas.

## 🔍 Attu - Interfaz Web para Milvus

**Attu** es una interfaz web de administración para Milvus que permite visualizar y gestionar tus datos vectoriales de forma gráfica.

### Acceso

Una vez que los servicios estén corriendo, puedes acceder a Attu en:

**[http://localhost:8000/#/connect](http://localhost:8000/#/connect)**

### ¿Qué puedes ver en Attu?

- **Colecciones**: Visualiza las colecciones creadas (`documents`, `summaries`, etc.)
- **Particiones**: Explora las particiones dentro de cada colección (ej: `cv_recruitment`)
- **Datos**: Inspecciona los vectores y metadatos indexados
- **Estadísticas**: Revisa el número de entidades, índices y configuración
- **Búsquedas**: Prueba búsquedas vectoriales directamente desde la interfaz

### Conexión en Attu

Al acceder por primera vez, conecta a Milvus usando:

- **Host**: `standalone` (o `localhost` si estás fuera de Docker)
- **Port**: `19530`

## Objetivos principales

- Implementar un pipeline completo de RAG.

- Permitir la indexación, vectorización y recuperación eficiente de documentos mediante Milvus.

- Integrar un modelo LLM para generar respuestas enriquecidas con contexto relevante.

- Garantizar una arquitectura limpia, desacoplada y fácil de extender.

## Tecnologías utilizadas

- Python

- Milvus / pymilvus

- Embeddings (Modelo configurable)

- LLM (Proveedor configurable: OpenAI, local, etc.)

- Docker (opcional para despliegue)

- Pytest (para testeo con enfoque TDD)

### Ejecutar tests

```bash
pytest
```

## Estado del proyecto

🚧 En construcción

La estructura, módulos y funcionalidades pueden cambiar a medida que avanza el desarrollo.

## Planes a futuro

- **Búsquedas del sistema**: Acabar el diseño de las distintas búsquedas disponibles:
  - **Simple**: finalizada.
  - **Con metadatos**: en desarrollo.
  - **Selector de documentos**: en desarrollo.

- **Servicios separados**: Exponer ambos servicios (búsqueda e inserción de documentos) como dos servicios independientes que se puedan desplegar desde `docker-compose`.

- **Sistema de colas con RabbitMQ**:
  - Colas para el servicio de ingesta.
  - Colas para el servicio de retriever.
  - Colas para las distintas llamadas a modelos con OpenAI, de forma que no se sature la API.