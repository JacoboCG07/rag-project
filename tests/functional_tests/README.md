# Functional Tests

Esta carpeta contiene tests funcionales que requieren servicios externos reales (APIs, bases de datos, etc.).

## Requisitos

Para ejecutar estos tests necesitas:

- **OpenAI API Key**: Configurada en la variable de entorno `OPENAI_API_KEY`
- Conexión a internet
- Las dependencias necesarias instaladas

## Ejecutar tests funcionales

```bash
# Ejecutar todos los tests funcionales
pytest tests/functional_tests/ -v

# Ejecutar solo tests de embeddings
pytest tests/functional_tests/test_openai_embedder_functional.py -v

# Ejecutar con marcador específico
pytest -m functional -v
```

## Nota

Estos tests pueden incurrir en costos de API. Se recomienda ejecutarlos solo cuando sea necesario verificar la integración con servicios reales.
