# Prompt: Extracción de Metadata de Documentos

Tu tarea es analizar la consulta del usuario y los documentos disponibles para extraer metadata relevante que se usará para filtrar la búsqueda.

## Documentos Disponibles

{markdown_documents}

## Consulta del Usuario

```
{user_query}
```

---

## Instrucciones

Analiza la consulta del usuario y determina qué metadata es relevante para cada documento mencionado o relacionado.

### Metadata a Extraer

Para cada documento relevante, extrae:

1. **`pages`** (List[int] o null)
   - Lista de números de página si la consulta los menciona explícitamente
   - Ejemplos: "páginas 1 a 5", "página 10", "primeras 3 páginas"
   - Si no se mencionan: `null`

2. **`chapters`** (List[str] o null)
   - Lista de nombres o números de capítulos si se mencionan
   - Usa los nombres/números exactos si están disponibles en la descripción
   - Ejemplos: "capítulo 1", "capítulo de instalación", "caps 2 y 3"
   - Si no se mencionan: `null`

3. **`search_image`** (boolean)
   - `true` si la consulta implica buscar o necesitar imágenes/diagramas/gráficos
   - `false` si solo busca texto o no menciona imágenes
   - Ejemplos que son `true`: "muéstrame diagramas", "imágenes de", "gráficos sobre"

4. **`num_image`** (List[int] o null)
   - Lista de números de imagen si se mencionan específicamente
   - Ejemplos: "imagen 5", "figuras 2 y 3"
   - Si no se mencionan números específicos: `null`

5. **`type_file`** (string)
   - Tipo de archivo del documento (obtenido de la información del documento)
   - Valores posibles: "PDF", "TXT", "DOCX", etc.
   - Usa el valor exacto del campo `type_file` del documento

---

## Reglas Importantes

1. **Solo incluye documentos relevantes**: Si la consulta menciona documentos específicos o temas relacionados, solo incluye esos documentos en la respuesta.

2. **Valores null**: Si una metadata no se menciona en la consulta, usa `null` (no omitas el campo).

3. **Rangos de páginas**: Si se menciona un rango (ej: "páginas 1-5"), expande a lista: `[1, 2, 3, 4, 5]`.

4. **Inferencia mínima**: No inventes metadata que no esté explícita o implícita en la consulta.

5. **Todos los documentos**: Si la consulta no especifica documentos particulares pero es general, incluye metadata para todos los documentos disponibles (típicamente con valores `null` excepto `type_file`).

---

## Formato de Respuesta

Responde **SOLO** con un objeto JSON válido. **NO incluyas** explicaciones, comentarios ni texto adicional.

```json
{
  "file_id1": {
    "pages": [1, 2, 3] o null,
    "chapters": ["cap1", "cap2"] o null,
    "search_image": true o false,
    "num_image": [1, 2] o null,
    "type_file": "PDF"
  },
  "file_id2": {
    "pages": null,
    "chapters": null,
    "search_image": false,
    "num_image": null,
    "type_file": "TXT"
  }
}
```

---

## Ejemplos

### Ejemplo 1: Páginas específicas

**Consulta**: "Buscar en las páginas 1 a 5 del manual de instalación"

**Documentos**: 
- doc_001: manual_instalacion.pdf (50 páginas)
- doc_002: guia_usuario.pdf (120 páginas)

**Respuesta**:
```json
{
  "doc_001": {
    "pages": [1, 2, 3, 4, 5],
    "chapters": null,
    "search_image": false,
    "num_image": null,
    "type_file": "PDF"
  }
}
```

---

### Ejemplo 2: Búsqueda de imágenes

**Consulta**: "Mostrar diagramas sobre arquitectura del sistema"

**Documentos**:
- doc_003: api_reference.pdf (200 páginas)
- doc_004: architecture_diagrams.pdf (30 páginas)

**Respuesta**:
```json
{
  "doc_004": {
    "pages": null,
    "chapters": null,
    "search_image": true,
    "num_image": null,
    "type_file": "PDF"
  }
}
```

---

### Ejemplo 3: Capítulos y tipo de archivo

**Consulta**: "Información del capítulo 2 en PDFs"

**Documentos**:
- doc_001: manual.pdf (50 páginas, 5 capítulos)
- doc_002: guia.txt (archivo de texto)

**Respuesta**:
```json
{
  "doc_001": {
    "pages": null,
    "chapters": ["2"],
    "search_image": false,
    "num_image": null,
    "type_file": "PDF"
  }
}
```

---

### Ejemplo 4: Consulta general (sin filtros específicos)

**Consulta**: "¿Cómo instalar el sistema?"

**Documentos**:
- doc_001: manual_instalacion.pdf
- doc_002: guia_usuario.pdf
- doc_003: api_reference.pdf

**Respuesta**:
```json
{
  "doc_001": {
    "pages": null,
    "chapters": null,
    "search_image": false,
    "num_image": null,
    "type_file": "PDF"
  }
}
```
*(Solo el documento relevante para instalación)*

---

## Ahora es tu turno

Analiza la consulta del usuario y los documentos proporcionados al inicio, y genera la respuesta JSON con la metadata extraída.

**Respuesta**:

