# 🎯 Guía de Uso - Estrategias de Búsqueda

Guía práctica para elegir la estrategia de búsqueda correcta según tus necesidades.

---

## 🔍 Estrategias Disponibles

### 1. Búsqueda Simple

**¿Qué hace?** Busca directamente en toda tu colección usando similitud vectorial.

**Código:**
```python
config = SearchPipelineConfig(
    search_type=SearchType.SIMPLE,
    search_limit=10
)

results = pipeline.search(query_embedding=embedding)
```

#### ✅ Ventajas
- ⚡ **Muy rápida** (50-200ms)
- 💰 **Gratis** (sin costos de LLM)
- 🔧 **Simple de implementar**
- 📊 **Comportamiento predecible**

#### ❌ Desventajas
- 🌊 **Busca en TODO** (puede traer resultados irrelevantes)
- 📉 **Menos precisa** en colecciones grandes (>500 docs)
- 🎲 **No entiende contexto** (solo similitud vectorial)

#### 🎯 Úsala cuando:
- ✅ Tienes **< 100 documentos**
- ✅ Necesitas **velocidad** (< 200ms)
- ✅ **Sin presupuesto** para LLM
- ✅ Documentos son **homogéneos** (mismo tema)
- ✅ Búsquedas **simples y directas**

---

### 2. Búsqueda con Selección de Documentos

**¿Qué hace?** Un LLM primero selecciona documentos relevantes, luego busca solo en esos.

**Código:**
```python
from llms.text import OpenAITextModel

text_model = OpenAITextModel(model="gpt-4o-mini")

config = SearchPipelineConfig(
    search_type=SearchType.WITH_SELECTION,
    text_model=text_model,
    search_limit=10
)

results = pipeline.search(
    query_embedding=embedding,
    user_query="Tu consulta aquí"  # ← REQUERIDO
)
```

#### ✅ Ventajas
- 🎯 **Muy precisa** (85-95% de precisión)
- 🧠 **Entiende contexto** (usa LLM)
- 🚀 **Escala bien** (1000+ documentos)
- 🔍 **Filtra ruido** automáticamente
- 📊 **Ideal para consultas complejas**

#### ❌ Desventajas
- ⏱️ **Más lenta** (2-3 segundos)
- 💰 **Cuesta dinero** ($0.001-0.01 por búsqueda)
- 🔌 **Requiere LLM** (OpenAI, Anthropic, etc.)
- 📚 **Necesitas resúmenes** (colección separada)
- 🧩 **Más compleja** de configurar

#### 🎯 Úsala cuando:
- ✅ Tienes **> 100 documentos**
- ✅ Documentos son **variados** (múltiples temas)
- ✅ Necesitas **alta precisión**
- ✅ Tienes **presupuesto** para LLM
- ✅ Consultas son **complejas** o contextuales
- ✅ Puedes esperar **2-3 segundos**

---

## 📊 Comparación Rápida

| ¿Qué necesitas? | Estrategia Recomendada |
|-----------------|------------------------|
| Velocidad extrema (< 200ms) | 🟢 **Simple** |
| Alta precisión | 🔵 **Con Selección** |
| Sin presupuesto | 🟢 **Simple** |
| Colección pequeña (< 100 docs) | 🟢 **Simple** |
| Colección grande (> 100 docs) | 🔵 **Con Selección** |
| Documentos homogéneos | 🟢 **Simple** |
| Documentos variados | 🔵 **Con Selección** |
| Consultas simples | 🟢 **Simple** |
| Consultas complejas | 🔵 **Con Selección** |

---

## 💰 Comparación de Costos

```
Escenario: 1000 búsquedas/día durante 30 días

Simple:
- Costo: $0
- Solo infraestructura

Con Selección:
- Costo LLM: $30-300/mes (según modelo)
- + infraestructura
- Total: ~$50-350/mes
```

---

## ⚡ Comparación de Velocidad

```
Simple:             ████ 50-200ms
Con Selección:      ████████████████████ 2-3 segundos
```

---

## 🎯 Árbol de Decisión

```
¿Cuántos documentos tienes?
│
├─ < 100 docs
│  └─ Usa: SIMPLE ✅
│
└─ > 100 docs
   │
   ├─ ¿Necesitas velocidad extrema?
   │  ├─ Sí → Usa: SIMPLE (considera optimizar)
   │  └─ No → Siguiente pregunta
   │
   └─ ¿Tienes presupuesto para LLM?
      ├─ Sí → Usa: CON SELECCIÓN ✅
      └─ No → Usa: SIMPLE (la precisión será menor)
```

---

## 💡 Recomendaciones

### Para empezar:
1. Comienza con **Simple** (más fácil, sin costo)
2. Si la precisión no es suficiente → Cambia a **Con Selección**

### Para producción:
- **Aplicaciones de usuario**: Con Selección (mejor experiencia)
- **Herramientas internas**: Simple (suficientemente bueno)
- **Casos críticos**: Con Selección (precisión máxima)

---

## 🚀 Ejemplos Completos

Ver: [`examples/search_pipeline_example.py`](../examples/search_pipeline_example.py)

---

**Última actualización**: 2026-01-14

