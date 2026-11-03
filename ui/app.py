"""
Interfaz de chat sencilla para el servicio chatbot RAG.

Uso:
    streamlit run ui/app.py

Requisito: servicio chatbot accesible (por defecto http://localhost:8018).
"""

from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

CHATBOT_BASE_URL = os.getenv("CHATBOT_BASE_URL", "http://localhost:8018")
# URL from which the user's browser reaches the chatbot (to open PDFs); usually the same.
CHATBOT_PUBLIC_URL = os.getenv("CHATBOT_PUBLIC_URL", CHATBOT_BASE_URL)
SEARCH_TYPES = ["simple", "with_selection", "with_selection_and_metadata"]
COLLECTIONS = ["principito_imagenes", "cv_recruitment"]
# Deployments de chat de Azure (el primero es el seleccionado por defecto)
LLM_MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-nano", "gpt-5.6-luna", "gpt-6-luna"]

st.set_page_config(page_title="Chat RAG", page_icon="💬")
st.title("💬 Chat RAG")

# --- Configuración (barra lateral) ---
with st.sidebar:
    st.header("Configuración")
    collection_name = st.selectbox("Colección", COLLECTIONS, index=0)
    search_type = st.selectbox("Tipo de búsqueda", SEARCH_TYPES, index=0)
    search_limit = st.slider("Límite de resultados", min_value=5, max_value=25, value=10)
    llm_model = st.selectbox("Modelo LLM", LLM_MODELS, index=0)
    st.caption(f"Servicio: {CHATBOT_BASE_URL}")
    if st.button("Borrar conversación"):
        st.session_state.mensajes = []
        st.rerun()

# --- Historial ---
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]

def mostrar_fuentes(fuentes: list[dict], key: str) -> None:
    """Un botón por documento citado que abre el PDF en la primera página citada."""
    for i, f in enumerate(fuentes):
        paginas = f.get("pages") or []
        etiqueta = f"📄 Ver en el documento · {f['file_name']}"
        if paginas:
            etiqueta += f" (p. {', '.join(paginas)})"
        st.link_button(etiqueta, CHATBOT_PUBLIC_URL.rstrip("/") + f["url"], key=f"{key}_{i}")


for n, m in enumerate(st.session_state.mensajes):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        mostrar_fuentes(m.get("sources", []), key=f"hist_{n}")


def preguntar(query: str) -> tuple[str, list[dict]]:
    payload = {
        "query": query,
        "collection_name": collection_name,
        "search_type": search_type,
        "search_limit": search_limit,
        "llm_model": llm_model,
        "job_id": f"ui_{st.session_state.session_id}.{len(st.session_state.mensajes)}",
    }
    try:
        resp = httpx.post(f"{CHATBOT_BASE_URL}/chat", json=payload, timeout=180.0)
    except httpx.HTTPError as e:
        return f"❌ No se puede contactar con el servicio chatbot: {e}", []
    if resp.status_code != 200:
        return f"❌ HTTP {resp.status_code}: {resp.text}", []
    data = resp.json()
    return data.get("answer", ""), data.get("sources", [])


# --- Entrada del usuario ---
if pregunta := st.chat_input("Escribe tu pregunta..."):
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta, fuentes = preguntar(pregunta)
        st.markdown(respuesta)
        mostrar_fuentes(fuentes, key=f"hist_{len(st.session_state.mensajes)}")

    st.session_state.mensajes.append(
        {"role": "assistant", "content": respuesta, "sources": fuentes}
    )
