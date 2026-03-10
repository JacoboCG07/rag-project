"""
Chatbot module - RAG integration (retrieval + generation)
"""

from .config import ChatbotConfig
from .rag_chatbot import RAGChatbot

__all__ = ["ChatbotConfig", "RAGChatbot"]
