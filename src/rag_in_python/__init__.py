"""
RAG in Python - A comprehensive RAG toolkit built with LlamaIndex and FAISS.

This package provides tools and utilities for building Retrieval-Augmented Generation
systems using modern Python practices, LlamaIndex framework, and FAISS vector storage.
"""

__version__ = "0.1.0"
__author__ = "RAG Development Team"
__email__ = "team@example.com"

from .core import RAGSystem
from .indexing import DocumentIndexer
from .retrieval import HybridRetriever
from .generation import ResponseGenerator

__all__ = [
    "RAGSystem",
    "DocumentIndexer", 
    "HybridRetriever",
    "ResponseGenerator",
]