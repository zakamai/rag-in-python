"""
Document retrieval functionality with hybrid search capabilities.
"""

from typing import List, Optional, Any
import logging

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval system combining vector search with additional filtering.
    """
    
    def __init__(self) -> None:
        """Initialize the hybrid retriever."""
        pass
        
    def retrieve(
        self,
        query: str,
        index: VectorStoreIndex,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Retrieve relevant documents using vector similarity.
        
        Args:
            query: Query string.
            index: Vector store index to search.
            top_k: Number of top documents to retrieve.
            similarity_threshold: Minimum similarity threshold.
            **kwargs: Additional retrieval parameters.
            
        Returns:
            List of retrieved documents.
        """
        # Create retriever from index
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )
        
        # Retrieve nodes
        retrieved_nodes: List[NodeWithScore] = retriever.retrieve(query)
        
        # Filter by similarity threshold and convert to documents
        documents = []
        for node_with_score in retrieved_nodes:
            if node_with_score.score and node_with_score.score >= similarity_threshold:
                # Convert node to document
                doc = Document(
                    text=node_with_score.node.text,
                    metadata={
                        **node_with_score.node.metadata,
                        "score": node_with_score.score,
                        "node_id": node_with_score.node.node_id,
                    }
                )
                documents.append(doc)
                
        logger.info(
            f"Retrieved {len(documents)} documents (out of {len(retrieved_nodes)} candidates) "
            f"above similarity threshold {similarity_threshold}"
        )
        
        return documents
        
    def retrieve_with_metadata_filter(
        self,
        query: str,
        index: VectorStoreIndex,
        metadata_filters: Optional[dict] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Document]:
        """
        Retrieve documents with additional metadata filtering.
        
        Args:
            query: Query string.
            index: Vector store index to search.
            metadata_filters: Dictionary of metadata key-value pairs to filter by.
            top_k: Number of top documents to retrieve.
            similarity_threshold: Minimum similarity threshold.
            
        Returns:
            List of retrieved and filtered documents.
        """
        # First retrieve based on similarity
        documents = self.retrieve(
            query=query,
            index=index,
            top_k=top_k * 2,  # Get more candidates for filtering
            similarity_threshold=similarity_threshold,
        )
        
        # Apply metadata filters if provided
        if metadata_filters:
            filtered_documents = []
            for doc in documents:
                if self._matches_metadata_filters(doc.metadata, metadata_filters):
                    filtered_documents.append(doc)
                    
            documents = filtered_documents[:top_k]  # Limit to top_k after filtering
            
            logger.info(
                f"Applied metadata filters, {len(documents)} documents remaining"
            )
            
        return documents[:top_k]
        
    def _matches_metadata_filters(self, metadata: dict, filters: dict) -> bool:
        """
        Check if document metadata matches the provided filters.
        
        Args:
            metadata: Document metadata.
            filters: Metadata filters to apply.
            
        Returns:
            True if metadata matches all filters, False otherwise.
        """
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True