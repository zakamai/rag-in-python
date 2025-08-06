"""
Document indexing functionality using LlamaIndex and FAISS.
"""

from typing import List, Optional
from pathlib import Path
import logging

import faiss
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Handles document loading and indexing with FAISS vector store.
    """
    
    def __init__(
        self,
        embedding_model: BaseEmbedding,
        vector_store_path: Optional[Path] = None,
        dimension: int = 1536,  # OpenAI embedding dimension
    ) -> None:
        """
        Initialize the document indexer.
        
        Args:
            embedding_model: Embedding model to use for document encoding.
            vector_store_path: Path to persist/load FAISS vector store.
            dimension: Vector dimension for FAISS index.
        """
        self.embedding_model = embedding_model
        self.vector_store_path = vector_store_path
        self.dimension = dimension
        
    def load_documents(self, file_paths: List[Path]) -> List[Document]:
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of file paths to load.
            
        Returns:
            List of loaded documents.
        """
        documents = []
        
        for file_path in file_paths:
            if file_path.is_file():
                try:
                    # Use SimpleDirectoryReader for single file
                    reader = SimpleDirectoryReader(
                        input_files=[str(file_path)]
                    )
                    file_docs = reader.load_data()
                    documents.extend(file_docs)
                    logger.info(f"Loaded {len(file_docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            elif file_path.is_dir():
                try:
                    # Use SimpleDirectoryReader for directory
                    reader = SimpleDirectoryReader(str(file_path))
                    dir_docs = reader.load_data()
                    documents.extend(dir_docs)
                    logger.info(f"Loaded {len(dir_docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading directory {file_path}: {e}")
            else:
                logger.warning(f"Path does not exist: {file_path}")
                
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
        
    def create_vector_store(self) -> FaissVectorStore:
        """
        Create a new FAISS vector store.
        
        Returns:
            Initialized FAISS vector store.
        """
        # Create FAISS index
        faiss_index = faiss.IndexFlatIP(self.dimension)
        
        # Create vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        return vector_store
        
    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> VectorStoreIndex:
        """
        Index documents into FAISS vector store.
        
        Args:
            documents: List of documents to index.
            batch_size: Number of documents to process in each batch.
            
        Returns:
            Vector store index.
        """
        # Create vector store
        vector_store = self.create_vector_store()
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index with batching
        logger.info(f"Creating index for {len(documents)} documents...")
        
        # Process documents in batches to avoid memory issues
        if len(documents) > batch_size:
            # Create index with first batch
            first_batch = documents[:batch_size]
            index = VectorStoreIndex.from_documents(
                first_batch,
                storage_context=storage_context,
                embed_model=self.embedding_model,
                show_progress=True,
            )
            
            # Add remaining documents in batches
            for i in range(batch_size, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 2}: documents {i}-{min(i+batch_size, len(documents))}")
                
                for doc in batch:
                    index.insert(doc)
        else:
            # Create index with all documents at once
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embedding_model,
                show_progress=True,
            )
            
        logger.info("Document indexing completed.")
        return index
        
    def save_index(self, index: VectorStoreIndex, path: Path) -> None:
        """
        Save vector index to disk.
        
        Args:
            index: Vector store index to save.
            path: Path to save the index.
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save the vector store
        vector_store = index.vector_store
        if isinstance(vector_store, FaissVectorStore):
            faiss.write_index(vector_store._faiss_index, str(path / "index.faiss"))
            
        # Save storage context
        index.storage_context.persist(persist_dir=str(path))
        
        logger.info(f"Index saved to {path}")
        
    def load_index(self, path: Path) -> VectorStoreIndex:
        """
        Load vector index from disk.
        
        Args:
            path: Path to load the index from.
            
        Returns:
            Loaded vector store index.
        """
        if not path.exists():
            raise FileNotFoundError(f"Index path does not exist: {path}")
            
        # Load FAISS index
        faiss_index = faiss.read_index(str(path / "index.faiss"))
        
        # Create vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(path),
        )
        
        # Load index from storage
        from llama_index.core import load_index_from_storage
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=self.embedding_model,
        )
        
        logger.info(f"Index loaded from {path}")
        return index