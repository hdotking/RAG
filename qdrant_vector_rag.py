import logging
import sys
from typing import Sequence

import qdrant_client
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.qdrant_client import QdrantClient


class RAGSystem:
    def __init__(
        self,
        collection_name: str,
        embedding_model_name: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_size: int = 768,
        distance: Distance = Distance.COSINE,
    ):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_size = embedding_size
        self.distance = distance

        # Configure logging
        self._setup_logging()

        # Initialize Qdrant client - Interface between your app and the vector DB.
        self.client = self._initialize_qdrant_client(qdrant_host, qdrant_port)

        # Set up embedding model
        self.embed_model = FastEmbedEmbedding(model_name=embedding_model_name)

    def _setup_logging(self) -> None:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    def _initialize_qdrant_client(self, host: str, port: int) -> QdrantClient:
        return qdrant_client.QdrantClient(host=host, port=port)

    def ensure_collection(self, vector_params: VectorParams) -> None:
        """Check if collection exists and create it if necessary."""
        collections = self.client.get_collections()
        existing_collections = [col.name for col in collections.collections]

        if self.collection_name not in existing_collections:
            print(f"Collection '{self.collection_name}' does not exist. Creating it...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params,
            )
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def create_index(
        self,
        documents: Sequence[Document],
    ) -> VectorStoreIndex:
        """Create vector store, storage context, and index."""
        # Define vector parameters with desired similarity measure
        vector_params = VectorParams(
            size=self.embedding_size,
            distance=self.distance,
        )

        # Ensure collection exists
        self.ensure_collection(vector_params)

        # Initialize the vector store  - an abstraction over the vector DB.
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            dense_config=vector_params,
        )

        #  Create storage context (Llama Index unified container for storage-related objects)
        # e.g., vector store, index, graph store, document store.
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from documents. VectorStoreIndex is the most common index.
        # It takes your Documents, splits them up into Nodes, then creates vector embeddings of the text of every node.
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )
        return index

    def create_query_engine(self, index: VectorStoreIndex) -> RetrieverQueryEngine:
        """Create the query engine."""
        # Configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=2,
        )

        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE,
        )

        # Assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return query_engine

    def query(self, query_engine: RetrieverQueryEngine, query: str):
        """Run a query against the index."""
        response = query_engine.query(query)
        return response


def main() -> None:
    # Initialize system
    rag_system = RAGSystem(
        collection_name="paul_graham",
        embedding_model_name="BAAI/bge-base-en-v1.5",
    )

    # Load local documents
    documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

    # Create index
    index = rag_system.create_index(documents)

    # Create query engine then query the engine.
    # https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/
    query_engine = rag_system.create_query_engine(index)
    response = rag_system.query(
        query_engine,
        "Who is Robert Morris, and what role did he play in the author's career transitions and projects?",
    )
    print(response)


if __name__ == "__main__":
    main()
