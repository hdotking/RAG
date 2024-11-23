import logging
import sys

import qdrant_client
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set embedding model
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
sample_text = "Test input text"
embedding = embed_model.get_text_embedding(sample_text)
print(f"Sample embedding dimension: {len(embedding)}")

# Load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# Initialize Qdrant client
client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333,
)

# Define vector parameters with desired similarity measure
vector_params = VectorParams(
    size=768,  # Ensure this matches the embedding dimensions
    distance=Distance.COSINE,  # Choose the appropriate distance metric
)


# Initialize the vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="paul_graham",
    dense_config=vector_params,
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index from documents
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,  # Explicitly pass the embedding model
)

# Configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2,
)

# Configure response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

# Assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# Query the index
response = query_engine.query(
    "Who is Robert Morris, and what role did he play in the author's career transitions and projects?"
)
print(response)
