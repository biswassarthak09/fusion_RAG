# The Vector Database

# takes those text chunks, turns them into math (embeddings), and stores them so we can search them.
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False" # Disables ChromaDB tracking

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_retriever(chunks):
    """Creates embeddings and a searchable vector database from text chunks."""
    
    # Using a model excellent for dense, scientific text
    embedding_model = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/all-mpnet-base-v2"
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    
    # Create an in-memory Chroma database
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model
    )
    
    # Return a retriever configured to fetch the top 15 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 15})