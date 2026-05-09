import os
import uuid

# Disable ChromaDB telemetry to prevent annoying console warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# 1. GLOBAL EMBEDDING MODEL
# ==========================================
# This MUST match the model used in create_vdb.py exactly!
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# ==========================================
# 2. DATABASE MANAGEMENT FUNCTIONS
# ==========================================

def load_permanent_database(persist_dir="./chroma_store"):
    """
    Loads the Permanent Knowledge Base created by create_vdb.py.
    Does NOT re-embed documents, just loads the existing database from disk.
    """
    print(f"🗄️ Loading Permanent Knowledge Base from '{persist_dir}'...")
    
    vectorstore = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embedding_model
    )
    
    print(f"   -> Loaded {vectorstore._collection.count()} permanent chunks.")
    return vectorstore

def add_temporary_doc(vectorstore, temp_chunks, session_id):
    """
    Tags temporary chunks with a session_id and adds them to the live database.
    Returns the unique IDs of the inserted chunks so they can be deleted later.
    """
    print(f"📥 Adding temporary document for session: '{session_id}'...")
    
    # Tag every chunk with the session ID
    for chunk in temp_chunks:
        chunk.metadata["session_id"] = session_id
        
    # Generate unique UUIDs for precise deletion later
    ids = [str(uuid.uuid4()) for _ in temp_chunks]
    
    # Insert into the database
    vectorstore.add_documents(documents=temp_chunks, ids=ids)
    
    print(f"   -> Added {len(temp_chunks)} temporary chunks.")
    return ids

def remove_temporary_doc(vectorstore, ids):
    """
    Deletes specific temporary chunks from the database using their unique IDs.
    """
    if ids:
        print(f"🗑️ Removing {len(ids)} temporary chunks from database...")
        vectorstore.delete(ids=ids)
    else:
        print("⚠️ No temporary documents to remove.")

# ==========================================
# 3. RETRIEVER LOGIC
# ==========================================

def get_filtered_retriever(vectorstore, current_session_id):
    """
    Creates a retriever that strictly searches the permanent KB ('global') 
    AND the user's specific temporary document.
    """
    return vectorstore.as_retriever(
        search_kwargs={
            "k": 15, # Fetch top 15 chunks to ensure we don't miss dates/facts
            "filter": {
                # 🚦 The Magic Filter: Only search these two session IDs
                "session_id": {"$in": ["global", current_session_id]}
            }
        }
    )

def get_specific_doc_retriever(vectorstore, filenames, current_session_id):
    """
    Creates a retriever that uses similarity search for a specific file.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
        
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 20,
            "filter": {
                "$and": [
                    {"source": {"$in": filenames}},
                    {"session_id": {"$in": ["global", current_session_id]}}
                ]
            }
        }
    )


def get_keyword_chunks(vectorstore, filenames, keywords):
    """
    Directly fetches all chunks from the given file(s) whose text contains
    any of the provided keywords. Used as a guaranteed top-up for the retriever
    when semantic search misses critical sections (e.g. contract duration clauses).
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    all_chunks = vectorstore.get(where={"source": {"$in": filenames}})
    matched = []
    from langchain_core.documents import Document
    for doc_id, meta, text in zip(all_chunks['ids'], all_chunks['metadatas'], all_chunks['documents']):
        text_lower = text.lower()
        if any(kw.lower() in text_lower for kw in keywords):
            matched.append(Document(page_content=text, metadata=meta))
    return matched