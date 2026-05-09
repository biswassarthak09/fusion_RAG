# The PDF Parser

# read the PDF and chop it into research-optimized chunks.

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(file_path, session_id="global"):
    """Loads a PDF, splits it into semantic chunks, and adds session tags."""
    print(f"📄 Loading and chunking: {file_path} (Session: {session_id})")
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Using the optimized 2000 chunk size for legal docs!
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(pages)
    
    # 🏷️ THE NEW PART: Tag every chunk with the session_id
    for chunk in chunks:
        chunk.metadata["session_id"] = session_id
        
    print(f"   -> Created {len(chunks)} chunks.")
    return chunks