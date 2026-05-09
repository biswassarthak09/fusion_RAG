# The PDF Parser

# read the PDF and chop it into research-optimized chunks.

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path: str):
    """Loads a PDF and splits it into chunks optimized for research papers."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")

    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split the text
    # 1000 chars with 200 overlap is ideal for keeping scientific context intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""] 
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks