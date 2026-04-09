# PHASE - 1 : The Execution Script

# This is the conductor. It imports your clean modules and runs the show.

from document_processor import load_and_split_pdf
from vector_db import create_retriever
from rag_pipeline import build_rag_chain

# Configuration
PDF_FILE = "vit.pdf"

def main():
    print(f"📄 1. Loading and parsing {PDF_FILE}...")
    try:
        chunks = load_and_split_pdf(PDF_FILE)
        print(f"   -> Successfully split into {len(chunks)} chunks.")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return

    print("🧠 2. Creating embeddings and vector database (this may take a moment)...")
    retriever = create_retriever(chunks)

    print("🔗 3. Building the RAG pipeline...")
    rag_chain = build_rag_chain(retriever)

    print("\n✅ System Ready!\n")
    print("-" * 50)
    
    # Simple chat loop
    while True:
        query = input("🔎 Ask a question (or type 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        print("Thinking...")
        answer = rag_chain.invoke(query)
        
        print(f"\n🤖 Answer:\n{answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()