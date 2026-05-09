# PHASE 2 : New Conductor

from document_processor import load_and_split_pdf
from vector_db import create_retriever
from graph_builder import build_graph

# Configuration
PDF_FILE = "Konzessionsvertrag_Freiburg_Wasser.PDF"

def main():
    print(f"📄 1. Loading and parsing {PDF_FILE}...")
    try:
        chunks = load_and_split_pdf(PDF_FILE)
        print(f"   -> Successfully split into {len(chunks)} chunks.")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return

    print("🧠 2. Creating embeddings and vector database...")
    retriever = create_retriever(chunks)

    print("🔗 3. Building the LangGraph Self-Correcting Loop...")
    app = build_graph(retriever)

    print("\n✅ System Ready!\n")
    print("-" * 50)
    
    # Simple chat loop
    while True:
        query = input("🔎 Ask a question (or type 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        # In LangGraph, we pass a dictionary matching our GraphState
        inputs = {"question": query}
        
        # Run the loop!
        result = app.invoke(inputs) # type: ignore
        
        # The final answer lives inside the 'generation' key of the state
        print(f"\n🤖 Final Verified Answer:\n{result['generation']}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()