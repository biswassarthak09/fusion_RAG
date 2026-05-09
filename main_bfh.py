import os
from document_processor import load_and_chunk_pdf
from graph_builder import build_graph
from state import GraphState # Or wherever your GraphState is defined!
from typing import cast
from vector_db import load_permanent_database, add_temporary_doc, remove_temporary_doc, get_specific_doc_retriever, get_keyword_chunks
from template_filler import fill_template

def main():
    print("🚀 Initializing Focused RAG System...")
    vectorstore = load_permanent_database("./chroma_store")
    
    # 🔍 FETCH UNIQUE FILENAMES FROM DB
    # This ensures your menu matches exactly what is in the database
    db_content = vectorstore.get()
    all_kb_files = sorted(list(set([
        m['source'] for m in db_content['metadatas'] 
        if m.get('session_id') == 'global'
    ])))
    
    temp_session_id = "user_temp_session"
    temp_chunk_ids = []
    selected_kb_doc = None

    while True:
        print("\n" + "="*55)
        status = f"📄 ACTIVE: {selected_kb_doc}" if selected_kb_doc else "❌ NO BASE DOC SELECTED"
        print(f"{status} {'(+ Temp Doc)' if temp_chunk_ids else ''}")
        print("-" * 55)
        print("1. Select Base Document (from KB)")
        print("2. Ask a Question")
        print("3. Add Additional Temporary PDF")
        print("4. Reset / Clear Selection")
        print("5. Quit")
        print("6. Fill Template (Maßnahmenliste from all PDFs)")
        choice = input("Select option (1-6): ")

        if choice == '1':
            print("\nAvailable Documents:")
            for i, doc in enumerate(all_kb_files):
                print(f"{i+1}. {os.path.basename(doc)}")
            idx = int(input("Select number: ")) - 1
            selected_kb_doc = all_kb_files[idx]
            print(f"✅ Context locked to: {selected_kb_doc}")

        elif choice == '2':
            if not selected_kb_doc:
                print("⚠️ Please select a base document first!")
                continue
            
            question = input("\n🔎 Question: ")
            # Create a retriever that ONLY sees the chosen file + temp file
            retriever = get_specific_doc_retriever(vectorstore, selected_kb_doc, temp_session_id)
            app = build_graph(retriever, vectorstore, selected_kb_doc, temp_session_id)
            
            inputs = cast(GraphState, {"question": question, "revision_number": 0})
            final_state = app.invoke(inputs)
            
            print("\n🤖 Answer:")
            if final_state.get("is_supported") == "no":
                print("Ich weiß es nicht. (Info nicht im gewählten Dokument gefunden.)")
            else:
                print(final_state.get("generation"))

        elif choice == '3':
            path = input("\nEnter temp PDF path: ")
            if os.path.exists(path):
                if temp_chunk_ids: remove_temporary_doc(vectorstore, temp_chunk_ids)
                chunks = load_and_chunk_pdf(path, session_id=temp_session_id)
                temp_chunk_ids = add_temporary_doc(vectorstore, chunks, temp_session_id)
                print(f"✅ {path} added to current context.")

        elif choice == '4':
            selected_kb_doc = None
            if temp_chunk_ids: remove_temporary_doc(vectorstore, temp_chunk_ids)
            temp_chunk_ids = []
            print("🔄 Context cleared.")

        elif choice == '5':
            if temp_chunk_ids: remove_temporary_doc(vectorstore, temp_chunk_ids)
            break

        elif choice == '6':
            print("\n📋 Starting template fill from all 5 PDF documents...")
            output_path = fill_template(vectorstore, all_kb_files)
            if output_path:
                print(f"\n✅ Done! File saved to: {output_path}")
            else:
                print("⚠️  Template fill failed or no measures found.")

if __name__ == "__main__":
    main()