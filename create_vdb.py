# create_vdb.py
# Builds a single persistent ChromaDB vector store from:
#   - Multiple PDF / DOCX files
#   - Multiple URLs (web-scraped text)
#   - .urls files (plain text, one URL per line) — auto-discovered from folder
#
# Usage examples:
#   python create_vdb.py --files doc1.pdf report.docx
#   python create_vdb.py --urls https://example.com https://another.com
#   python create_vdb.py --files doc1.pdf --urls https://example.com --output ./my_db
#   python create_vdb.py --folder ./knowledge_base   # picks up all .pdf, .docx AND .urls files

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
import os
import sys
import argparse

os.environ["ANONYMIZED_TELEMETRY"] = "False"


# ── Config ────────────────────────────────────────────────────────────────────
PERSIST_DIR = "./chroma_store"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
SUPPORTED_EXTS = {".pdf", ".docx"}
URL_EXT = ".url"   # plain-text file with one URL per line
OLLAMA_MODEL = "llama3.2"  # Default model to check
# ─────────────────────────────────────────────────────────────────────────────


def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )


def load_file(file_path: str):
    """Load a single PDF or DOCX file and return raw documents."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        print(f"  📄 PDF  : {file_path}")
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        print(f"  📝 DOCX : {file_path}")
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Only .pdf and .docx are supported.")

    return loader.load()


def load_url(url: str):
    """Scrape a URL and return raw documents."""
    print(f"  🌐 URL  : {url}")
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {"timeout": 15}
    return loader.load()


def collect_files_from_folder(folder: str):
    """Return all supported doc files and URLs from .urls files inside a folder."""
    doc_files = []
    urls = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        full_path = os.path.join(folder, fname)
        if ext in SUPPORTED_EXTS:
            doc_files.append(full_path)
        elif ext == URL_EXT:
            # Windows Internet Shortcut format: contains a line like URL=https://...
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.upper().startswith("URL="):
                        # extract everything after "URL="
                        url = line[4:].strip()
                        if url:
                            print(f"  📋 Found URL in {fname}: {url}")
                            urls.append(url)
                        break  # each .url file has exactly one URL
    return doc_files, urls


def build_chunks(files: list[str], urls: list[str]):
    """Load all sources, combine, and split into chunks."""
    all_docs = []
    errors = []

    if files:
        print(f"\n📂 Loading {len(files)} file(s)...")
        for f in files:
            try:
                all_docs.extend(load_file(f))
            except Exception as e:
                errors.append(f"  ⚠️  Skipped '{f}': {e}")

    if urls:
        print(f"\n🌍 Scraping {len(urls)} URL(s)...")
        for url in urls:
            try:
                all_docs.extend(load_url(url))
            except Exception as e:
                errors.append(f"  ⚠️  Skipped '{url}': {e}")

    if errors:
        print("\n" + "\n".join(errors))

    if not all_docs:
        raise RuntimeError("No documents were loaded. Check your inputs.")

    print(
        f"\n✂️  Splitting {len(all_docs)} document section(s) into chunks...")
    chunks = get_splitter().split_documents(all_docs)

    # 🏷️ ADD THIS NEW BLOCK: Tag everything as permanent KB
    print("🏷️  Tagging all chunks with session_id='global'...")
    for chunk in chunks:
        chunk.metadata["session_id"] = "global"
    
    print(f"   -> {len(chunks)} chunks total.")
    return chunks


def save_to_vectordb(chunks, persist_dir: str):
    """Embed all chunks and persist to ChromaDB on disk."""
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"💾 Saving vector DB to: {persist_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print(
        f"✅ Vector DB saved! ({vectorstore._collection.count()} vectors stored)")
    return vectorstore


def test_ollama_connection(model_name: str = OLLAMA_MODEL) -> bool:
    """Test Ollama connection and availability."""
    print(f"\n🔗 Testing Ollama connection...")
    try:
        llm = ChatOllama(model=model_name, temperature=0.1, timeout=10.0)
        llm.invoke("test")
        print(f"   ✓ Ollama is running with model '{model_name}'")
        return True
    except Exception as e:
        print(f"   ⚠️  Ollama not available: {e}")
        print(f"\n💡 To use Ollama with fill_template.py, start it separately:")
        print(f"   1. In another terminal, run: ollama serve")
        print(f"   2. Make sure you have the model: ollama pull {model_name}")
        print(f"   3. Ollama will be available at http://localhost:11434")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build a persistent ChromaDB from multiple PDFs, DOCXs, and/or URLs."
    )
    parser.add_argument(
        "--files", "-f", nargs="+", metavar="FILE",
        help="One or more .pdf / .docx file paths"
    )
    parser.add_argument(
        "--folder", metavar="DIR",
        help="Folder containing .pdf / .docx files (all will be loaded)"
    )
    parser.add_argument(
        "--urls", "-u", nargs="+", metavar="URL",
        help="One or more URLs to scrape"
    )
    parser.add_argument(
        "--output", "-o", default=PERSIST_DIR,
        help=f"Output directory for the vector DB (default: {PERSIST_DIR})"
    )
    parser.add_argument(
        "--test-ollama", action="store_true",
        help="Test Ollama connection (for use with fill_template.py)"
    )
    parser.add_argument(
        "--ollama-model", default=OLLAMA_MODEL,
        help=f"Ollama model to test (default: {OLLAMA_MODEL})"
    )
    args = parser.parse_args()

    # Handle Ollama test mode
    if args.test_ollama:
        test_ollama_connection(args.ollama_model)
        return

    files = list(args.files or [])
    urls = list(args.urls or [])

    # Expand folder into individual files + URLs from .urls files
    if args.folder:
        folder_files, folder_urls = collect_files_from_folder(args.folder)
        if not folder_files and not folder_urls:
            print(
                f"⚠️  No supported files or .urls found in folder: {args.folder}")
        else:
            print(
                f"📁 Found {len(folder_files)} file(s) and {len(folder_urls)} URL(s) in '{args.folder}'")
            files.extend(folder_files)
            urls.extend(folder_urls)

    if not files and not urls:
        parser.print_help()
        print("\n❌ Provide at least one --files, --folder, or --urls argument.")
        sys.exit(1)

    try:
        chunks = build_chunks(files, urls)
        save_to_vectordb(chunks, args.output)

        # Test Ollama availability for template filling
        test_ollama_connection(args.ollama_model)

        print(f"\n🎉 Done! Load it later with:")
        print(f'   from langchain_chroma import Chroma')
        print(f'   from langchain_huggingface import HuggingFaceEmbeddings')
        print(
            f'   emb = HuggingFaceEmbeddings(model_name="{EMBEDDING_MODEL}")')
        print(
            f'   db  = Chroma(persist_directory="{args.output}", embedding_function=emb)')
        print(f'   retriever = db.as_retriever(search_kwargs={{"k": 5}})')

        print(f"\n📝 To fill templates, use:")
        print(f'   python fill_template.py --template myfile.docx \\')
        print(f'     --queries "PLACEHOLDER:query text"')
    except Exception as e:
        print(f"\n❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()