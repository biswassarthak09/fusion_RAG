# PHASE 2 - The Workers:

# This file contains the actual "Agents" (functions) that will do the work. We have three workers:
# The Retriever: Searches the database.
# The Writer (generate_answer): Writes the draft.
# The Critic (grade_answer): Reads the draft and the PDF, and grades the answer.

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import GraphState
import os

# PHASE - 3 : The Web Search Worker (Optional Bonus)
from ddgs import DDGS
from langchain_core.documents import Document

# 1. Initialize our local Llama 3.2 Engine
llm = ChatOllama(model="llama3.2", temperature=0)

# Keywords that signal contract duration / termination clauses
_CONTRACT_KEYWORDS = [
    "vertragsdauer", "laufzeit", "kündigung", "kündigungsfrist",
    "vertragsablauf", "vertragsende", "beginnt am", "endet am"
]

def get_retrieve_node(retriever, vectorstore=None, filenames=None, session_id=None):
    """
    Worker 1: The Retriever.
    Combines semantic search with a keyword top-up to guarantee critical
    contract clauses (duration, termination) are always included.
    """
    def retrieve(state: GraphState):
        print("\n🔍 Worker 1: Retrieving documents from PDF...")
        question = state["question"]
        
        # 1. Semantic search
        documents = retriever.invoke(question)

        # 2. Keyword top-up: always include chunks with contract-critical terms
        if vectorstore is not None and filenames is not None:
            from vector_db import get_keyword_chunks
            keyword_docs = get_keyword_chunks(vectorstore, filenames, _CONTRACT_KEYWORDS)
            for doc in keyword_docs:
                # Deduplicate by content
                if not any(doc.page_content[:100] == d.page_content[:100] for d in documents):
                    documents.append(doc)
            if keyword_docs:
                print(f"   📌 Keyword top-up added {len(keyword_docs)} contract-clause chunk(s).")

        print(f"\n--- 🛠️ RAG DEBUG: RETRIEVED CHUNKS ({len(documents)} total) ---")
        if not documents:
            print("❌ NO CHUNKS FOUND! Check your metadata filter.")
        for i, doc in enumerate(documents[:5]):
            source = doc.metadata.get('source')
            page = doc.metadata.get('page')
            content_snippet = doc.page_content[:200].replace('\n', ' ')
            print(f"Chunk {i+1} | Source: {source} | Page: {page}")
            print(f"Snippet: {content_snippet}...")
        print("--------------------------------------\n")
        
        # Preserve revision_number so the loop counter keeps incrementing across re-retrieves
        current_revision = state.get("revision_number", 0)
        return {"documents": documents, "revision_number": current_revision}
        
    return retrieve


def generate_answer(state: GraphState):
    """
    Worker 2: The Writer
    """
    print("✍️  Worker 2: Writing draft answer...")
    question = state["question"]
    documents = state["documents"]
    revision_number = state.get("revision_number", 0)

    # 1. Sort: keyword-matched chunks first so the LLM sees them at the top,
    #    then cap at 6 chunks to stay within llama3.2's effective context window.
    keyword_terms = [
        "vertragsdauer", "laufzeit", "kündigung", "kündigungsfrist",
        "vertragsablauf", "vertragsende", "beginnt am", "endet am"
    ]
    def _has_keyword(doc):
        return any(kw in doc.page_content.lower() for kw in keyword_terms)

    priority = [d for d in documents if _has_keyword(d)]
    rest = [d for d in documents if not _has_keyword(d)]
    top_docs = (priority + rest)[:8]  # keyword chunks first, cap at 8 total

    context = ""
    for doc in top_docs:
        # Use os.path.basename to make the citation look cleaner (e.g. "vertrag.pdf" instead of "C:/path/vertrag.pdf")
        source = os.path.basename(doc.metadata.get('source', 'Unbekannt'))
        page = doc.metadata.get('page', 'Unbekannt')
        context += f"--- QUELLE: {source} | SEITE: {page} ---\n{doc.page_content}\n\n"

    # 2. General writer prompt — works for any question type
    template = """Du bist ein präziser Assistent. Beantworte die Frage ausschließlich auf Basis des bereitgestellten Kontexts.

    STRENGE REGELN:
    1. Lies den gesamten Kontext sorgfältig durch und suche nach allen Abschnitten, die zur Frage passen.
    2. WICHTIG: Verwende NUR Informationen, die wörtlich im Kontext stehen. Erfinde NICHTS.
    3. WICHTIG: Achte auf die genauen Rollen der Parteien. 'Die Gesellschaft' oder 'Konzessionär' ist das Energieversorgungsunternehmen (z.B. badenova). 'Die Stadt' ist der Konzedent.
    4. Schreibe in vollständigen Sätzen. Keine Listen.
    5. Nenne am Ende die QUELLE und SEITE in Klammern.
    6. Schreibe 'Ich weiß es nicht.' NUR wenn der Kontext keine relevanten Informationen enthält.

    Kontext:
    {context}

    Frage: {question}
    Antwort:"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    print(f"\n📝 Writer output: {generation.strip()[:300]}")
    
    return {"generation": generation.strip(), "revision_number": revision_number + 1}

import re as _re


def _extract_key_facts(text: str) -> list[str]:
    """Extracts verifiable numeric tokens: years, dates, 'N Jahre' patterns."""
    facts = []
    facts += _re.findall(r'\b(20\d{2})\b', text)
    facts += _re.findall(r'\b\d{2}\.\d{2}\.\d{4}\b', text)
    facts += [m.replace(' ', '') for m in _re.findall(r'\b\d+\s*jahre\b', text.lower())]
    return list(set(facts))


def _extract_key_phrases(text: str, min_len: int = 6) -> list[str]:
    """
    Extracts meaningful noun phrases from text by taking runs of
    non-stopword words. Used for grounded text overlap checking.
    """
    stopwords = {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
        "einem", "einer", "und", "oder", "ist", "sind", "hat", "haben",
        "wird", "werden", "von", "zu", "in", "im", "an", "auf", "mit",
        "für", "als", "auch", "sich", "nicht", "nach", "bei", "aus",
        "durch", "über", "unter", "zwischen", "sowie", "dass", "wenn",
        "the", "a", "an", "of", "in", "is", "are", "to", "and", "or"
    }
    words = _re.findall(r'\b[a-zA-ZäöüÄÖÜß]{4,}\b', text.lower())
    return [w for w in words if w not in stopwords and len(w) >= min_len]


def grade_answer(state: GraphState):
    """
    Worker 3: The Critic.

    Grading strategy (works reliably with llama3.2 / any model size):

    Step 1 — Pre-check: if writer said 'Ich weiß es nicht' → FAIL immediately.

    Step 2 — Numeric fact check: extract dates/years from source chunks.
              If the answer contains any of them → PASS (grounded in source).

    Step 3 — Text overlap check: extract meaningful words from the answer
              and check what fraction appear in the source chunks.
              If overlap ≥ 40% → PASS (answer is grounded in source text).
              This replaces the unreliable LLM critic for qualitative questions,
              since llama3.2 cannot follow strict YES/NO format instructions.

    NOTE: A larger model (llama3.1:8b+) could replace Steps 2-3 with a single
    reliable LLM call. The LLM critic call is kept below as a logged step for
    transparency, but the decision is made by the deterministic checks above.
    """
    print("🧐 Worker 3: Critic is grading the answer...")
    generation = state["generation"]
    documents = state["documents"]

    # Step 1: writer gave up
    if "ich weiß es nicht" in generation.lower():
        print("   ❌ Grade: FAIL (Writer said 'I don't know' — forcing rewrite!)")
        return {"is_supported": "no"}

    # Use the top keyword-matched chunks as the critic's reference
    keyword_terms = [
        "vertragsdauer", "laufzeit", "kündigung", "kündigungsfrist",
        "vertragsablauf", "vertragsende", "beginnt am", "endet am",
        "verpflichtet", "pflicht", "gesellschaft"
    ]
    priority = [d for d in documents if any(kw in d.page_content.lower() for kw in keyword_terms)]
    rest = [d for d in documents if d not in priority]
    critic_docs = (priority + rest)[:4]
    source_text = " ".join(doc.page_content for doc in critic_docs)

    # Step 2: numeric fact check
    source_facts = _extract_key_facts(source_text)
    answer_facts = _extract_key_facts(generation)
    grounded_facts = [f for f in answer_facts if f in source_facts]
    if grounded_facts:
        print(f"   ✅ Grade: PASS (Numeric facts verified: {grounded_facts})")
        return {"is_supported": "yes"}

    # Step 3: text overlap check
    source_phrases = set(_extract_key_phrases(source_text))
    answer_phrases = _extract_key_phrases(generation)
    if answer_phrases:
        overlap = sum(1 for p in answer_phrases if p in source_phrases)
        overlap_ratio = overlap / len(answer_phrases)
        print(f"   📊 Text overlap: {overlap}/{len(answer_phrases)} words ({overlap_ratio:.0%})")
        if overlap_ratio >= 0.40:
            print("   ✅ Grade: PASS (Answer is grounded in source text)")
            return {"is_supported": "yes"}

    # If both checks fail, the answer likely contains hallucinated content
    print("   ❌ Grade: FAIL (Low overlap with source — likely hallucination)")
    return {"is_supported": "no"}



# PHASE - 3 : The Web Search Worker (Optional Bonus)
def route_question(state: GraphState):
    """
    The Front Door Router.
    Decides if the question is about the PDF or needs a web search.
    """
    print("\n🚦 Router: Analyzing question intent...")
    question = state["question"]
    
    # We ask the LLM to classify the question
    template = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains the user's currently loaded documents or YouTube video transcripts.
    
    Rules:
    - Use 'vectorstore' for ANY question that sounds like it relates to the characters, events, or topics in the provided video or text.
    - Use 'websearch' ONLY for questions about live internet data (like stock prices, weather, or current events).
    
    Question: {question}
    
    Respond with exactly one word: vectorstore OR websearch"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    decision = chain.invoke({"question": question})
    
    if "websearch" in decision.lower():
        print("   -> 🌐 Routing to Web Search")
        return "websearch"
    else:
        print("   -> 📚 Routing to PDF Database")
        return "vectorstore"

def web_search(state: GraphState):
    """
    Worker 4: The Web Researcher.
    Uses the native DDGS library to bypass outdated LangChain wrappers.
    """
    print("🌐 Worker 4: Searching the live internet...")
    question = state["question"]
    
    try:
        # Connect directly to DuckDuckGo
        with DDGS() as ddgs:
            # Grab the top 3 search results
            results = list(ddgs.text(question, max_results=3))
            
            # Extract just the text snippets and glue them together
            if results:
                docs = "\n\n".join([result["body"] for result in results])
                print("   -> Successfully retrieved live web data!")
            else:
                docs = "No results found on the internet."
                
    except Exception as e:
        print(f"   -> ⚠️ Web search failed: {e}")
        docs = "There was an error accessing the internet."
    
    # Wrap the raw text in a Document so the Writer/Critic know how to read it
    web_results = Document(page_content=docs)
    
    return {"documents": [web_results], "revision_number": 0}