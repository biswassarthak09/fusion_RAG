# PHASE 2 - The Workers:

# This file contains the actual "Agents" (functions) that will do the work. We have three workers:
# The Retriever: Searches the database.
# The Writer (generate_answer): Writes the draft.
# The Critic (grade_answer): Reads the draft and the PDF, and grades the answer.

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import GraphState

# PHASE - 3 : The Web Search Worker (Optional Bonus)
from ddgs import DDGS
from langchain_core.documents import Document

# 1. Initialize our local Llama 3.2 Engine
llm = ChatOllama(model="llama3.2", temperature=0)

def get_retrieve_node(retriever):
    """
    Worker 1: The Retriever.
    We wrap this in a function so we can pass our ChromaDB retriever into it later.
    """
    def retrieve(state: GraphState):
        print("\n🔍 Worker 1: Retrieving documents from PDF...")
        question = state["question"]
        
        # Search the vector database
        documents = retriever.invoke(question)
        
        # Update the clipboard with the found docs, and set revisions to 0
        return {"documents": documents, "revision_number": 0}
        
    return retrieve

def generate_answer_old(state: GraphState):
    """
    Worker 2: The Writer.
    Reads the PDF chunks and writes a draft answer.
    """
    print("✍️  Worker 2: Writing draft answer...")
    question = state["question"]
    documents = state["documents"]
    revision_number = state.get("revision_number", 0)

    # Glue all the retrieved chunks into one big block of text
    # context = "\n\n".join([doc.page_content for doc in documents])

    # Extract text AND metadata (source file and page number)
    context = "\n\n".join([
        f"Quellenname: {doc.metadata.get('source', 'Unbekannt')} | Seite: {doc.metadata.get('page', 'Unbekannt')}\nInhalt: {doc.page_content}" 
        for doc in documents
    ])

    # # ADD THESE TWO LINES TO DEBUG:
    # print("\n--- 🕵️‍♂️ DEBUG: WHAT THE AI IS READING ---")
    # print(context)
    # print("-------------------------------------------\n")

    # THE WRITER'S PROMPT (Needs {context} and {question})
    template = """<|system|>
    Du bist ein hochpräziser juristischer KI-Assistent. Du beantwortest Fragen AUSSCHLIESSLICH basierend auf dem Text unter 'Kontext'.

    REGELN:
    1. Länge: 1 bis 3 Sätze.
    2. ZITATE SIND PFLICHT: Du MUSST am Ende jedes Satzes die exakte Quelle in Klammern angeben. 
    BEISPIEL: "Der Vertrag endet 2032 (Konzessionsvertrag.pdf - Seite 1)." 
    Antworten ohne dieses Format sind strengstens verboten!
    3. Wenn die Antwort NICHT im Kontext steht, darfst du NICHTS erklären. Du darfst nur ausgeben: "Ich weiß es nicht, diese Information ist in der Wissensbasis nicht enthalten."
    </|system|>

    <|user|>
    Kontext:
    {context}

    Frage: {question}
    </|user|>

    <|assistant|>"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # Generate the draft
    generation = chain.invoke({"context": context, "question": question})
    
    # Update the clipboard with the new draft, and increase the revision count
    return {"generation": generation, "revision_number": revision_number + 1}

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_answer(state: GraphState):
    """
    Worker 2: The Writer
    """
    print("✍️  Worker 2: Writing draft answer...")
    question = state["question"]
    documents = state["documents"]
    revision_number = state.get("revision_number", 0)

    # 1. Inject the Source and Page directly above each paragraph
    context = ""
    for doc in documents:
        source = doc.metadata.get('source', 'Unbekannt')
        page = doc.metadata.get('page', 'Unbekannt')
        context += f"--- QUELLE: {source} | SEITE: {page} ---\n{doc.page_content}\n\n"

    # 2. Strict Prompt to ban bullet points and force organic citations
    template = """Du bist ein präziser Assistent. Beantworte die Frage nur mit dem Text aus dem 'Kontext'.

    STRENGE REGELN:
    1. Verwende KEINE Listen, KEINE Bulletpoints (* oder -) und KEINE Zeilenumbrüche. Schreibe exakt EINEN fließenden Satz.
    2. Du MUSST die 'QUELLE' und 'SEITE' des genutzten Textabschnitts am Ende des Satzes in Klammern schreiben.
    3. Wenn die Antwort fehlt, schreibe nur: "Ich weiß es nicht."

    Kontext:
    {context}

    Frage: {question}
    Antwort (Ein Satz, mit Quelle am Ende):"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})
    
    return {"generation": generation.strip(), "revision_number": revision_number + 1}

def grade_answer_old(state: GraphState):
    """
    Worker 3: The Critic.
    Reads the draft and checks it strictly against the PDF context.
    """
    print("🧐 Worker 3: Critic is grading the answer...")
    documents = state["documents"]
    generation = state["generation"]

    # context = "\n\n".join([doc.page_content for doc in documents])

    # Extract text AND metadata (source file and page number)
    context = "\n\n".join([
        f"Quellenname: {doc.metadata.get('source', 'Unbekannt')} | Seite: {doc.metadata.get('page', 'Unbekannt')}\nInhalt: {doc.page_content}" 
        for doc in documents
    ])

    # THE CRITIC'S PROMPT (Upgraded to handle "I don't know")
    template = """Du bist ein strenger Prüfer für juristische und kommunale Texte.
    Vergleiche den bereitgestellten Kontext mit der Antwort des Studenten.

    Bewertungskriterien:
    1. Wenn der Student die Frage korrekt und AUSSCHLIESSLICH mit Fakten aus dem Kontext beantwortet -> Note: YES
    2. Wenn der Student korrekt angibt, dass die Antwort nicht im Kontext steht (z.B. "Ich weiß es nicht") -> Note: YES
    3. Wenn der Student Zahlen oder Fakten erfindet, fantasiert oder externe Informationen nutzt -> Note: NO
    4. Wenn der Student vergisst, die Quellen zu zitieren -> Note: NO

    Kontext:
    {context}

    Antwort des Studenten:
    {generation}

    Antworte mit exakt einem Wort: YES oder NO."""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # Get the grade from the LLM
    score = chain.invoke({"context": context, "generation": generation})
    
    # Clean up the output to ensure it's just exactly "yes" or "no"
    is_supported = "yes" if "yes" in score.lower() else "no"
    
    if is_supported == "yes":
        print("   ✅ Grade: PASS (No hallucinations found)")
    else:
        print("   ❌ Grade: FAIL (Hallucination detected. Forcing rewrite!)")
        
    # Update the clipboard with the grade
    return {"is_supported": is_supported}


def grade_answer(state: GraphState):
    """
    Worker 3: The Critic.
    Reads the draft and checks it strictly against the PDF context.
    """
    print("🧐 Worker 3: Critic is grading the answer...")
    documents = state["documents"]
    generation = state["generation"]

    # context = "\n\n".join([doc.page_content for doc in documents])

    # Extract text AND metadata (source file and page number)
    context = "\n\n".join([
        f"Quellenname: {doc.metadata.get('source', 'Unbekannt')} | Seite: {doc.metadata.get('page', 'Unbekannt')}\nInhalt: {doc.page_content}" 
        for doc in documents
    ])

    # THE CRITIC'S PROMPT (Upgraded to handle "I don't know")
    template = """Du bist ein Prüfer. Vergleiche den bereitgestellten Kontext mit der Antwort des Studenten.

    Bewertungskriterien:
    1. Wenn die Antwort korrekte Fakten aus dem Kontext enthält UND eine Quelle (Seite X) zitiert -> Note: YES
    2. Wenn die Antwort "Ich weiß es nicht" lautet -> Note: YES
    3. Wenn der Student Zahlen völlig erfindet -> Note: NO

    Kontext:
    {context}

    Antwort des Studenten:
    {generation}

    Antworte mit exakt einem Wort: YES oder NO."""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # Get the grade from the LLM
    score = chain.invoke({"context": context, "generation": generation})
    
    # Clean up the output to ensure it's just exactly "yes" or "no"
    is_supported = "yes" if "yes" in score.lower() else "no"
    
    if is_supported == "yes":
        print("   ✅ Grade: PASS (No hallucinations found)")
    else:
        print("   ❌ Grade: FAIL (Hallucination detected. Forcing rewrite!)")
        
    # Update the clipboard with the grade
    return {"is_supported": is_supported}



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