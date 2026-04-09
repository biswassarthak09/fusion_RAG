# PHASE 2 - The Workers:

# This file contains the actual "Agents" (functions) that will do the work. We have three workers:
# The Retriever: Searches the database.
# The Writer (generate_answer): Writes the draft.
# The Critic (grade_answer): Reads the draft and the PDF, and grades the answer.

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
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

def generate_answer(state: GraphState):
    """
    Worker 2: The Writer.
    Reads the PDF chunks and writes a draft answer.
    """
    print("✍️  Worker 2: Writing draft answer...")
    question = state["question"]
    documents = state["documents"]
    revision_number = state.get("revision_number", 0)

    # Glue all the retrieved chunks into one big block of text
    context = "\n\n".join([doc.page_content for doc in documents])

    # THE WRITER'S PROMPT (Needs {context} and {question})
    template = """You are an AI Research Assistant. Answer the question using ONLY the provided context.
    If you don't know the answer, say "I don't know."
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # Generate the draft
    generation = chain.invoke({"context": context, "question": question})
    
    # Update the clipboard with the new draft, and increase the revision count
    return {"generation": generation, "revision_number": revision_number + 1}


def grade_answer(state: GraphState):
    """
    Worker 3: The Critic.
    Reads the draft and checks it strictly against the PDF context.
    """
    print("🧐 Worker 3: Critic is grading the answer...")
    documents = state["documents"]
    generation = state["generation"]

    context = "\n\n".join([doc.page_content for doc in documents])

    # THE CRITIC'S PROMPT (Upgraded to handle "I don't know")
    template = """You are a strict grader. Look at the retrieved context and the student's answer.
    
    Grading Rubric:
    1. If the student correctly answers the question using ONLY facts from the context -> Grade: YES
    2. If the student correctly states that the answer is not in the context (e.g., 'I don't know', 'not mentioned') -> Grade: YES
    3. If the student makes up numbers, facts, or hallucinates information -> Grade: NO
    
    Context:
    {context}
    
    Student Answer:
    {generation}
    
    Respond with exactly one word: YES or NO."""
    
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