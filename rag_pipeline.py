# PHASE - 1 : The RAG Pipeline

# sets up the LLM, the instructions (prompt), and connects the search engine to the brain.

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(retriever):
    """Builds the chain connecting the retriever, prompt, and local LLM."""
    
    # Initialize the local model via Ollama
    llm = ChatOllama(model="llama3.2", temperature=0.1)

    # Define how the AI should behave
    template = """You are an AI Research Assistant analyzing a technical paper.
    Use ONLY the retrieved context below to answer the user's question.
    
    Guidelines:
    1. Focus on Methodology, Implementation, or Results if relevant.
    2. If the context does not contain the answer, strictly say: "The paper does not mention this."
    
    Context:
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # Build the pipeline: Question -> Retrieve Context -> Fill Prompt -> LLM -> Text Output
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain