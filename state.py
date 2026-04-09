# PHASE 2 : The Memory

# In LangGraph, you don't just pass text from one function to the next. 
# You pass a State (a dictionary). 
# Think of this like a clipboard that gets passed between the workers. 
# As each worker finishes their job, they update the clipboard.

from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    This is the 'clipboard' that gets passed around our loop.
    Every node can read from it and write to it.
    """
    question: str             # The user's original question
    documents: List[Document] # The chunks of text retrieved from the PDF
    generation: str           # The draft answer written by the AI
    revision_number: int      # How many times we've tried to rewrite it
    is_supported: str         # The Critic's grade: "yes" or "no"