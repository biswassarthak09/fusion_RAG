# PHASE 2 & 3: The Conveyor Belt:

# This file acts as the Factory Manager. 
# It uses LangGraph to define exactly how the clipboard (GraphState) 
# moves between your three workers.

from langgraph.graph import END, StateGraph
from state import GraphState
from nodes import get_retrieve_node, generate_answer, grade_answer

def build_graph(retriever, vectorstore=None, filenames=None, session_id=None):
    """Wires the workers together into an Adaptive RAG loop."""
    
    workflow = StateGraph(GraphState)

    # 1. Add all our workers
    workflow.add_node("retrieve", get_retrieve_node(retriever, vectorstore, filenames, session_id))
    workflow.add_node("generate", generate_answer)
    workflow.add_node("grade", grade_answer)

    # 2. Direct Entry Point (Bypassing the router entirely)
    workflow.set_entry_point("retrieve")

    # 3. Connect the flow
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade")

    def decide_to_generate(state: GraphState):
        if state["is_supported"] == "yes":
            return "useful"
        elif state["revision_number"] >= 3:
            print("   ⚠️ Max revisions reached. Outputting best attempt.")
            return "useful"
        else:
            # Re-retrieve instead of just re-generating with the same chunks
            return "re_retrieve"

    # 4. The Self-Correcting Loop — re-retrieve on failure for fresh context
    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {
            "useful": END,
            "re_retrieve": "retrieve"
        }
    )

    return workflow.compile()