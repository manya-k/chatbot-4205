"""
MemoryLens - LangGraph Chatbot
Three variants with deliberately different graph structures:
 
  Variant 0: Plain LLM
    graph: retrieval (chunks) -> answer -> latency -> END
    - only chunk retrievl
    - no memory read/write
    - no conversation history
    - pure LLM from training knowledge only
 
  Variant A: Chunk Retrieval
    graph: memory_read  -> retrieval  -> answer  -> memory_write  -> latency  -> END
    - flat 300-word chunk retrieval
    - AgentState memory (read + write)
    - conversation history (last 3 turns)
 
  Variant B: Episodic Retrieval
    graph: memory_read  -> retrieval  -> answer  -> memory_write  -> latency  -> END
    - structured episode retrieval with metadata + links
    - AgentState memory (read + write)
    - conversation history (last 3 turns)
 
Comparison story:
  V0 vs VA   ->  does retrieval + memory help at all?
  VA vs VB   ->  does episodic structure specifically help?
 
Design decisions:
  - Memory stored in AgentState ONLY — no persistent database
  - Every run starts from a clean slate — no cross-run contamination
  - Variant A and B have identical graph structure
  - The ONLY variable between A and B is which ChromaDB collection
 
Hypothesis (requires refinements):
  "Does structuring retrieved context as temporally-linked episodes with
   user attribute metadata satisfy more user-stated constraints across
   multi-turn queries than retrieving fixed-size text chunks from the
   same raw data?"
 
Usage:
    python chatbot.py --mode plain     # Variant 0
    python chatbot.py --mode chunks    # Variant A
    python chatbot.py --mode episodes  # Variant B (default)
 
Requirements:
    pip install langgraph langchain langchain-ollama langchain-community chromadb ollama
    ollama pull llava
"""

import sys
import time
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "llava"

llm        = OllamaLLM(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

chunk_store = Chroma(
    collection_name="system_a_chunks",
    embedding_function=embeddings,
    persist_directory="knowledge_base/chroma_db"
)
episode_store = Chroma(
    collection_name="system_b_episodes",
    embedding_function=embeddings,
    persist_directory="knowledge_base/chroma_db"
)

"""
Agent State - based on the documentation the agent state is required as 
part of the initial steps in langraph
"""
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]  # conversation history
    query: str # current user query
    retrieved_context: str # knowledge retrieved from ChromaDB
    user_memory: List[str] # preferences loaded this turn
    mode: str # "chunks" or "episodes"
    start_time: float # latency tracking
    latency: float  # recorded response time


# ===========================================================
# Varient 0: Base LLM nodes
# separate because it does not require any memory read or write
# ============================================================
def v0_retrieval_node(state: AgentState) -> AgentState:
    """
    Variant 0: flat chunk retrieval.
    Same chunk_store as Variant A — retrieval is held constant.
    This ensures V0 vs VA comparison isolates memory only.
    """
    results = chunk_store.similarity_search(state["query"], k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    return {**state, "retrieved_context": context}
 

def v0_answer_node(state: AgentState) -> AgentState:
    """
    Variant 0: answer from retrieved context only.
    No memory prepended. No conversation history.
    Single-turn stateless response.
    """
    prompt = f"""You are a personalised travel assistant with access to travel memories.
 
RULES:
- Answer ONLY using the provided context. Do not invent facts.
- If the context does not contain the answer, say so clearly.
- Be specific — use real place names, costs, and details from the context.
 
RETRIEVED CONTEXT:
{state["retrieved_context"]}
 
User: {state["query"]}
Assistant:"""
 
    response     = llm.invoke(prompt)
    new_messages = [HumanMessage(content=state["query"]), AIMessage(content=response)]
    return {**state, "messages": new_messages}
 
  
def v0_latency_node(state: AgentState) -> AgentState:
    elapsed = round(time.time() - state.get("start_time", time.time()), 2)
    return {**state, "latency": elapsed}

# ===========================================================
# Varient A and B: with retrieval and memory
# identical nodes used by both the variants to effectively evaluate the 
# difference in the performance of the structure of thhe embeddings 
# ============================================================
def memory_read_node(state: AgentState) -> AgentState:
    """
    Node 1: Memory Read
    given that the persistant memory is out of the scope of this investigation
    memory is always stored in agent state as such memory read is not required
    had added earlier but results showed the inaccurate data so removing 
    """
    return state 


def retrieval_node(state: AgentState) -> AgentState:

    """
    Node 2: Retrieval 
    for the retrieval node, the same mechanism is applied for all the variants.
    for example:
    - same query text
    - same k=3
    - same similarity_search call

    The difference is which ChromaDB collection is searched:
        Variant A → chunk_store   (flat 300-word chunks, no metadata)
        Variant B → episode_store (structured episodes with links + metadata)
    
    This ensures any performance difference is purely due to data structure
    """
    query = state["query"]
    memory = state["user_memory"]
    K     = 3  

    if state["mode"] == "episodes":
        # Variant B: search structured episodes, 
        results = episode_store.similarity_search(query, k=K)
        parts = []
        for doc in results:
            meta = doc.metadata
            # Pass structured metadata to LLM alongside content
            # This is the structural advantage as chunks have no metadata
            part = (
                f"[{meta.get('destination','').upper()} — {meta.get('location','')}]\n"
                f"{doc.page_content}\n"
                f"Emotion: {meta.get('emotion','')}\n"
                f"Cost AUD: {meta.get('cost_aud','unknown')}\n"
                f"Tags: {meta.get('tags','')}\n"
                f"Linked next: {meta.get('linked_next','')}\n"
                f"Linked theme: {meta.get('linked_theme','')}"
            )
            parts.append(part)
        context = "\n\n---\n\n".join(parts)

    else:
        # Variant A: search flat chunks and got no links
        results = chunk_store.similarity_search(query, k=K)
        context = "\n\n".join([doc.page_content for doc in results])


    if memory:
        memory_str = "\n".join(f"- {m}" for m in memory)
        context    = f"KNOWN USER PREFERENCES:\n{memory_str}\n\n{context}"

    return {**state, "retrieved_context": context}


def answer_node(state: AgentState) -> AgentState:
    """
    Node 3: Answer
    Identical for both variants — same prompt structure, same LLM.
    Only the content of retrieved_context differs.
    """

    query   = state["query"]
    context = state["retrieved_context"]

    # Build last 3 conversation turns for context
    history_str = ""
    for msg in state["messages"][-6:]:
        if isinstance(msg, HumanMessage):
            history_str += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"Assistant: {msg.content}\n"

    prompt = f"""You are a personalised travel assistant with access to the user's own travel memories.

RULES:
- Answer ONLY using the provided context. Do not invent facts.
- Always honour every user preference or constraint mentioned in the context or conversation.
- If the context does not contain the answer, say so clearly.
- Be specific - use real place names, costs, and details from the context.
- Keep answers concise and personal.

RETRIEVED CONTEXT:
{context if context else "No context retrieved."}

{f"RECENT CONVERSATION:{chr(10)}{history_str}" if history_str else ""}

User: {query}
Assistant:"""

    response     = llm.invoke(prompt)
    new_messages = [HumanMessage(content=query), AIMessage(content=response)]
    return {**state, "messages": new_messages}


def memory_write_node(state: AgentState) -> AgentState:
    """
    Variants A + B.
    Extracts any new user constraint and stores it in AgentState only.
    Database write was later removed so that memory resets at end of session.
    this was done so that both models are evaluated fairly and to avoid the 
    impacts of structured memory 

    Clean experimental conditions: no cross-run contamination.
    """
    query = state["query"]

    prompt = f"""Extract any personal travel preference or constraint the user just revealed.

Examples:
- Budget limits: "I have $50/day"
- Dislikes: "I hate crowded places", "I don't like spicy food"
- Dietary needs: "I am vegetarian", "I am allergic to nuts"
- Physical limits: "I can't do long hikes"
- Travel style: "I prefer local food over tourist restaurants"

User said: "{query}"

If a clear personal preference or constraint was stated, return it as one short sentence.
If nothing personal was revealed, return exactly: NONE

Extracted:"""

    extracted = llm.invoke(prompt).strip()

    if extracted and extracted.upper() != "NONE" and len(extracted) > 5:
        updated_memory = state.get("user_memory", []) + [extracted]
        return {**state, "user_memory": updated_memory}

    return state


def latency_node(state: AgentState) -> AgentState:
    """
    Node 5: Latency 
    """
    elapsed = round(time.time() - state.get("start_time", time.time()), 2)
    return {**state, "latency": elapsed}

def build_graph(mode: str):
    """
    Graph Builder

    for variant 0 (plain):
        answer -> latency -> end

    for variant A (memory, chunks) and varient B (memory, structure)
        memory read -> retrieval -> answer -> memory write -> latency -> end
    """
    graph = StateGraph(AgentState)
    if mode == "plain":
        graph.add_node("retrieval", v0_retrieval_node)
        graph.add_node("answer",    v0_answer_node)
        graph.add_node("latency",   v0_latency_node)

        graph.set_entry_point("retrieval")
        graph.add_edge("retrieval", "answer")
        graph.add_edge("answer",    "latency")
        graph.add_edge("latency",   END)
 
    else:
        # chunks and episodes share identical graph structure
        graph.add_node("memory_read",  memory_read_node)
        graph.add_node("retrieval",    retrieval_node)
        graph.add_node("answer",       answer_node)
        graph.add_node("memory_write", memory_write_node)
        graph.add_node("latency",      latency_node)
 
        graph.set_entry_point("memory_read")
        graph.add_edge("memory_read",  "retrieval")
        graph.add_edge("retrieval",    "answer")
        graph.add_edge("answer",       "memory_write")
        graph.add_edge("memory_write", "latency")
        graph.add_edge("latency",      END)
 
    return graph.compile()
 

# ── Public API (used by evaluate.py) ─────────────────────────────────────────

def run_turn(state: AgentState, app) -> AgentState:
    """Run a single turn through the graph. Used by evaluate.py."""
    state["start_time"] = time.time()
    return app.invoke(state)


def build_initial_state(mode: str) -> AgentState:
    """Create a fresh state"""
    return AgentState(
        messages          = [],
        query             = "",
        retrieved_context = "",
        user_memory       = [],
        mode              = mode,
        start_time        = 0.0,
        latency           = 0.0
    )


def chat(mode: str = "episodes"):
    labels = {
        "plain": "VARIANT 0: Plain LLM (no memory: base LLM)",
        "chunks":   "VARIANT A: Chunk Retrieval (basic chunk retrieval, and memory)",
        "episodes": "VARIANT B: Episodic Memory (sturctured memory and memory)"
    }
    print("\n" + "="*55)
    print(f"  MEMORYLENS — {labels.get(mode, mode)}")
    print("="*55)
    print(f"  Memory: AgentState only (resets on clear)")
    print(f"\n  Type 'quit'   to exit")
    print(f"  Type 'memory' to see stored preferences")
    print(f"  Type 'clear'  to reset conversation\n")
 
    app   = build_graph(mode)
    state = build_initial_state(mode)
 
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
 
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "memory":
            if mode == "plain":
                print("Variant 0 has no memory")
                continue

            mem = state.get("user_memory", [])
            if mem:
                print("\n  Stored preferences this session:")
                for p in mem:
                    print(f"    - {p}")
                print()
            else:
                print("  No preferences stored yet.\n")
            continue
        if user_input.lower() == "clear":
            state = build_initial_state(mode)
            print("  Conversation cleared.\n")
            continue
 
        state["query"] = user_input
        state = run_turn(state, app)
 
        if state["messages"]:
            print(f"\nAssistant: {state['messages'][-1].content}")
            print(f"[{state['latency']}s — {mode}]\n")


if __name__ == "__main__":
    mode = "plain"

    if len(sys.argv) == 3:
        arg = sys.argv[2].lower()
        if arg in ["plain", "chunks", "episodes"]:
            mode = arg
        else: 
            print("Usage: python chatbot.py --mode [plain|chunks|episodes]")
            sys.exit(1)
    else: 
        print("Usage: python chatbot.py --mode [plain|chunks|episodes]")
        sys.exit(1)

    chat(mode=mode)
