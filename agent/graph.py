import os
import logging
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

# Import search & models
from database.search import hybrid_search
from embeddings.local_models import rerank
from agent.guardrails import input_safety_check, scrub_pii, validate_citations

# 1. Load Configurations and Initialize Telemetry
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telemetry setup for Arize Phoenix
# Only activate if the Phoenix daemon is expected to be running
if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        
        provider = TracerProvider()
        # Arize Phoenix default HTTP OTLP trace receiver path is /v1/traces
        endpoint = f"{os.getenv('PHOENIX_COLLECTOR_ENDPOINT')}/v1/traces"
        processor = SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        
        OpenAIInstrumentor().instrument()
        logger.info(f"OpenTelemetry OpenAI tracing enabled pointing to Phoenix at: {endpoint}")
    except Exception as e:
        logger.warning(f"Could not initialize Arize Phoenix tracing: {e}")

# Setup clients: dynamic check for Gemini API or local LM Studio
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")

if GEMINI_API_KEY:
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=GEMINI_API_KEY
    )
    model_name = GEMINI_MODEL
    logger.info(f"Using Google AI Studio via OpenAI compatibility layer with model: {model_name}")
else:
    LM_STUDIO_API_BASE = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
    LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "gemma-e4b")
    client = OpenAI(
        base_url=LM_STUDIO_API_BASE,
        api_key="lm-studio"
    )
    model_name = LM_STUDIO_MODEL
    logger.info(f"Using local LM Studio with model: {model_name}")


# 2. Define LangGraph Agent State
class AgentState(TypedDict):
    query: str
    user_roles: list[str]
    user_jurisdictions: list[str]
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    answer: str
    is_safe: bool
    safety_message: str
    broken_citations: list[str]


# 3. Define Graph Nodes
def input_guardrail_node(state: AgentState) -> dict:
    """Verifies that the user prompt is safe before running retrieval."""
    logger.info("Starting input guardrail safety verification...")
    is_safe, message = input_safety_check(state["query"], client, model_name)
    return {
        "is_safe": is_safe,
        "safety_message": message
    }

def retrieve_node(state: AgentState) -> dict:
    """Executes pre-filtered hybrid search across dense and sparse indexes."""
    logger.info(f"Executing hybrid retrieval for user roles: {state['user_roles']}, jurisdictions: {state['user_jurisdictions']}...")
    chunks = hybrid_search(
        query_text=state["query"],
        user_roles=state["user_roles"],
        user_jurisdictions=state["user_jurisdictions"],
        limit=10
    )
    return {"retrieved_chunks": chunks}

def rerank_node(state: AgentState) -> dict:
    """Reranks candidates using local BGE cross-encoder."""
    logger.info("Executing local cross-encoder reranking...")
    reranked = rerank(
        query=state["query"],
        chunks=state["retrieved_chunks"],
        top_k=3
    )
    return {"reranked_chunks": reranked}

def synthesize_node(state: AgentState) -> dict:
    """Generates the final response using either Google Gemini API or local LM Studio."""
    logger.info("Synthesizing answer...")
    
    # Format context with stable, deterministic headings to maximize prompt cache performance
    context_str = ""
    for idx, chunk in enumerate(state["reranked_chunks"]):
        doc_title = chunk.get("doc_title", "Unknown Policy")
        page_num = chunk.get("page_number", "?")
        context_str += f"\n[Document: {doc_title}, page {page_num}]\nContent: {chunk['content']}\n"
        
    system_prompt = """You are an expert regulated-domain insurance chatbot helper. You answer user queries based ONLY on the provided policy context.
Strict Compliance Rules:
1. Citation Enforcement: You MUST cite your sources. For every factual claim, append a bracketed citation pointing to the exact Source Anchor found at the end of the matching context content, such as [TX-WD-2026-V1, Section I, page 3] or [CA-WF-2026-V2, Section I, page 5]. Do not use generic placeholders or modify the anchors. You must use the literal Source Anchor from the text.
2. Strict Context Boundary: Do not make up information. If the context does not contain the answer, say "I am sorry, but the provided documentation does not contain the information required to answer your query."
3. No Role Leakage: Do not mention permissions, allowed roles, or jurisdictions to the user."""

    user_prompt = f"""<context>
{context_str}
</context>

<query>
{state['query']}
</query>
Response:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Keep temperature low for facts
            max_tokens=512
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {"answer": f"Error: Synthesis failure. Details: {e}"}

def output_guardrail_node(state: AgentState) -> dict:
    """Validates citations in generated answer and scrubs PII."""
    logger.info("Running output guardrail validations...")
    answer = state["answer"]
    
    # 1. Validate Citations
    is_valid_citation, filtered_ans, broken_cits = validate_citations(answer, state["reranked_chunks"])
    
    if not is_valid_citation:
        logger.warning(f"Citation validation failed due to broken citations: {broken_cits}")
        # Append clear warning or refuse response if it violates verification standards
        refusal_msg = f"Refusal: The generated answer contains unverifiable citations {broken_cits} or is missing citations. Please revise query."
        return {
            "answer": refusal_msg,
            "broken_citations": broken_cits
        }
        
    # 2. Scrub PII
    scrubbed_ans = scrub_pii(filtered_ans)
    
    return {
        "answer": scrubbed_ans,
        "broken_citations": []
    }


# 4. Build LangGraph Workflow State Machine
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("synthesize", synthesize_node)
workflow.add_node("output_guardrail", output_guardrail_node)

# Define Logic Paths (Edges)
workflow.set_entry_point("input_guardrail")

def safety_router(state: AgentState):
    """Routes state based on safety verdict."""
    if state["is_safe"]:
        return "retrieve"
    else:
        return END

workflow.add_conditional_edges(
    "input_guardrail",
    safety_router,
    {
        "retrieve": "retrieve",
        END: END
    }
)

workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "synthesize")
workflow.add_edge("synthesize", "output_guardrail")
workflow.add_edge("output_guardrail", END)

# Compile the graph
rag_agent = workflow.compile()


# 5. Execution Interface
def run_rag_agent(query: str, roles: list[str], jurisdictions: list[str]) -> AgentState:
    """Runs the full RAG pipeline state machine for a user query and returns state results."""
    initial_state = {
        "query": query,
        "user_roles": roles,
        "user_jurisdictions": jurisdictions,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "is_safe": True,
        "safety_message": "",
        "broken_citations": []
    }
    
    # Execute graph
    final_state = rag_agent.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    # Smoke test query
    test_q = "What is the liability cap for water damage claims in Texas?"
    roles = ["guest"]
    jurs = ["US-TX"]
    
    print(f"Executing test query: '{test_q}' with credentials: {roles} in {jurs}")
    result = run_rag_agent(test_q, roles, jurs)
    print("\n--- AGENT ANSWER ---")
    print(result["answer"])
    print("--------------------")
