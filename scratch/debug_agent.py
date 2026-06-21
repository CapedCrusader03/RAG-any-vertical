import logging
from agent.graph import rag_agent

logging.basicConfig(level=logging.INFO)

test_q = "What is Texas water limit?"
roles = ["underwriter"]
jurs = ["US-NC"]

initial_state = {
    "query": test_q,
    "user_roles": roles,
    "user_jurisdictions": jurs,
    "retrieved_chunks": [],
    "reranked_chunks": [],
    "answer": "",
    "is_safe": True,
    "safety_message": "",
    "broken_citations": []
}

print("=== Streaming Graph Execution ===")
for event in rag_agent.stream(initial_state):
    for node_name, state_update in event.items():
        print(f"\n>>> Node Finished: {node_name} <<<")
        for k, v in state_update.items():
            if k in ["retrieved_chunks", "reranked_chunks"]:
                print(f"  {k}: count = {len(v)}")
            elif k == "answer":
                print(f"  {k}: {repr(v)}")
            else:
                print(f"  {k}: {v}")
