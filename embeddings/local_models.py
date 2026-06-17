import logging
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device allocation: use CUDA if available, fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE} for local embeddings and reranking")

# Preload embedding model (nomic-ai/nomic-embed-text-v1.5)
# Note: nomic-embed-text-v1.5 requires query prefix for retrieval tasks.
logger.info("Loading embedding model 'nomic-ai/nomic-embed-text-v1.5'...")
try:
    embedding_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", 
        trust_remote_code=True,
        device=DEVICE
    )
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}. Falling back to cpu.")
    embedding_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", 
        trust_remote_code=True,
        device="cpu"
    )

# Preload reranking model (BAAI/bge-reranker-large)
logger.info("Loading reranker model 'BAAI/bge-reranker-large'...")
try:
    reranker_model = CrossEncoder(
        "BAAI/bge-reranker-large",
        device=DEVICE
    )
    logger.info("Reranker model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load reranker model: {e}. Falling back to cpu.")
    reranker_model = CrossEncoder(
        "BAAI/bge-reranker-large",
        device="cpu"
    )

def get_embedding(text: str, is_query: bool = False) -> list[float]:
    """
    Generates embedding vector for a given text chunk.
    Nomic Embed requires a specific prefix to toggle task-specific parameters.
    """
    prefix = "search_query: " if is_query else "search_document: "
    text_to_embed = prefix + text
    
    # Generate embedding
    embedding = embedding_model.encode(text_to_embed, convert_to_numpy=True)
    return embedding.tolist()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a batch of document chunks."""
    prefixed_texts = ["search_document: " + text for text in texts]
    embeddings = embedding_model.encode(prefixed_texts, convert_to_numpy=True)
    return embeddings.tolist()

def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Rerank a list of retrieved chunks relative to the user query.
    Returns the top_k chunks sorted by their reranker score.
    """
    if not chunks:
        return []
        
    pairs = [[query, chunk["content"]] for chunk in chunks]
    scores = reranker_model.predict(pairs)
    
    # Attach scores to chunks and sort
    for idx, score in enumerate(scores):
        chunks[idx]["rerank_score"] = float(score)
        
    sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_chunks[:top_k]

if __name__ == "__main__":
    # Smoke tests
    test_text = "Water damage limits in Texas insurance policy"
    emb = get_embedding(test_text, is_query=True)
    print(f"Embedding dimensions: {len(emb)}")
    
    test_chunks = [
        {"content": "Texas insurance liability coverage handles water damage up to $10,000."},
        {"content": "California regulatory compliance rules for auto insurance policies."},
        {"content": "Underwriting criteria for high-risk properties in Florida."}
    ]
    reranked = rerank(test_text, test_chunks, top_k=2)
    print("Reranked results:")
    for chunk in reranked:
        print(f"Score: {chunk['rerank_score']:.4f} | Content: {chunk['content']}")
