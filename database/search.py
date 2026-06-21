import os
import logging
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import tantivy

from database.connection import get_db_connection
from embeddings.local_models import get_embedding

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
INDEX_DIR = os.getenv("TANTIVY_INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "tantivy_index"))
RRF_K = 60

# 1. Initialize Tantivy Schema and Index
os.makedirs(INDEX_DIR, exist_ok=True)

schema_builder = tantivy.SchemaBuilder()
schema_builder.add_text_field("chunk_id", stored=True)
schema_builder.add_text_field("content", stored=True)
schema_builder.add_text_field("allowed_roles", stored=True) # Space-separated tokens
schema_builder.add_text_field("allowed_jurisdictions", stored=True) # Space-separated tokens
schema = schema_builder.build()

# Create or open the index
try:
    tantivy_index = tantivy.Index(schema, path=INDEX_DIR)
    logger.info(f"Opened existing Tantivy index at: {INDEX_DIR}")
except Exception:
    tantivy_index = tantivy.Index(schema, path=INDEX_DIR, reuse=False)
    logger.info(f"Created new Tantivy index at: {INDEX_DIR}")


def index_chunk_tantivy(chunk_id: str, content: str, roles: list[str], jurisdictions: list[str]):
    """Add or update a document chunk inside the Tantivy index."""
    writer = tantivy_index.writer()
    roles_str = " ".join(roles)
    jurs_str = " ".join(jurisdictions)
    
    # In production, we might want to delete first if overwrite is supported
    # tantivy-py doesn't expose easy delete_term, so we assume clean builds or manual clears.
    writer.add_document(tantivy.Document(
        chunk_id=chunk_id,
        content=content,
        allowed_roles=roles_str,
        allowed_jurisdictions=jurs_str
    ))
    writer.commit()


def index_chunks_tantivy(chunks: list[dict]):
    """Add multiple document chunks inside the Tantivy index in a single batch commit."""
    writer = tantivy_index.writer()
    try:
        for chunk in chunks:
            roles_str = " ".join(chunk["roles"])
            jurs_str = " ".join(chunk["jurisdictions"])
            writer.add_document(tantivy.Document(
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                allowed_roles=roles_str,
                allowed_jurisdictions=jurs_str
            ))
        writer.commit()
    except Exception as e:
        logger.error(f"Failed to batch index chunks in Tantivy: {e}")
        raise e


def clear_tantivy_index():
    """Wipes the Tantivy index clean."""
    global tantivy_index
    writer = tantivy_index.writer()
    # To clear, we delete the index directory and reinitialize
    writer.commit()
    import shutil
    try:
        shutil.rmtree(INDEX_DIR)
        os.makedirs(INDEX_DIR, exist_ok=True)
        tantivy_index = tantivy.Index(schema, path=INDEX_DIR, reuse=False)
        logger.info("Tantivy index cleared and re-initialized.")
    except Exception as e:
        logger.error(f"Failed to clear Tantivy index: {e}")


# 2. Database Pre-Filtered Dense Retrieval
def dense_search(
    query_embedding: list[float], 
    user_roles: list[str], 
    user_jurisdictions: list[str], 
    limit: int = 20
) -> list[dict]:
    """Execute metadata-filtered dense search inside pgvector."""
    query_sql = """
        SELECT 
            c.id::text as chunk_id, 
            c.content, 
            c.allowed_roles, 
            c.allowed_jurisdictions, 
            c.page_number,
            d.title as doc_title,
            (c.embedding <=> %s::vector) as distance
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE 
            c.allowed_roles && %s::varchar[] 
            AND c.allowed_jurisdictions && %s::varchar[]
        ORDER BY distance ASC
        LIMIT %s;
    """
    
    results = []
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query_sql, (query_embedding, user_roles, user_jurisdictions, limit))
                for row in cursor.fetchall():
                    # Distance to similarity: 1 - cosine_distance
                    row["score"] = 1.0 - float(row["distance"])
                    results.append(dict(row))
    except Exception as e:
        logger.error(f"Dense search execution failed: {e}")
    return results


# 3. Tantivy Pre-Filtered Sparse Retrieval
def sparse_search(
    query_text: str, 
    user_roles: list[str], 
    user_jurisdictions: list[str], 
    limit: int = 20
) -> list[dict]:
    """Execute token-filtered sparse search inside Tantivy index using BM25."""
    # Clean the query string from special chars that crash query parser
    cleaned_query = "".join([c if c.isalnum() or c.isspace() else "" for c in query_text])
    if not cleaned_query.strip():
        return []
        
    # Construct filters: (allowed_roles:role1 OR allowed_roles:role2) AND (allowed_jurisdictions:jur1)
    roles_filter = " OR ".join([f"allowed_roles:{role}" for role in user_roles])
    jurs_filter = " OR ".join([f"allowed_jurisdictions:{jur}" for jur in user_jurisdictions])
    
    full_query_str = f"content:({cleaned_query})"
    if roles_filter:
        full_query_str += f" AND ({roles_filter})"
    if jurs_filter:
        full_query_str += f" AND ({jurs_filter})"
        
    logger.debug(f"Tantivy query: {full_query_str}")
    
    results = []
    try:
        searcher = tantivy_index.searcher()
        # Parse query
        query = tantivy_index.parse_query(full_query_str, ["content"])
        hits = searcher.search(query, limit).hits
        
        for score, doc_address in hits:
            doc = searcher.doc(doc_address)
            results.append({
                "chunk_id": doc["chunk_id"][0],
                "content": doc["content"][0],
                "allowed_roles": doc["allowed_roles"][0].split() if doc["allowed_roles"] else [],
                "allowed_jurisdictions": doc["allowed_jurisdictions"][0].split() if doc["allowed_jurisdictions"] else [],
                "score": float(score)
            })
    except Exception as e:
        logger.error(f"Sparse search execution failed: {e}")
    return results


# 4. Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(
    dense_results: list[dict], 
    sparse_results: list[dict], 
    limit: int = 10
) -> list[dict]:
    """Merge and re-rank results from dense and sparse search using Reciprocal Rank Fusion."""
    rrf_scores = {}
    doc_map = {}
    
    # Process dense results
    for rank, doc in enumerate(dense_results):
        doc_id = doc["chunk_id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank + 1))
        
    # Process sparse results
    for rank, doc in enumerate(sparse_results):
        doc_id = doc["chunk_id"]
        # Merge dict keys but preserve dense database metadata if available (e.g. document title/page)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        else:
            # Update content if necessary
            doc_map[doc_id].update({k: v for k, v in doc.items() if k not in doc_map[doc_id]})
            
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank + 1))
        
    # Sort documents by score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    final_results = []
    for doc_id in sorted_doc_ids[:limit]:
        merged_doc = doc_map[doc_id]
        merged_doc["rrf_score"] = rrf_scores[doc_id]
        final_results.append(merged_doc)
        
    return final_results


def hybrid_search(
    query_text: str, 
    user_roles: list[str], 
    user_jurisdictions: list[str], 
    limit: int = 10
) -> list[dict]:
    """Pre-filtered hybrid retrieval pipeline."""
    # 1. Generate dense query embedding
    query_emb = get_embedding(query_text, is_query=True)
    
    # 2. Run retrievals concurrently or sequentially
    dense_hits = dense_search(query_emb, user_roles, user_jurisdictions, limit=limit*2)
    sparse_hits = sparse_search(query_text, user_roles, user_jurisdictions, limit=limit*2)
    
    # 3. Fuse rankings via RRF
    fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits, limit=limit)
    return fused_hits
