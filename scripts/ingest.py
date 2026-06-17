import os
import json
import logging
import uuid
from dotenv import load_dotenv
from pypdf import PdfReader

from database.connection import get_db_connection
from database.search import index_chunk_tantivy, clear_tantivy_index
from embeddings.local_models import get_embeddings, get_embedding

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def parse_pdf(file_path: str) -> list[dict]:
    """Extract page-by-page text content from a PDF file."""
    reader = PdfReader(file_path)
    pages_data = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_data.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })
    return pages_data

def chunk_text(pages_data: list[dict]) -> list[dict]:
    """Semantic chunking with sliding-window page alignment."""
    chunks = []
    chunk_index = 0
    
    current_chunk = []
    current_length = 0
    current_page = 1
    
    for page in pages_data:
        text = page["text"]
        page_num = page["page_number"]
        words = text.split()
        
        for i in range(0, len(words), 100): # Process words in small windows
            segment = words[i:i+150] # overlapping sliding window
            segment_text = " ".join(segment)
            
            chunks.append({
                "content": segment_text,
                "page_number": page_num,
                "chunk_index": chunk_index
            })
            chunk_index += 1
            
    return chunks

def ingest_document(file_path: str, meta_path: str):
    """
    Ingests a single document. Reads text, chunks it, retrieves metadata,
    writes to PostgreSQL and Tantivy.
    """
    if not os.path.exists(file_path):
        logger.error(f"Document file not found: {file_path}")
        return
        
    if not os.path.exists(meta_path):
        raise ValueError(f"[Refusal] Access control metadata missing for: {file_path}. Regulated deployments must have role & jurisdiction tags.")

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    allowed_roles = meta.get("allowed_roles")
    allowed_jurisdictions = meta.get("allowed_jurisdictions")
    title = meta.get("title", os.path.basename(file_path))
    source_url = meta.get("source_url", "")
    
    if not allowed_roles or not allowed_jurisdictions:
        raise ValueError(f"[Refusal] allowed_roles and allowed_jurisdictions cannot be empty in {meta_path}")

    logger.info(f"Ingesting: {title} | Roles: {allowed_roles} | Jurisdictions: {allowed_jurisdictions}")

    # Parse file
    if file_path.lower().endswith(".pdf"):
        pages = parse_pdf(file_path)
    else:
        # Default to raw text reading for txt/md
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        pages = [{"page_number": 1, "text": raw_text}]

    # Chunk content
    chunks = chunk_text(pages)
    if not chunks:
        logger.warning(f"No text extracted from: {file_path}")
        return

    # Extract contents for batch embeddings
    contents = [c["content"] for c in chunks]
    embeddings = get_embeddings(contents)

    # Insert into Database and Tantivy
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 1. Create document record
            doc_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO documents (id, title, source_url) VALUES (%s, %s, %s);",
                (doc_id, title, source_url)
            )
            
            # 2. Insert chunks
            for idx, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                embedding = embeddings[idx]
                
                # Write to PG
                cursor.execute(
                    """
                    INSERT INTO document_chunks 
                    (id, document_id, content, embedding, allowed_roles, allowed_jurisdictions, chunk_index, page_number)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        chunk_id, 
                        doc_id, 
                        chunk["content"], 
                        embedding, 
                        allowed_roles, 
                        allowed_jurisdictions, 
                        chunk["chunk_index"], 
                        chunk["page_number"]
                    )
                )
                
                # Write to Tantivy
                index_chunk_tantivy(
                    chunk_id=chunk_id,
                    content=chunk["content"],
                    roles=allowed_roles,
                    jurisdictions=allowed_jurisdictions
                )
        conn.commit()

    logger.info(f"Successfully indexed document '{title}' with {len(chunks)} chunks.")

def ingest_directory(directory_path: str):
    """Scan directory for document files and ingest them alongside metadata."""
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return
        
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".meta.json"):
            continue # Metadata is processed alongside document
            
        doc_path = os.path.join(directory_path, file_name)
        base_name, _ = os.path.splitext(file_name)
        meta_path = os.path.join(directory_path, f"{base_name}.meta.json")
        
        try:
            ingest_document(doc_path, meta_path)
        except Exception as e:
            logger.error(f"Failed to ingest {file_name}: {e}")

if __name__ == "__main__":
    # Standard manual trigger for testing
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        logger.info(f"Running batch ingestion on: {target_dir}")
        ingest_directory(target_dir)
    else:
        logger.info("Usage: python scripts/ingest.py <directory_path>")
