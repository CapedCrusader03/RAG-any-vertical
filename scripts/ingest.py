import os
# Force Docling to run layout models on CPU to preserve GPU memory for LLM inference
os.environ["DOCLING_DEVICE"] = "cpu"
import json
import logging
import uuid
from dotenv import load_dotenv
from pypdf import PdfReader

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from database.connection import get_db_connection
from database.search import index_chunk_tantivy, clear_tantivy_index
from embeddings.local_models import get_embeddings, get_embedding

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global progress tracker dictionary mapping task_id to progress state
ingestion_tasks = {}

def update_progress(task_id: str, state: str, log: str, percentage: int):
    if task_id:
        ingestion_tasks[task_id] = {
            "state": state,
            "log": log,
            "percentage": percentage
        }
        logger.info(f"[Task {task_id} - {percentage}%] State: {state} | Log: {log}")

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

def ingest_document(file_path: str, meta_path: str, task_id: str = None):
    """
    Ingests a single document. Reads text, chunks it, retrieves metadata,
    writes to PostgreSQL and Tantivy, reporting progress if task_id is given.
    """
    if not os.path.exists(file_path):
        err = f"Document file not found: {file_path}"
        logger.error(err)
        update_progress(task_id, "failed", err, 100)
        return
        
    if not os.path.exists(meta_path):
        err = f"[Refusal] Access control metadata missing for: {file_path}."
        logger.error(err)
        update_progress(task_id, "failed", err, 100)
        raise ValueError(err)

    # Load metadata
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        err = f"Failed to parse metadata JSON: {e}"
        logger.error(err)
        update_progress(task_id, "failed", err, 100)
        raise e
        
    allowed_roles = meta.get("allowed_roles")
    allowed_jurisdictions = meta.get("allowed_jurisdictions")
    title = meta.get("title", os.path.basename(file_path))
    source_url = meta.get("source_url", "")
    
    if not allowed_roles or not allowed_jurisdictions:
        err = f"[Refusal] allowed_roles and allowed_jurisdictions cannot be empty in {meta_path}"
        logger.error(err)
        update_progress(task_id, "failed", err, 100)
        raise ValueError(err)

    log_msg = f"Ingesting: {title} | Roles: {allowed_roles} | Jurisdictions: {allowed_jurisdictions}"
    update_progress(task_id, "processing", log_msg, 10)

    # Parse and chunk using Docling layout-aware parsing
    try:
        log_msg = f"Parsing and chunking policy file with Docling layout analyzer..."
        update_progress(task_id, "processing", log_msg, 30)
        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        chunker = HybridChunker()
        doc_chunks = list(chunker.chunk(doc))

        chunks = []
        for idx, chunk in enumerate(doc_chunks):
            # Extract page numbers safely from provenance list
            pages = set()
            if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                pages.add(prov.page_no)
                            elif isinstance(prov, dict) and 'page_no' in prov:
                                pages.add(prov['page_no'])
            
            page_number = min(pages) if pages else 1
            chunks.append({
                "content": chunk.text,
                "page_number": page_number,
                "chunk_index": idx
            })
    except Exception as e:
        logger.error(f"Docling conversion failed for {file_path}: {e}. Falling back to naive parsing.")
        log_msg = f"Docling failed, falling back to naive text extraction..."
        update_progress(task_id, "processing", log_msg, 40)
        # Parse file (Fallback)
        if file_path.lower().endswith(".pdf"):
            pages = parse_pdf(file_path)
        else:
            # Default to raw text reading for txt/md
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            pages = [{"page_number": 1, "text": raw_text}]
        # Chunk content (Fallback)
        chunks = chunk_text(pages)

    if not chunks:
        err = f"No text extracted from: {file_path}"
        logger.warning(err)
        update_progress(task_id, "failed", err, 100)
        return

    # Extract contents for batch embeddings
    log_msg = f"Generating vector embeddings for {len(chunks)} chunks..."
    update_progress(task_id, "processing", log_msg, 60)
    contents = [c["content"] for c in chunks]
    embeddings = get_embeddings(contents)

    # Insert into Database and Tantivy
    log_msg = f"Indexing document inside PostgreSQL and Tantivy search engines..."
    update_progress(task_id, "processing", log_msg, 80)
    
    try:
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
    except Exception as e:
        err = f"Database insertion failure: {e}"
        logger.error(err)
        update_progress(task_id, "failed", err, 100)
        raise e

    success_msg = f"Successfully indexed document '{title}' with {len(chunks)} chunks."
    update_progress(task_id, "completed", success_msg, 100)

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
