import logging
from scripts.ingest import ingest_document

logging.basicConfig(level=logging.INFO)

doc_path = "data/sample_corpus/901a1401-96f7-42bc-b900-0a0c2f845b17_nc-tst.pdf"
meta_path = "data/sample_corpus/901a1401-96f7-42bc-b900-0a0c2f845b17_nc-tst.meta.json"

print(f"=== Testing Ingestion on: {doc_path} ===")
try:
    ingest_document(doc_path, meta_path, task_id="test-run-1")
    print("=== SUCCESS ===")
except Exception as e:
    print(f"=== FAILED: {e} ===")
