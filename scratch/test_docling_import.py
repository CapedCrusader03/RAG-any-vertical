import os
os.environ["DOCLING_DEVICE"] = "cpu"
import sys

try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    print("Imports successful!")
    
    # Try converting one of the files
    converter = DocumentConverter()
    sample_file = r"d:\Projects\RAG-any-vertical\data\sample_corpus\policy_tx.txt"
    if os.path.exists(sample_file):
        result = converter.convert(sample_file)
        doc = result.document
        chunker = HybridChunker()
        chunks = list(chunker.chunk(doc))
        print(f"Successfully converted and chunked sample text file into {len(chunks)} chunks.")
        if chunks:
            chunk = chunks[0]
            print(f"Chunk type: {type(chunk)}")
            print(f"Chunk text: {chunk.text[:100]}...")
            print(f"Chunk meta: {chunk.meta}")
            print(f"Chunk meta attributes: {dir(chunk.meta)}")
            if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                item = chunk.meta.doc_items[0]
                print(f"Item attributes: {dir(item)}")
                if hasattr(item, 'prov'):
                    print(f"Prov attributes: {item.prov}")
    else:
        print(f"Sample file not found at: {sample_file}")
except Exception as e:
    print(f"Error occurred: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
