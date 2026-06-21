import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

doc_path = "data/sample_corpus/901a1401-96f7-42bc-b900-0a0c2f845b17_nc-tst.pdf"

print("=== Starting docling conversion test (no OCR) ===")
start_time = time.time()

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False # Disable OCR
pipeline_options.do_table_structure = True
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=2,
    device=AcceleratorDevice.CPU
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend
        )
    }
)

try:
    result = converter.convert(doc_path)
    doc = result.document
    print(f"Conversion finished in {time.time() - start_time:.2f} seconds.")
    
    chunker = HybridChunker()
    doc_chunks = list(chunker.chunk(doc))
    print(f"Successfully generated {len(doc_chunks)} chunks.")
except Exception as e:
    print(f"Error during conversion: {e}")
