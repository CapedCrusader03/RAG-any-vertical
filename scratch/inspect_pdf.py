from pypdf import PdfReader

doc_path = "data/sample_corpus/901a1401-96f7-42bc-b900-0a0c2f845b17_nc-tst.pdf"
reader = PdfReader(doc_path)
print(f"Total Pages: {len(reader.pages)}")

# Inspect page 15
page_idx = 14 # 0-indexed page 15
page = reader.pages[page_idx]
print(f"--- Page 15 (Index {page_idx}) Info ---")
print(f"MediaBox: {page.mediabox}")
print(f"CropBox: {page.cropbox}")
print(f"Text Length: {len(page.extract_text())}")
images = page.images
print(f"Number of Images: {len(images)}")
for idx, img in enumerate(images):
    print(f"  Image {idx}: {img.name}, size: {len(img.data)} bytes")
