import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: str) -> str:
    print(f"📄 Extracting text from: {file_path}")
    doc = fitz.open(file_path)
    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text:
            full_text.append(text.strip())

    return "\n".join(full_text)
