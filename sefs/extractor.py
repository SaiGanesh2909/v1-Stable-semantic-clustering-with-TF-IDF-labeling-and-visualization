import os
import fitz
  # PyMuPDF
def extract_text(file_path):
    if not os.path.exists(file_path):
        return ""

    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif file_path.endswith(".pdf"):
            text_chunks = []
            with fitz.open(file_path) as doc:
                for page in doc:
                    text_chunks.append(page.get_text())

            return " ".join(text_chunks[:10])  # limit pages for performance

    except:
        return ""

    return ""


