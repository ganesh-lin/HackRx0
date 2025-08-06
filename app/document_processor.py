import PyPDF2
import requests
import os

def download_pdf(url: str) -> str:
    """Download PDF from URL and save locally."""
    response = requests.get(url)
    pdf_path = "temp_policy.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    return pdf_path

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    os.remove(pdf_path)  # Clean up
    return text