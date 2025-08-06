import PyPDF2
import requests
import os
import logging

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_pdf(url: str) -> str:
    """Download PDF from URL and save locally."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        pdf_path = "temp_policy.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Downloaded PDF from {url}")
        return pdf_path
    except Exception as e:
        logging.error(f"Failed to download PDF: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        os.remove(pdf_path)  # Clean up
        logging.info(f"Extracted text from PDF: {pdf_path}")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {str(e)}")
        raise