import requests
import logging

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_pdf(url: str) -> str:
    """Download PDF from URL."""
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