import requests

def download_pdf(url: str) -> str:
    """Download PDF from URL."""
    response = requests.get(url)
    pdf_path = "temp_policy.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    return pdf_path