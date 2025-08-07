import PyPDF2
import requests
import os
import logging
import re
import tempfile
import time
import uuid
from urllib.parse import urlparse
from typing import List, Dict, Any
from docx import Document
import email
from email.mime.text import MIMEText
from bs4 import BeautifulSoup
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DocumentProcessor:
    def __init__(self):
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", 512))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        if TIKTOKEN_AVAILABLE:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None
    
    def download_document(self, url: str) -> str:
        """Download document from URL and save locally."""
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Determine file extension from URL or content type
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1].lower()
            
            if not file_extension:
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_extension = '.pdf'
                elif 'docx' in content_type or 'document' in content_type:
                    file_extension = '.docx'
                else:
                    file_extension = '.txt'
            
            # Create temporary file with unique name
            unique_id = str(uuid.uuid4())[:8]
            timestamp = str(int(time.time()))
            temp_filename = f"hackrx_doc_{timestamp}_{unique_id}{file_extension}"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, prefix=temp_filename) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            logging.info(f"Downloaded document from {url} to {temp_path}")
            return temp_path
            
        except Exception as e:
            logging.error(f"Failed to download document: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with improved error handling."""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        logging.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                
                logging.info(f"Extracted text from PDF: {len(text)} characters")
                
        except Exception as e:
            logging.error(f"Failed to extract text from PDF: {str(e)}")
            raise
        finally:
            # Ensure file is closed and removed
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {pdf_path}: {e}")
        
        return self.clean_text(text)

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file."""
        text = ""
        try:
            doc = Document(docx_path)
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + "\t"
                    text += "\n"
                
            logging.info(f"Extracted text from DOCX: {len(text)} characters")
                
        except Exception as e:
            logging.error(f"Failed to extract text from DOCX: {str(e)}")
            raise
        finally:
            # Ensure file is closed and removed
            try:
                if os.path.exists(docx_path):
                    os.remove(docx_path)
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {docx_path}: {e}")
        
        return self.clean_text(text)

    def extract_text_from_email(self, email_path: str) -> str:
        """Extract text from email file."""
        try:
            with open(email_path, 'r', encoding='utf-8') as file:
                msg = email.message_from_file(file)
            
            text = ""
            
            # Extract headers
            text += f"Subject: {msg.get('Subject', 'N/A')}\n"
            text += f"From: {msg.get('From', 'N/A')}\n"
            text += f"To: {msg.get('To', 'N/A')}\n"
            text += f"Date: {msg.get('Date', 'N/A')}\n\n"
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    elif part.get_content_type() == "text/html":
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        soup = BeautifulSoup(html_content, 'html.parser')
                        text += soup.get_text()
            else:
                if msg.get_content_type() == "text/plain":
                    text += msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif msg.get_content_type() == "text/html":
                    html_content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text += soup.get_text()
            
            # Clean up temporary file
            if os.path.exists(email_path):
                os.remove(email_path)
                
            logging.info(f"Extracted text from email: {len(text)} characters")
            return self.clean_text(text)
            
        except Exception as e:
            logging.error(f"Failed to extract text from email: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\\&\%\$\#\@]', ' ', text)
        
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def process_document(self, url: str) -> str:
        """Main method to process any document type."""
        try:
            file_path = self.download_document(url)
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                return self.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                return self.extract_text_from_docx(file_path)
            elif file_extension in ['.eml', '.msg']:
                return self.extract_text_from_email(file_path)
            else:
                # Try to read as plain text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                os.remove(file_path)
                return self.clean_text(text)
                
        except Exception as e:
            logging.error(f"Failed to process document: {str(e)}")
            raise

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into meaningful chunks with overlap."""
        try:
            # Clean text and normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Split text into sentences with better handling
            # Use more comprehensive sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            chunks = []
            current_chunk = ""
            current_tokens = 0
            chunk_id = 0
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                sentence = sentence.strip()
                
                # Calculate tokens - fallback to word count if tiktoken not available
                if self.encoding:
                    sentence_tokens = len(self.encoding.encode(sentence))
                else:
                    sentence_tokens = len(sentence.split()) * 1.3  # Rough approximation
                
                # If adding this sentence would exceed max chunk size
                if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        "id": chunk_id,
                        "text": current_chunk.strip(),
                        "tokens": current_tokens,
                        "start_sentence": chunk_id * 10,  # Approximate
                        "end_sentence": chunk_id * 10 + len(current_chunk.split('.'))
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = ""
                    overlap_tokens = 0
                    
                    # Add overlap from end of previous chunk
                    current_sentences = current_chunk.split('.')
                    for i in range(min(3, len(current_sentences))):  # Take last 3 sentences for overlap
                        overlap_sentence = current_sentences[-(i+1)].strip()
                        if overlap_sentence:
                            if self.encoding:
                                overlap_sentence_tokens = len(self.encoding.encode(overlap_sentence))
                            else:
                                overlap_sentence_tokens = len(overlap_sentence.split()) * 1.3
                            
                            if overlap_tokens + overlap_sentence_tokens <= self.chunk_overlap:
                                overlap_text = overlap_sentence + ". " + overlap_text
                                overlap_tokens += overlap_sentence_tokens
                            else:
                                break
                    
                    current_chunk = overlap_text + sentence + ". "
                    current_tokens = overlap_tokens + sentence_tokens
                    chunk_id += 1
                else:
                    current_chunk += sentence + ". "
                    current_tokens += sentence_tokens
            
            # Add the last chunk if it has content
            if current_chunk.strip():
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk.strip(),
                    "tokens": current_tokens,
                    "start_sentence": chunk_id * 10,
                    "end_sentence": chunk_id * 10 + len(current_chunk.split('.'))
                })
            
            # Post-process: merge very small chunks
            final_chunks = []
            i = 0
            while i < len(chunks):
                current = chunks[i]
                
                # If chunk is very small, try to merge with next
                if i < len(chunks) - 1 and current["tokens"] < 100:
                    next_chunk = chunks[i + 1]
                    if current["tokens"] + next_chunk["tokens"] <= self.max_chunk_size:
                        merged_chunk = {
                            "id": current["id"],
                            "text": current["text"] + " " + next_chunk["text"],
                            "tokens": current["tokens"] + next_chunk["tokens"],
                            "start_sentence": current["start_sentence"],
                            "end_sentence": next_chunk["end_sentence"]
                        }
                        final_chunks.append(merged_chunk)
                        i += 2  # Skip next chunk as it's merged
                        continue
                
                final_chunks.append(current)
                i += 1
            
            logging.info(f"Created {len(final_chunks)} chunks from text (after merging)")
            return final_chunks
            
        except Exception as e:
            logging.error(f"Failed to chunk text: {str(e)}")
            # Fallback: simple splitting
            chunk_size = 1000
            chunks = []
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                tokens = len(chunk_text.split()) if not self.encoding else len(self.encoding.encode(chunk_text))
                chunks.append({
                    "id": i // chunk_size,
                    "text": chunk_text,
                    "tokens": tokens,
                    "start_sentence": i,
                    "end_sentence": i + chunk_size
                })
            return chunks

# Backward compatibility functions
def download_pdf(url: str) -> str:
    """Backward compatibility function."""
    processor = DocumentProcessor()
    return processor.download_document(url)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Backward compatibility function."""
    processor = DocumentProcessor()
    return processor.extract_text_from_pdf(pdf_path)