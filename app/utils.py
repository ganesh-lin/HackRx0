import requests
import logging
import re
import os
import hashlib
import time
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, unquote
import mimetypes
from datetime import datetime, timedelta

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_pdf(url: str) -> str:
    """Download PDF from URL (backward compatibility)."""
    return download_file(url)

def download_file(url: str, max_size_mb: int = 50) -> str:
    """Download file from URL with size limits and validation."""
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        # Make request with headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValueError(f"File too large: {size_mb:.1f}MB > {max_size_mb}MB")
        
        # Determine file extension
        content_type = response.headers.get('content-type', '').lower()
        file_extension = get_file_extension_from_url_or_content(url, content_type)
        
        # Create temporary file
        timestamp = int(time.time())
        filename = f"temp_document_{timestamp}{file_extension}"
        
        # Download with size check
        total_size = 0
        max_size_bytes = max_size_mb * 1024 * 1024
        
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    total_size += len(chunk)
                    if total_size > max_size_bytes:
                        os.remove(filename)
                        raise ValueError(f"File too large during download: {total_size/1024/1024:.1f}MB")
                    f.write(chunk)
        
        logging.info(f"Downloaded file from {url}: {total_size/1024/1024:.1f}MB")
        return filename
        
    except Exception as e:
        logging.error(f"Failed to download file from {url}: {str(e)}")
        raise

def get_file_extension_from_url_or_content(url: str, content_type: str = "") -> str:
    """Determine file extension from URL or content type."""
    # Try URL first
    parsed_url = urlparse(url)
    path = unquote(parsed_url.path)
    
    if path:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.pdf', '.docx', '.doc', '.txt', '.html', '.eml', '.msg']:
            return ext
    
    # Try content type
    if content_type:
        if 'pdf' in content_type:
            return '.pdf'
        elif 'word' in content_type or 'document' in content_type:
            return '.docx'
        elif 'text' in content_type:
            return '.txt'
        elif 'html' in content_type:
            return '.html'
    
    # Default to PDF
    return '.pdf'

def extract_filename_from_url(url: str) -> str:
    """Extract a reasonable filename from URL."""
    try:
        parsed_url = urlparse(url)
        path = unquote(parsed_url.path)
        
        if path and path != '/':
            # Get the last part of the path
            filename = os.path.basename(path)
            if filename and '.' in filename:
                return filename
        
        # Fallback: use domain + timestamp
        domain = parsed_url.netloc.replace('www.', '').replace('.', '_')
        timestamp = int(time.time())
        return f"{domain}_{timestamp}.pdf"
        
    except Exception:
        # Ultimate fallback
        return f"document_{int(time.time())}.pdf"

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logging.error(f"Failed to calculate file hash: {e}")
        return None

def validate_question(question: str) -> bool:
    """Validate if a question is reasonable."""
    if not question or not question.strip():
        return False
    
    # Check length
    if len(question.strip()) < 5 or len(question) > 1000:
        return False
    
    # Check for reasonable question words
    question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'does', 'is', 'are', 'can', 'will']
    question_lower = question.lower()
    
    # Should contain at least one question word or end with question mark
    has_question_word = any(word in question_lower for word in question_words)
    has_question_mark = question.strip().endswith('?')
    
    return has_question_word or has_question_mark

def sanitize_text(text: str) -> str:
    """Sanitize text by removing potentially harmful content."""
    if not text:
        return ""
    
    # Remove potential code injection patterns
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def format_processing_time(seconds: float) -> str:
    """Format processing time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text (rough approximation)."""
    # Very rough estimation: ~4 characters per token for English
    return max(1, len(text) // 4)

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def is_valid_url(url: str) -> bool:
    """Validate if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def get_mime_type(file_path: str) -> str:
    """Get MIME type of a file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters."""
    # Remove invalid characters for filenames
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove excessive dots and spaces
    filename = re.sub(r'\.+', '.', filename)
    filename = re.sub(r'\s+', '_', filename)
    
    # Ensure reasonable length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    
    return name + ext

def parse_time_period(text: str) -> Optional[Dict[str, int]]:
    """Parse time periods from text (e.g., '30 days', '2 years')."""
    patterns = [
        (r'(\d+)\s*days?', 'days'),
        (r'(\d+)\s*weeks?', 'weeks'),
        (r'(\d+)\s*months?', 'months'),
        (r'(\d+)\s*years?', 'years')
    ]
    
    results = {}
    
    for pattern, unit in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            results[unit] = [int(match) for match in matches]
    
    return results if results else None

def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    # Pattern to match integers and decimals
    pattern = r'\b\d+(?:\.\d+)?\b'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]

def extract_currency_amounts(text: str) -> List[Dict[str, Any]]:
    """Extract currency amounts from text."""
    patterns = [
        (r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', 'INR'),
        (r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', 'USD'),
        (r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rupees?|rs\.?)', 'INR'),
        (r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:dollars?|usd)', 'USD')
    ]
    
    amounts = []
    
    for pattern, currency in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            amount_str = match.group(1).replace(',', '')
            try:
                amount = float(amount_str)
                amounts.append({
                    'amount': amount,
                    'currency': currency,
                    'text': match.group(0),
                    'position': match.span()
                })
            except ValueError:
                continue
    
    return amounts

def calculate_confidence_score(matches: int, total_chunks: int, query_complexity: str = "medium") -> float:
    """Calculate confidence score based on various factors."""
    if total_chunks == 0:
        return 0.0
    
    # Base score from match ratio
    match_ratio = min(matches / max(total_chunks * 0.1, 1), 1.0)
    base_score = match_ratio * 0.8
    
    # Adjust for query complexity
    complexity_multipliers = {
        "simple": 1.1,
        "medium": 1.0,
        "complex": 0.9
    }
    
    multiplier = complexity_multipliers.get(query_complexity, 1.0)
    
    # Add bonus for having matches
    if matches > 0:
        base_score += 0.2
    
    final_score = min(base_score * multiplier, 1.0)
    return round(final_score, 2)

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (2 ** attempt)
            logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def create_cache_key(*args, **kwargs) -> str:
    """Create a cache key from arguments."""
    key_parts = []
    
    for arg in args:
        key_parts.append(str(arg))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def is_expired(timestamp: datetime, expiry_hours: int = 24) -> bool:
    """Check if a timestamp is expired."""
    expiry_time = timestamp + timedelta(hours=expiry_hours)
    return datetime.now() > expiry_time