"""
Validation utilities
"""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def validate_pdf_path(pdf_path):
    """Check if PDF file exists and is valid"""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if pdf_path.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    if pdf_path.stat().st_size == 0:
        raise ValueError(f"PDF is empty: {pdf_path}")
    
    logger.info(f"✅ PDF validated: {pdf_path}")
    return pdf_path

def validate_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir