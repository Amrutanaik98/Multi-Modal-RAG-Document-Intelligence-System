"""
Global configuration for the project
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import pytesseract

# IMPORTANT: Tell Python where Tesseract is installed
# Windows path - adjust if you installed to different location
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Processing Config
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 100))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
PDF_DPI = 150  # Lower DPI = faster processing
OCR_LANGUAGE = "eng"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "financial-documents")

# Logging
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validation
if not OPENAI_API_KEY:
    print("⚠️ Warning: OPENAI_API_KEY not set in .env")

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data Dir: {RAW_DATA_DIR}")
    print(f"Processed Data Dir: {PROCESSED_DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")