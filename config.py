"""
Complete Configuration for Educational RAG System
Enhanced version with all necessary configurations
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# Data directories (for educational content)
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, LOGS_DIR, NOTEBOOKS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TEXT PROCESSING CONFIGURATION
# ============================================================================
CHUNK_SIZE = 1000  # How many characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
BATCH_SIZE = 10  # How many documents to process at once
MIN_CHUNK_SIZE = 100  # Minimum chunk size (avoid tiny chunks)
MAX_CHUNK_SIZE = 2000  # Maximum chunk size

# ============================================================================
# EDUCATIONAL TOPICS (Data sources we'll scrape)
# ============================================================================
EDUCATIONAL_TOPICS = {
    'machine_learning': [
        'https://en.wikipedia.org/wiki/Machine_learning',
        'https://en.wikipedia.org/wiki/Supervised_learning',
        'https://en.wikipedia.org/wiki/Unsupervised_learning',
        'https://en.wikipedia.org/wiki/Reinforcement_learning'
    ],
    'deep_learning': [
        'https://en.wikipedia.org/wiki/Deep_learning',
        'https://en.wikipedia.org/wiki/Artificial_neural_network',
        'https://en.wikipedia.org/wiki/Transformer_(machine_learning)',
        'https://en.wikipedia.org/wiki/Convolutional_neural_network',
        'https://en.wikipedia.org/wiki/Recurrent_neural_network'
    ],
    'nlp': [
        'https://en.wikipedia.org/wiki/Natural_language_processing',
        'https://en.wikipedia.org/wiki/Word_embedding',
        'https://en.wikipedia.org/wiki/Language_model',
        'https://en.wikipedia.org/wiki/Named-entity_recognition',
        'https://en.wikipedia.org/wiki/Sentiment_analysis'
    ],
    'llm': [
        'https://en.wikipedia.org/wiki/Large_language_model',
        'https://en.wikipedia.org/wiki/Transformer_(machine_learning)',
        'https://en.wikipedia.org/wiki/Prompt_engineering',
        'https://en.wikipedia.org/wiki/Transfer_learning'
    ],
    'rag': [
        'https://en.wikipedia.org/wiki/Information_retrieval',
        'https://en.wikipedia.org/wiki/Document_retrieval',
        'https://en.wikipedia.org/wiki/Semantic_search'
    ],
    'ai_basics': [
        'https://en.wikipedia.org/wiki/Artificial_intelligence',
        'https://en.wikipedia.org/wiki/Machine_learning_model',
        'https://en.wikipedia.org/wiki/Neural_network'
    ]
}

# ============================================================================
# API KEYS & CREDENTIALS
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "educational-rag")

# Optional: For advanced features
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
SENTRY_DSN = os.getenv("SENTRY_DSN")

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension of embeddings
EMBEDDING_DEVICE = "cuda"  # or "cpu" if no GPU
MAX_EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding

# ============================================================================
# PINECONE CONFIGURATION
# ============================================================================
PINECONE_METRIC = "cosine"  # "cosine", "euclidean", or "dotproduct"
PINECONE_NAMESPACES = {
    "machine_learning": "ml",
    "deep_learning": "dl",
    "nlp": "nlp",
    "llm": "llm",
    "rag": "rag",
    "ai_basics": "ai"
}

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
TOP_K_RETRIEVAL = 5  # Number of top documents to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score
ENABLE_RERANKING = True  # Use cross-encoder for reranking
ENABLE_HYBRID_SEARCH = True  # Use both dense and sparse search

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
LLM_MODEL = "gpt-4"  # or "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.7  # 0 = deterministic, 1 = creative
LLM_MAX_TOKENS = 1000  # Max tokens in response
LLM_TOP_P = 0.9  # Nucleus sampling
LLM_TIMEOUT = 30  # Timeout in seconds

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================
ENABLE_CACHING = True
CACHE_TYPE = "redis"  # "redis", "memory", or "disk"
CACHE_HOST = os.getenv("REDIS_HOST", "localhost")
CACHE_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_DB = 0
CACHE_TTL = 3600  # Time-to-live in seconds (1 hour)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/educational_rag")
DB_ECHO = DEBUG = os.getenv("DEBUG", "False").lower() == "true"
DB_POOL_SIZE = 10
DB_POOL_RECYCLE = 3600

# ============================================================================
# WEB SCRAPING CONFIGURATION
# ============================================================================
SCRAPER_TIMEOUT = 10  # Timeout for HTTP requests
SCRAPER_RETRIES = 3  # Number of retries
SCRAPER_BACKOFF = 2  # Backoff factor for retries
SCRAPER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
SCRAPER_RATE_LIMIT = 1  # Delay between requests in seconds
MAX_DOCUMENTS_PER_TOPIC = 50  # Max docs to scrape per topic

# ============================================================================
# DATA QUALITY CONFIGURATION
# ============================================================================
MIN_DOCUMENT_LENGTH = 100  # Minimum characters for a document
MAX_DOCUMENT_LENGTH = 100000  # Maximum characters for a document
MIN_CHUNK_LENGTH = 50  # Minimum characters for a chunk
DUPLICATE_THRESHOLD = 0.95  # Similarity threshold for deduplication
LANGUAGE = "en"  # Expected language

# ============================================================================
# FEATURE FLAGS
# ============================================================================
ENABLE_HALLUCINATION_DETECTION = True
ENABLE_FACT_CHECKING = True
ENABLE_MULTI_TURN = True
ENABLE_FEEDBACK_COLLECTION = True
ENABLE_ANALYTICS = True
ENABLE_MONITORING = True

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 4))
API_TIMEOUT = 60  # Request timeout in seconds
RATE_LIMIT_REQUESTS = 100  # Requests per minute
RATE_LIMIT_WINDOW = 60  # Time window in seconds

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
STREAMLIT_THEME = "dark"  # "light" or "dark"

# ============================================================================
# MONITORING & ALERTS
# ============================================================================
ENABLE_SENTRY = bool(SENTRY_DSN)
SENTRY_TRACES_SAMPLE_RATE = 0.1
SENTRY_PROFILES_SAMPLE_RATE = 0.1
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
ALERT_THRESHOLD = 0.8  # Alert if error rate exceeds this

# ============================================================================
# COST TRACKING
# ============================================================================
TRACK_COSTS = True
OPENAI_COST_PER_1K_PROMPT = 0.015  # GPT-4 prompt cost
OPENAI_COST_PER_1K_COMPLETION = 0.045  # GPT-4 completion cost
PINECONE_COST_PER_QUERY = 0.000001  # Approximate

# ============================================================================
# VALIDATION
# ============================================================================
if not OPENAI_API_KEY:
    print("⚠️ Warning: OPENAI_API_KEY not set in .env")
if not PINECONE_API_KEY:
    print("⚠️ Warning: PINECONE_API_KEY not set in .env")

print("✅ Configuration loaded successfully!")
print(f"   Project Root: {PROJECT_ROOT}")
print(f"   Data Dir: {RAW_DATA_DIR}")
print(f"   Topics: {len(EDUCATIONAL_TOPICS)} categories with {sum(len(urls) for urls in EDUCATIONAL_TOPICS.values())} URLs")