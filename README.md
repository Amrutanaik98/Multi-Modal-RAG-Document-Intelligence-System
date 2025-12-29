# 🧠 RAG Intelligence System v5.0

A complete **Retrieval-Augmented Generation (RAG)** pipeline for building AI-powered document search and question-answering systems. This project combines semantic search, vector embeddings, and large language models to provide intelligent answers grounded in your document collection.

## 🎯 Overview

The RAG Intelligence System is an end-to-end solution that:
- **Scrapes** documents from multiple sources (Wikipedia, arXiv, HuggingFace)
- **Processes** and chunks text into optimal segments
- **Embeds** chunks using Sentence Transformers
- **Indexes** embeddings in Pinecone for fast retrieval
- **Generates** context-aware answers using HuggingFace LLMs
- **Serves** queries via FastAPI REST API
- **Visualizes** results through Streamlit UI

## 📁 Project Structure

```
rag-system/
├── backend/
│   ├── fastapi_backend.py              # Main FastAPI server
│   ├── huggingface_integration_llm.py  # LLM integration (Mistral, Zephyr, Phi, etc.)
│   ├── pinecone_upload.py              # Pinecone vector database management
│   ├── embedding_to_pinecone.py        # Embedding pipeline with Pinecone upload
│   └── main_orchestrator.py            # Databricks notebook - complete pipeline
│
├── scripts/
│   ├── embedding_pipeline.py           # Text → Embeddings conversion
│   ├── embedding_quality.py            # Embedding validation & analysis
│   ├── metadata_extractor.py           # Metadata extraction (keywords, topics, difficulty)
│   ├── text_chunker.py                 # Text splitting into chunks
│   └── text_cleaner.py                 # Text preprocessing & normalization
│
├── frontend/
│   └── streamlit_app.py                # Interactive web UI
│
├── rag_query_interface.py              # Query execution and RAG logic
└── README.md
```

## 🚀 Features

### Phase 1: Data Preparation
- **Text Cleaning**: Remove URLs, emails, special characters
- **Text Chunking**: Split documents with configurable size and overlap
- **Metadata Extraction**: Auto-detect topics, difficulty level, content type, keywords
- **Quality Validation**: Outlier detection, diversity metrics, embedding analysis

### Phase 2: Embedding & Indexing
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2 - 384 dimensions)
- **Vector Storage**: Pinecone serverless indexes for scalable retrieval
- **Batch Processing**: Efficient batch uploads with progress tracking
- **Metadata Storage**: Rich metadata for filtering and context

### Phase 3: Query & Generation
- **Semantic Search**: Cosine similarity + hybrid search (semantic + keyword)
- **LLM Integration**: Multiple HuggingFace models
  - Mistral 7B (fast & accurate)
  - Zephyr 7B (optimized for chat)
  - Phi 2 (lightweight)
  - Llama 2 7B
  - Intel Neural Chat 7B
- **Context-Aware Generation**: LLM generates answers based on retrieved documents
- **Summarization & Paraphrasing**: Additional LLM capabilities

### Phase 4: Serving & Visualization
- **FastAPI Backend**: Production-ready REST API
- **Streamlit Frontend**: Interactive UI with results visualization
- **Health Monitoring**: System status and performance metrics
- **Response Statistics**: Retrieval scores, timing, document attribution

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip or conda
- API Keys (optional but recommended):
  - HuggingFace API Token: https://huggingface.co/settings/tokens
  - Pinecone API Key: https://www.pinecone.io/

### Step 1: Clone & Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root:

```env
# HuggingFace LLM API
HF_API_TOKEN=your_huggingface_api_key

# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_api_key

# Optional: Data sources
WIKIPEDIA_ENABLED=true
ARXIV_ENABLED=true
HUGGINGFACE_ENABLED=true
```

### Step 3: Run the Complete Pipeline

```bash
# Data scraping + processing + embedding (Databricks notebook)
python backend/main_orchestrator.py

# Or run individual steps:
python scripts/text_cleaner.py
python scripts/text_chunker.py
python scripts/embedding_pipeline.py
python scripts/embedding_quality.py
python scripts/metadata_extractor.py
```

### Step 4: Upload to Pinecone

```bash
python backend/embedding_to_pinecone.py
```

### Step 5: Start the Backend API

```bash
python backend/fastapi_backend.py
# API available at: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Step 6: Launch the Frontend

```bash
streamlit run frontend/streamlit_app.py
# UI available at: http://localhost:8501
```

## 📚 File Descriptions

### Backend Files

**`fastapi_backend.py`** (450+ lines)
- REST API server with multiple endpoints
- Embedding loading & management
- Similarity search (semantic + hybrid)
- LLM answer generation
- Response formatting

**`huggingface_integration_llm.py`** (350+ lines)
- HuggingFace Inference API integration
- Multiple LLM model support
- Summarization & paraphrasing
- Keyword extraction
- Fallback answer generation

**`pinecone_upload.py`** (400+ lines)
- Pinecone serverless index management
- Batch vector uploads
- Semantic search queries
- Index statistics & monitoring
- Bulk operations (delete, describe)

**`embedding_to_pinecone.py`** (300+ lines)
- Complete embedding pipeline
- Local file saving + Pinecone upload
- Embedding quality reporting
- Batch processing

**`main_orchestrator.py`** (Databricks Notebook - 500+ lines)
- End-to-end pipeline orchestration
- Document scraping from 3 sources
- Text processing & chunking
- Embedding generation
- Error handling & logging

### Scripts

**`embedding_pipeline.py`** - Convert chunks to embeddings
**`embedding_quality.py`** - Validate embedding quality (outlier detection, diversity)
**`metadata_extractor.py`** - Extract keywords, topics, difficulty levels
**`text_chunker.py`** - Split text with overlap
**`text_cleaner.py`** - Normalize & clean text

### Frontend

**`streamlit_app.py`** (500+ lines)
- Interactive query interface
- Beautiful gradient UI
- Document retrieval visualization
- Performance metrics display
- Example questions
- Health monitoring

**`rag_query_interface.py`** (Databricks Notebook)
- Query execution
- Result formatting
- Basic answer generation

## 🔌 API Endpoints

### Health & Info
```bash
GET /health                 # System status
GET /                       # Root endpoint
GET /documents             # List all documents
GET /models                # Available LLM models
```

### Query Endpoints
```bash
POST /query                # Full RAG query with embeddings
POST /query/simple         # Simple query endpoint
```

**Request Example:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5,
  "llm_model": "mistral",
  "include_summary": false,
  "max_answer_tokens": 500
}
```

**Response Example:**
```json
{
  "query": "What is machine learning?",
  "retrieved_chunks": [...],
  "answer": "Machine learning is...",
  "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
  "avg_similarity": 0.72,
  "response_time": 2.34,
  "timestamp": "2025-01-15T10:30:00"
}
```

## 🎨 UI Features

The Streamlit frontend includes:
- **Dark/Light Mode**: Responsive design
- **Search Settings**: Configurable retrieval & model parameters
- **Real-time Results**: Streaming answer generation
- **Document Visualization**: Color-coded relevance scores
- **System Metrics**: Response time, similarity scores
- **Example Queries**: Quick-start templates

## 📊 Data Pipeline

```
Raw Documents (Wikipedia, arXiv, HuggingFace)
        ↓
    Cleaning (remove URLs, special chars)
        ↓
   Text Chunking (512-char chunks, 50-char overlap)
        ↓
Embedding Generation (Sentence Transformers)
        ↓
Quality Analysis (outliers, diversity, norms)
        ↓
Metadata Extraction (keywords, topics, difficulty)
        ↓
Pinecone Upload (serverless indexing)
        ↓
Query Interface (FastAPI + Streamlit)
```

## ⚙️ Configuration

Key settings in `config.py`:
- `CHUNK_SIZE`: 512 characters
- `CHUNK_OVERLAP`: 50 characters
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `MAX_EMBEDDING_BATCH_SIZE`: 32
- `PROCESSED_DATA_DIR`: data/processed/
- `OUTPUT_DIR`: output/

## 🧪 Testing

```bash
# Test HuggingFace connection
python backend/huggingface_integration_llm.py

# Test Pinecone connection
python backend/pinecone_upload.py

# Test embedding quality
python scripts/embedding_quality.py

# Run example queries
python rag_query_interface.py
```

## 📈 Performance

- **Embedding Generation**: ~100 docs/min (batch processing)
- **Vector Search**: <50ms (Pinecone)
- **LLM Response**: 2-5s (HuggingFace API)
- **Total Response Time**: 3-7s per query

## 🔒 Security

- API keys stored in `.env` (not in version control)
- Input validation on all endpoints
- Rate limiting available
- Metadata size limits (prevent abuse)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| HuggingFace API key error | Check `.env` file, regenerate token at huggingface.co |
| Pinecone connection failed | Verify API key, check region setting |
| No documents found | Run data scraping pipeline first |
| Backend won't start | Check port 8000 is free, check dependencies |
| Slow queries | Increase `top_k`, optimize Pinecone index |

## 📝 Example Queries

```
"What is machine learning?"
"Explain transformers in NLP"
"How do neural networks work?"
"What is retrieval-augmented generation?"
"Compare deep learning and traditional ML"
```

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time document updates
- [ ] User authentication & multi-tenancy
- [ ] Query caching & analytics
- [ ] Fine-tuning on custom datasets
- [ ] Advanced filtering & metadata search
- [ ] Feedback loop for answer quality

## 📄 License

MIT License - Free for personal and commercial use

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests.

## 📞 Support

- Documentation: `/docs` endpoint
- API Docs: `http://localhost:8000/docs`
- Issues: GitHub Issues

---

**Made with ❤️ for AI & RAG systems** | Version 5.0.0 | 2025
