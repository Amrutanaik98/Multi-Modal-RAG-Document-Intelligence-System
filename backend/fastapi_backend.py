# backend/fastapi_backend.py
# Complete RAG Backend with HuggingFace LLM Integration

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import time
import logging

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# HuggingFace imports
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    query: str
    top_k: int = 5
    llm_model: str = 'mistral'
    include_summary: bool = False
    max_answer_tokens: int = 500

class RetrievedChunk(BaseModel):
    """Model for retrieved document chunks"""
    chunk_id: str
    chunk_text: str
    document_id: str
    topic: str
    source: str
    url: str
    similarity_score: float

class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    query: str
    retrieved_chunks: List[RetrievedChunk]
    answer: str
    llm_model: str
    summary: Optional[str] = None
    avg_similarity: float
    timestamp: str
    response_time: float

# ============================================================================
# HUGGINGFACE LLM INTEGRATION
# ============================================================================

class HuggingFaceLLMIntegration:
    """Use HuggingFace hosted models for inference"""
    
    def __init__(self, hf_api_key: str = None):
        """Initialize HuggingFace client"""
        from huggingface_hub import InferenceClient
        
        self.hf_api_key = hf_api_key or os.getenv("HF_API_TOKEN")
        
        if not self.hf_api_key:
            logger.warning("⚠️  HF_API_TOKEN not found in environment variables")
            self.client = None
        else:
            try:
                self.client = InferenceClient(api_key=self.hf_api_key)
                logger.info("✅ HuggingFace client initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing HuggingFace: {e}")
                self.client = None
        
        self.model_options = {
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
            'zephyr': 'HuggingFaceH4/zephyr-7b-beta',
            'neural_chat': 'Intel/neural-chat-7b-v3-1',
            'llama2': 'meta-llama/Llama-2-7b-chat-hf',
            'phi': 'microsoft/phi-2'
        }
    
    def generate_answer_hf(self, 
                          query: str, 
                          context: str,
                          model: str = 'mistral',
                          max_tokens: int = 500) -> Dict:
        """Generate answer using HuggingFace model"""
        
        if not self.client:
            return {
                'status': 'error',
                'error': 'HuggingFace API not initialized',
                'answer': self._generate_fallback_answer(query, context),
                'model': 'fallback',
                'timestamp': datetime.now().isoformat()
            }
        
        model_id = self.model_options.get(model, self.model_options['mistral'])
        
        prompt = f"""You are an expert AI assistant. Use the provided context to answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        try:
            response = self.client.text_generation(
                prompt=prompt,
                model=model_id,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
            return {
                'status': 'success',
                'answer': response,
                'model': model_id,
                'tokens': max_tokens,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"HuggingFace error: {e}")
            return {
                'status': 'error_fallback',
                'error': str(e),
                'answer': self._generate_fallback_answer(query, context),
                'model': 'fallback',
                'timestamp': datetime.now().isoformat()
            }
    
    @staticmethod
    def _generate_fallback_answer(query: str, context: str) -> str:
        """Fallback answer generation if HuggingFace fails"""
        answer = f"""Based on the retrieved documents:\n\n{context[:500]}...\n\nThis information is relevant to your query: "{query}"

To get a more detailed answer:
1. Review the full documents
2. Ask follow-up questions
3. Try rephrasing your query"""
        return answer
    
    def summarize_documents(self, 
                           documents: List[str],
                           summary_length: str = 'medium') -> Dict:
        """Summarize multiple documents"""
        
        if not self.client:
            return {
                'status': 'error',
                'error': 'HuggingFace API not initialized',
                'summary': None
            }
        
        combined_text = "\n\n".join(documents[:3])
        
        length_instructions = {
            'short': 'in 2-3 sentences',
            'medium': 'in 1 paragraph',
            'long': 'in 2-3 paragraphs'
        }
        
        prompt = f"""Summarize the following text {length_instructions.get(summary_length, 'clearly')}:

{combined_text}

SUMMARY:"""
        
        try:
            response = self.client.text_generation(
                prompt=prompt,
                model='mistralai/Mistral-7B-Instruct-v0.2',
                max_new_tokens=300,
                temperature=0.5
            )
            
            return {
                'status': 'success',
                'summary': response,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'summary': None}

# ============================================================================
# PINECONE INTEGRATION
# ============================================================================

class PineconeVectorDB:
    """Manage embeddings in Pinecone"""
    
    def __init__(self, api_key: str = None):
        """Initialize Pinecone"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            self.ServerlessSpec = ServerlessSpec
            self.api_key = api_key or os.getenv("PINECONE_API_KEY")
            
            if self.api_key:
                self.pc = Pinecone(api_key=self.api_key)
                logger.info("✅ Pinecone client initialized")
            else:
                logger.warning("⚠️  PINECONE_API_KEY not found")
                self.pc = None
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            self.pc = None
        
        self.index_name = "rag-documents"
    
    def create_index(self, dimension: int = 384, metric: str = "cosine"):
        """Create Pinecone index"""
        
        if not self.pc:
            return {'status': 'error', 'message': 'Pinecone not initialized'}
        
        try:
            indexes = self.pc.list_indexes()
            if self.index_name in [idx.name for idx in indexes]:
                logger.info(f"Index '{self.index_name}' already exists")
                return {'status': 'exists'}
            
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=self.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"✅ Created index: {self.index_name}")
            return {'status': 'success', 'message': f'Index {self.index_name} created'}
        
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def upload_embeddings(self, embeddings_data: List[Dict]):
        """Upload embeddings to Pinecone"""
        
        if not self.pc:
            return {'status': 'error', 'message': 'Pinecone not initialized'}
        
        try:
            index = self.pc.Index(self.index_name)
            
            vectors_to_upsert = []
            for item in embeddings_data:
                embedding = item['embedding']
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                vector = {
                    'id': str(item['id']),
                    'values': embedding,
                    'metadata': {
                        'chunk_text': item.get('chunk_text', '')[:1000],
                        'document_id': item.get('document_id', ''),
                        'topic': item.get('topic', ''),
                        'source': item.get('source', ''),
                        'url': item.get('url', ''),
                        'timestamp': datetime.now().isoformat()
                    }
                }
                vectors_to_upsert.append(vector)
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                try:
                    index.upsert(vectors=batch)
                    logger.info(f"✅ Uploaded batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error uploading batch: {e}")
            
            return {'status': 'success', 'uploaded': len(vectors_to_upsert)}
        
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar vectors in Pinecone"""
        
        if not self.pc:
            return []
        
        try:
            index = self.pc.Index(self.index_name)
            
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return [{
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            } for match in results['matches']]
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# ============================================================================
# EMBEDDING LOADER
# ============================================================================

class EmbeddingLoader:
    """Load and manage embeddings"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_data = {}
        self.documents = {}
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load embeddings from processed data directory"""
        processed_dir = Path("data/processed")
        
        if not processed_dir.exists():
            logger.warning("Processed data directory not found. Using sample data.")
            self._load_sample_data()
            return
        
        # Load embeddings from JSON files
        embedding_files = list(processed_dir.glob("embeddings_*.json"))
        
        for file in embedding_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for embedding_obj in data.get('embeddings', []):
                        doc_id = embedding_obj['id']
                        self.embeddings_data[doc_id] = embedding_obj
                logger.info(f"Loaded {len(data.get('embeddings', []))} embeddings from {file.name}")
            except Exception as e:
                logger.error(f"Error loading {file.name}: {e}")
        
        # Load chunks
        chunk_files = list(processed_dir.glob("chunks_*.json"))
        
        for file in chunk_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for chunk in data.get('chunks', []):
                        chunk_id = chunk['id']
                        self.documents[chunk_id] = chunk
                logger.info(f"Loaded {len(data.get('chunks', []))} chunks from {file.name}")
            except Exception as e:
                logger.error(f"Error loading {file.name}: {e}")
    
    def _load_sample_data(self):
        """Load sample data if no processed data exists"""
        sample_docs = [
            {
                'id': 'chunk_001',
                'title': 'Machine Learning Basics',
                'content': 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It uses algorithms to analyze data and make predictions based on patterns.',
                'topic': 'machine_learning',
                'source': 'educational',
                'url': 'https://example.com/ml-basics'
            },
            {
                'id': 'chunk_002',
                'title': 'Deep Learning',
                'content': 'Deep learning uses neural networks with multiple layers. It powers modern AI applications like image recognition and natural language processing.',
                'topic': 'deep_learning',
                'source': 'research',
                'url': 'https://example.com/dl'
            },
            {
                'id': 'chunk_003',
                'title': 'Natural Language Processing',
                'content': 'NLP is a field focused on interaction between computers and human language. It enables machines to understand and generate language using techniques like tokenization and parsing.',
                'topic': 'nlp',
                'source': 'tutorial',
                'url': 'https://example.com/nlp'
            },
            {
                'id': 'chunk_004',
                'title': 'Transformers Architecture',
                'content': 'Transformers use attention mechanisms to process sequential data efficiently. They form the basis of modern LLMs like BERT and GPT.',
                'topic': 'transformers',
                'source': 'research',
                'url': 'https://example.com/transformers'
            },
            {
                'id': 'chunk_005',
                'title': 'Vector Embeddings',
                'content': 'Embeddings are numerical representations of text that capture semantic meaning. They allow similar texts to have similar vectors in high-dimensional space.',
                'topic': 'embeddings',
                'source': 'technical',
                'url': 'https://example.com/embeddings'
            },
            {
                'id': 'chunk_006',
                'title': 'RAG Systems',
                'content': 'Retrieval-Augmented Generation combines information retrieval with generative models for more accurate answers. It retrieves relevant documents then generates answers based on context.',
                'topic': 'rag',
                'source': 'educational',
                'url': 'https://example.com/rag'
            }
        ]
        
        # Generate embeddings for sample data
        for doc in sample_docs:
            embedding = self.model.encode(doc['content']).tolist()
            self.embeddings_data[doc['id']] = {
                'id': doc['id'],
                'embedding': embedding,
                'content': doc['content'],
                'title': doc['title'],
                'topic': doc['topic'],
                'source': doc['source'],
                'url': doc['url']
            }
            self.documents[doc['id']] = doc
        
        logger.info(f"Loaded {len(sample_docs)} sample documents")
    
    def get_all_embeddings(self):
        """Get all embeddings as numpy array"""
        if not self.embeddings_data:
            return np.array([])
        
        embeddings = []
        for doc_id in sorted(self.embeddings_data.keys()):
            embedding = self.embeddings_data[doc_id].get('embedding', [])
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_document(self, doc_id: str):
        """Get document content by ID"""
        return self.documents.get(doc_id, self.embeddings_data.get(doc_id, {}))

# ============================================================================
# SIMILARITY SEARCH
# ============================================================================

class SimilaritySearchEngine:
    """Advanced similarity search"""
    
    def __init__(self, embedding_loader: EmbeddingLoader):
        self.loader = embedding_loader
        self.embeddings_array = embedding_loader.get_all_embeddings()
        self.doc_ids = sorted(embedding_loader.embeddings_data.keys())
    
    def calculate_cosine_similarity(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Calculate cosine similarity"""
        
        if len(self.embeddings_array) == 0:
            return []
        
        # Normalize vectors
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        embeddings_norm = self.embeddings_array / (np.linalg.norm(self.embeddings_array, axis=1, keepdims=True) + 1e-8)
        
        # Calculate similarities
        similarities = np.dot(embeddings_norm, query_embedding)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                doc = self.loader.get_document(doc_id)
                
                results.append({
                    'chunk_id': doc_id,
                    'chunk_text': doc.get('content', ''),
                    'document_id': doc.get('title', 'Unknown'),
                    'topic': doc.get('topic', 'unknown'),
                    'source': doc.get('source', 'unknown'),
                    'url': doc.get('url', ''),
                    'similarity_score': float(similarities[idx])
                })
        
        return results
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Hybrid search combining semantic and keyword"""
        
        # Semantic search
        semantic_results = self.calculate_cosine_similarity(query_embedding, top_k)
        
        # Keyword search
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores = {}
        
        for doc_id, doc in self.loader.documents.items():
            content = doc.get('content', '').lower()
            title = doc.get('title', '').lower()
            
            score = 0
            for word in query_words:
                if len(word) > 2:
                    score += content.count(word)
                    score += title.count(word) * 2
            
            scores[doc_id] = score
        
        # Combine results
        combined = {}
        
        for i, result in enumerate(semantic_results):
            doc_id = result['chunk_id']
            score = (1 - i / top_k) * 0.6
            combined[doc_id] = combined.get(doc_id, 0) + score
        
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for i, (doc_id, _) in enumerate(top_docs):
            score = (1 - i / top_k) * 0.4
            combined[doc_id] = combined.get(doc_id, 0) + score
        
        # Get top k
        top_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, combined_score in top_docs:
            doc = self.loader.get_document(doc_id)
            results.append({
                'chunk_id': doc_id,
                'chunk_text': doc.get('content', ''),
                'document_id': doc.get('title', 'Unknown'),
                'topic': doc.get('topic', 'unknown'),
                'source': doc.get('source', 'unknown'),
                'url': doc.get('url', ''),
                'similarity_score': float(combined_score)
            })
        
        return results

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="RAG Query System v4.0",
    description="Retrieval-Augmented Generation with HuggingFace LLMs",
    version="4.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger.info("Initializing RAG system...")
embedding_loader = EmbeddingLoader()
search_engine = SimilaritySearchEngine(embedding_loader)
llm = HuggingFaceLLMIntegration()
pinecone_db = PineconeVectorDB()

logger.info("RAG system initialized successfully")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Query API v4.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "documents_loaded": len(embedding_loader.embeddings_data),
        "embedding_dimension": 384,
        "llm_available": llm.client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Main RAG query endpoint"""
    
    start_time = time.time()
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if request.top_k < 1 or request.top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
    
    # Encode query
    query_embedding = embedding_loader.model.encode(request.query)
    
    # Retrieve documents
    chunks = search_engine.hybrid_search(request.query, query_embedding, request.top_k)
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")
    
    # Format context
    context = "\n\n".join([
        f"[{c['document_id']}]\n{c['chunk_text']}" 
        for c in chunks
    ])
    
    # Generate answer
    hf_response = llm.generate_answer_hf(
        query=request.query,
        context=context,
        model=request.llm_model,
        max_tokens=request.max_answer_tokens
    )
    
    # Optional: Generate summary
    summary = None
    if request.include_summary:
        summary_response = llm.summarize_documents(
            [c['chunk_text'] for c in chunks],
            summary_length='medium'
        )
        summary = summary_response.get('summary')
    
    # Calculate metrics
    response_time = time.time() - start_time
    avg_similarity = np.mean([c['similarity_score'] for c in chunks])
    
    # Convert to response model
    chunk_responses = [RetrievedChunk(**c) for c in chunks]
    
    return QueryResponse(
        query=request.query,
        retrieved_chunks=chunk_responses,
        answer=hf_response['answer'],
        llm_model=hf_response.get('model', 'unknown'),
        summary=summary,
        avg_similarity=float(avg_similarity),
        timestamp=datetime.now().isoformat(),
        response_time=response_time
    )

@app.post("/query/simple")
async def query_simple(query: str, top_k: int = 5):
    """Simple query endpoint"""
    
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    query_embedding = embedding_loader.model.encode(query)
    chunks = search_engine.hybrid_search(query, query_embedding, top_k)
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No documents found")
    
    context = "\n\n".join([f"[{c['document_id']}]\n{c['chunk_text']}" for c in chunks])
    
    hf_response = llm.generate_answer_hf(
        query=query,
        context=context,
        model='mistral'
    )
    
    return {
        'query': query,
        'retrieved_count': len(chunks),
        'answer': hf_response['answer'],
        'chunks': chunks,
        'timestamp': datetime.now().isoformat()
    }

@app.get("/documents")
async def list_documents():
    """List all documents"""
    docs = []
    for doc_id, doc in embedding_loader.documents.items():
        docs.append({
            'id': doc_id,
            'title': doc.get('title', 'Unknown'),
            'topic': doc.get('topic', 'unknown'),
            'source': doc.get('source', 'unknown'),
            'url': doc.get('url', '')
        })
    
    return {
        'total_documents': len(docs),
        'documents': docs
    }

@app.get("/models")
async def list_models():
    """List available LLM models"""
    return {
        'available_models': list(llm.model_options.keys()),
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dimension': 384,
        'descriptions': {
            'mistral': 'Mistral 7B Instruct (Fast & Accurate)',
            'zephyr': 'Zephyr 7B (Optimized for Chat)',
            'neural_chat': 'Intel Neural Chat 7B',
            'llama2': 'Llama 2 7B Chat',
            'phi': 'Microsoft Phi 2 (Lightweight)'
        }
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 RAG BACKEND WITH HUGGINGFACE LLMs")
    print("=" * 80)
    print(f"\n📚 Documents loaded: {len(embedding_loader.embeddings_data)}")
    print(f"🤖 LLM models available: {len(llm.model_options)}")
    print("\n📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\n" + "=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )