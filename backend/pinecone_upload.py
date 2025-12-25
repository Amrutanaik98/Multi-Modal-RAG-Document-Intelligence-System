# backend/pinecone_integration.py
# Pinecone Vector Database Integration for Scalable Vector Storage

import os
from pinecone import Pinecone, ServerlessSpec
import json
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PINECONE VECTOR DATABASE INTEGRATION
# ============================================================================

class PineconeVectorDB:
    """
    Manage embeddings in Pinecone vector database
    
    Features:
    - Serverless index creation
    - Batch vector uploads
    - Semantic search
    - Metadata filtering
    - Bulk operations
    
    Get API key from: https://www.pinecone.io/
    """
    
    def __init__(self, api_key: str = None, environment: str = "us-east-1"):
        """
        Initialize Pinecone client
        
        Args:
            api_key: Pinecone API key (or from PINECONE_API_KEY env var)
            environment: Region for serverless index (us-east-1 recommended)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment
        self.index_name = "educational-rag"
        
        if not self.api_key:
            logger.warning("⚠️  PINECONE_API_KEY not found in environment variables")
            logger.warning("Visit: https://www.pinecone.io/")
            self.pc = None
            self.is_available = False
        else:
            try:
                self.pc = Pinecone(api_key=self.api_key)
                self.is_available = True
                logger.info("✅ Pinecone client initialized successfully")
                logger.info(f"Environment: {environment}")
            except Exception as e:
                logger.error(f"❌ Error initializing Pinecone: {e}")
                self.pc = None
                self.is_available = False
    
    def create_index(self, 
                    dimension: int = 384, 
                    metric: str = "cosine",
                    index_name: str = None) -> Dict:
        """
        Create Pinecone serverless index
        
        Args:
            dimension: Embedding dimension (384 for MiniLM, 768 for BERT)
            metric: Distance metric ('cosine', 'euclidean', 'dotproduct')
            index_name: Custom index name
        
        Returns:
            Dict with creation status
        """
        
        if not self.pc:
            return {
                'status': 'error',
                'message': 'Pinecone client not initialized',
                'index_name': index_name or self.index_name
            }
        
        index_name = index_name or self.index_name
        
        try:
            # Check if index already exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if index_name in existing_indexes:
                logger.info(f"ℹ️  Index '{index_name}' already exists")
                return {
                    'status': 'exists',
                    'message': f'Index {index_name} already exists',
                    'index_name': index_name
                }
            
            # Create serverless index
            logger.info(f"Creating index: {index_name}")
            
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment
                )
            )
            
            logger.info(f"✅ Created index: {index_name}")
            
            return {
                'status': 'success',
                'message': f'Index {index_name} created successfully',
                'index_name': index_name,
                'dimension': dimension,
                'metric': metric,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'index_name': index_name
            }
    
    def upload_embeddings(self, 
                         embeddings_data: List[Dict],
                         batch_size: int = 100) -> Dict:
        """
        Upload embeddings to Pinecone in batches
        
        Args:
            embeddings_data: List of dicts with:
                - id: unique identifier
                - embedding: vector (list or numpy array)
                - text: document text (for metadata)
                - topic: document topic
                - title: document title (optional)
                - source: source URL (optional)
            batch_size: Number of vectors per batch
        
        Returns:
            Dict with upload statistics
        """
        
        if not self.pc:
            return {
                'status': 'error',
                'message': 'Pinecone client not initialized',
                'uploaded': 0
            }
        
        try:
            index = self.pc.Index(self.index_name)
            
            # Prepare vectors
            vectors_to_upsert = []
            
            for item in embeddings_data:
                # Handle numpy arrays
                embedding = item['embedding']
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                elif not isinstance(embedding, list):
                    embedding = list(embedding)
                
                vector = {
                    'id': str(item['id']),
                    'values': embedding,
                    'metadata': {
                        'text': str(item.get('text', ''))[:2000],  # Limit text size
                        'topic': str(item.get('topic', 'unknown')),
                        'title': str(item.get('title', 'Unknown'))[:200],
                        'source': str(item.get('source', 'unknown'))[:500],
                        'timestamp': datetime.now().isoformat()
                    }
                }
                vectors_to_upsert.append(vector)
            
            # Upload in batches
            total_uploaded = 0
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                
                try:
                    index.upsert(vectors=batch)
                    total_uploaded += len(batch)
                    batch_num = i // batch_size + 1
                    logger.info(f"✅ Uploaded batch {batch_num} ({len(batch)} vectors)")
                except Exception as e:
                    logger.error(f"❌ Error uploading batch {batch_num}: {e}")
            
            logger.info(f"✅ Total uploaded: {total_uploaded} embeddings")
            
            return {
                'status': 'success',
                'message': f'Uploaded {total_uploaded} embeddings',
                'uploaded': total_uploaded,
                'batches': (len(vectors_to_upsert) + batch_size - 1) // batch_size,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'uploaded': 0
            }
    
    def search_similar(self, 
                      query_embedding: List[float], 
                      top_k: int = 5,
                      filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors in Pinecone
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter (e.g., {'topic': 'nlp'})
        
        Returns:
            List of similar documents with scores
        """
        
        if not self.pc:
            logger.warning("Pinecone not available, returning empty results")
            return []
        
        try:
            index = self.pc.Index(self.index_name)
            
            # Perform search
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            retrieved = []
            for match in results.get('matches', []):
                retrieved.append({
                    'id': match['id'],
                    'score': float(match['score']),
                    'metadata': match.get('metadata', {})
                })
            
            logger.info(f"✅ Retrieved {len(retrieved)} similar vectors")
            return retrieved
        
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def delete_index(self, index_name: str = None) -> Dict:
        """
        Delete a Pinecone index
        
        Args:
            index_name: Name of index to delete
        
        Returns:
            Dict with deletion status
        """
        
        if not self.pc:
            return {'status': 'error', 'message': 'Pinecone client not initialized'}
        
        index_name = index_name or self.index_name
        
        try:
            self.pc.delete_index(index_name)
            logger.info(f"✅ Deleted index: {index_name}")
            return {
                'status': 'success',
                'message': f'Deleted index {index_name}',
                'index_name': index_name
            }
        except Exception as e:
            logger.error(f"❌ Error deleting index: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'index_name': index_name
            }
    
    def describe_index(self, index_name: str = None) -> Dict:
        """
        Get information about a Pinecone index
        
        Args:
            index_name: Name of index to describe
        
        Returns:
            Dict with index information
        """
        
        if not self.pc:
            return {'status': 'error', 'message': 'Pinecone client not initialized'}
        
        index_name = index_name or self.index_name
        
        try:
            index_desc = self.pc.describe_index(index_name)
            return {
                'status': 'success',
                'index_name': index_name,
                'dimension': index_desc.get('dimension', 'unknown'),
                'metric': index_desc.get('metric', 'unknown'),
                'host': index_desc.get('host', 'unknown'),
                'status': index_desc.get('status', 'unknown')
            }
        except Exception as e:
            logger.error(f"❌ Error describing index: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def list_indexes(self) -> Dict:
        """List all Pinecone indexes"""
        
        if not self.pc:
            return {'status': 'error', 'indexes': []}
        
        try:
            indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            logger.info(f"✅ Found {len(index_names)} indexes")
            return {
                'status': 'success',
                'indexes': index_names,
                'count': len(index_names)
            }
        except Exception as e:
            logger.error(f"❌ Error listing indexes: {e}")
            return {'status': 'error', 'indexes': []}
    
    def get_index_stats(self, index_name: str = None) -> Dict:
        """
        Get statistics about an index
        
        Args:
            index_name: Name of index
        
        Returns:
            Dict with index statistics
        """
        
        if not self.pc:
            return {'status': 'error', 'message': 'Pinecone client not initialized'}
        
        index_name = index_name or self.index_name
        
        try:
            index = self.pc.Index(index_name)
            stats = index.describe_index_stats()
            
            return {
                'status': 'success',
                'index_name': index_name,
                'vector_count': stats.get('total_vector_count', 0),
                'namespaces': stats.get('namespaces', {}),
                'dimension': stats.get('dimension', 'unknown')
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def delete_vectors(self, vector_ids: List[str], index_name: str = None) -> Dict:
        """
        Delete specific vectors from index
        
        Args:
            vector_ids: List of vector IDs to delete
            index_name: Name of index
        
        Returns:
            Dict with deletion status
        """
        
        if not self.pc:
            return {'status': 'error', 'message': 'Pinecone client not initialized'}
        
        index_name = index_name or self.index_name
        
        try:
            index = self.pc.Index(index_name)
            index.delete(ids=vector_ids)
            logger.info(f"✅ Deleted {len(vector_ids)} vectors")
            return {
                'status': 'success',
                'deleted_count': len(vector_ids),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error deleting vectors: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def bulk_upsert_from_json(self, json_file_path: str) -> Dict:
        """
        Upload embeddings from JSON file to Pinecone
        
        Expected JSON format:
        {
            "embeddings": [
                {
                    "id": "doc_001",
                    "embedding": [...],
                    "text": "...",
                    "topic": "...",
                    "title": "..."
                },
                ...
            ]
        }
        
        Args:
            json_file_path: Path to JSON file
        
        Returns:
            Dict with upload status
        """
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            embeddings_list = data.get('embeddings', [])
            logger.info(f"Loading {len(embeddings_list)} embeddings from {json_file_path}")
            
            result = self.upload_embeddings(embeddings_list)
            return result
        
        except Exception as e:
            logger.error(f"❌ Error loading from JSON: {e}")
            return {'status': 'error', 'message': str(e), 'uploaded': 0}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_pinecone_connection(api_key: str = None) -> Dict:
    """
    Test Pinecone API connection
    
    Args:
        api_key: Optional API key to test
    
    Returns:
        Connection status and available indexes
    """
    
    db = PineconeVectorDB(api_key)
    
    if not db.is_available:
        return {
            'status': 'error',
            'message': 'Pinecone API not available',
            'indexes': []
        }
    
    try:
        indexes = db.list_indexes()
        return {
            'status': 'success',
            'message': 'Connected to Pinecone successfully',
            'indexes': indexes['indexes'],
            'index_count': indexes['count']
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Connection test failed: {e}',
            'indexes': []
        }

# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 TESTING PINECONE INTEGRATION")
    print("=" * 80 + "\n")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test connection
    result = test_pinecone_connection()
    
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    
    if result['status'] == 'success':
        print(f"\n✅ Connection successful!")
        print(f"Available indexes: {result['indexes']}")
    else:
        print(f"\n❌ Connection failed!")
    
    print("\n" + "=" * 80 + "\n")