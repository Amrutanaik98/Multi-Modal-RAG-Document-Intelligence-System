"""
Create Pinecone Index
Create the vector database index before uploading embeddings
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX, PINECONE_ENV

def create_pinecone_index():
    """Create Pinecone index"""
    
    print("\n" + "="*70)
    print("CREATING PINECONE INDEX")
    print("="*70 + "\n")
    
    # Initialize Pinecone
    print("[CONNECTING] Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("[OK] Connected to Pinecone\n")
    
    index_name = PINECONE_INDEX
    
    # Check if index already exists
    print(f"[CHECKING] Checking if index '{index_name}' exists...")
    existing_indexes = pc.list_indexes()
    
    index_exists = False
    for idx in existing_indexes.indexes:
        if idx.name == index_name:
            index_exists = True
            print(f"[OK] Index '{index_name}' already exists!")
            break
    
    if not index_exists:
        print(f"[CREATING] Index '{index_name}' doesn't exist. Creating it now...\n")
        
        # Create index
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension of sentence-transformers embeddings
            metric="cosine",  # Similarity metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        print(f"[OK] Index '{index_name}' created successfully!")
        print("[WAITING] Waiting for index to be ready (this may take a minute)...\n")
        
        # Wait for index to be ready
        import time
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(2)
            
            try:
                index = pc.Index(index_name)
                stats = index.describe_index_stats()
                print(f"[READY] Index is ready! Current vectors: {stats.total_vector_count}")
                break
            except:
                attempt += 1
                print(f"[WAITING] Still waiting... (attempt {attempt}/{max_attempts})")
    
    # Get index info
    print(f"\n[INFO] Getting index information...")
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    
    print(f"\nIndex Details:")
    print(f"  Name: {index_name}")
    print(f"  Dimension: 384")
    print(f"  Metric: cosine")
    print(f"  Total Vectors: {stats.total_vector_count}")
    print(f"  Namespaces: {list(stats.namespaces.keys()) if stats.namespaces else 'None yet'}")
    
    print("\n" + "="*70)
    print("INDEX READY FOR UPLOAD!")
    print("="*70 + "\n")
    
    return index_name


if __name__ == "__main__":
    try:
        create_pinecone_index()
    except Exception as e:
        print(f"\n[ERROR] Failed to create index: {e}")
        raise