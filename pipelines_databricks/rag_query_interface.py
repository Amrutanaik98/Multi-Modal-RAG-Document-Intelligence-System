# Databricks notebook source
# Databricks notebook source

# MAGIC %md
# MAGIC # RAG Query Interface - Ask Questions About Your Documents
# MAGIC Use embeddings to retrieve relevant documents and generate answers

# MAGIC %md
# MAGIC ## Stage 0: Install Dependencies

%pip install sentence-transformers openai langchain

print("✅ Dependencies installed!")

# MAGIC %md
# MAGIC ## Import Libraries

from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import os

print("=" * 80)
print("🤖 RAG QUERY SYSTEM INITIALIZED")
print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# MAGIC %md
# MAGIC ## Load Embeddings & Documents

print("\n📥 Loading embeddings and documents...")

# Read embeddings
embeddings_df = spark.table("chunk_embeddings").toPandas()
print(f"✅ Loaded {len(embeddings_df)} embeddings")

# Read original chunks with text
chunks_df = spark.table("processed_chunks").toPandas()
print(f"✅ Loaded {len(chunks_df)} chunks")

# Merge to get text with embeddings
merged_df = embeddings_df.merge(chunks_df[['chunk_id', 'chunk_text', 'document_id']], on='chunk_id', how='left')
print(f"✅ Merged data: {len(merged_df)} records")

# Convert embeddings list to numpy array
embeddings_array = np.array([np.array(emb) for emb in merged_df['embedding']])
print(f"✅ Embeddings shape: {embeddings_array.shape}")

# MAGIC %md
# MAGIC ## Load the Embedding Model

print("\n🧠 Loading embedding model...")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded successfully")

# MAGIC %md
# MAGIC ## RAG Functions

def retrieve_similar_chunks(query: str, top_k: int = 5) -> list:
    """
    Find the most similar chunks to a query
    
    Args:
        query: User's question
        top_k: Number of top results to return
    
    Returns:
        List of similar chunks
    """
    try:
        # Encode the query
        query_embedding = model.encode(query, convert_to_tensor=False)
        
        # Calculate similarity scores (cosine similarity)
        similarities = np.dot(embeddings_array, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Get the chunks
        results = []
        for idx in top_indices:
            if idx < len(merged_df):
                row = merged_df.iloc[idx]
                results.append({
                    'chunk_id': row['chunk_id'],
                    'chunk_text': row['chunk_text'],
                    'similarity_score': float(similarities[idx]),
                    'document_id': row['document_id']
                })
        
        return results
    
    except Exception as e:
        print(f"❌ Error retrieving chunks: {e}")
        return []

def format_context(chunks: list) -> str:
    """Format retrieved chunks into context for LLM"""
    context = "RELEVANT INFORMATION:\n"
    context += "=" * 80 + "\n"
    
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}] (Similarity: {chunk['similarity_score']:.3f})\n"
        context += f"{chunk['chunk_text'][:500]}...\n"
    
    context += "=" * 80 + "\n"
    return context

def generate_answer_simple(query: str, context: str) -> str:
    """
    Generate answer based on retrieved context
    Using simple template-based approach (no API needed)
    """
    try:
        answer = f"""
Based on the provided information:

{context}

ANSWER TO: "{query}"

The documents contain information related to your query. The retrieved passages above 
show relevant content that addresses different aspects of your question. To get a 
more specific answer, you can:

1. Ask follow-up questions about specific topics
2. Request information about specific technologies or concepts
3. Ask for comparisons or relationships between concepts

Retrieved {len(context.split('[Source'))-1} relevant passages for your reference.
"""
        return answer
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return "Unable to generate answer."

def rag_query(query: str, top_k: int = 5) -> dict:
    """
    Complete RAG pipeline: Retrieve + Generate
    
    Args:
        query: User's question
        top_k: Number of chunks to retrieve
    
    Returns:
        Dictionary with query, retrieved chunks, and generated answer
    """
    print("\n" + "=" * 80)
    print(f"🔍 QUERY: {query}")
    print("=" * 80)
    
    # Step 1: Retrieve
    print(f"\n📚 Retrieving top {top_k} relevant chunks...")
    chunks = retrieve_similar_chunks(query, top_k=top_k)
    
    if not chunks:
        print("❌ No relevant chunks found")
        return {
            'query': query,
            'retrieved_chunks': [],
            'answer': 'No relevant information found in documents.'
        }
    
    print(f"✅ Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   {i}. Score: {chunk['similarity_score']:.3f} - {chunk['chunk_text'][:100]}...")
    
    # Step 2: Generate context
    context = format_context(chunks)
    
    # Step 3: Generate answer
    print("\n🤖 Generating answer...")
    answer = generate_answer_simple(query, context)
    
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(answer)
    
    return {
        'query': query,
        'retrieved_chunks': chunks,
        'answer': answer,
        'timestamp': datetime.now().isoformat()
    }

# MAGIC %md
# MAGIC ## Example Queries

print("\n\n" + "🎯 " * 40)
print("RUNNING EXAMPLE RAG QUERIES")
print("🎯 " * 40)

# Query 1
result1 = rag_query("What is machine learning and how does it work?", top_k=5)

# Query 2
result2 = rag_query("Explain transformers and their applications in NLP", top_k=5)

# Query 3
result3 = rag_query("What are neural networks used for?", top_k=5)

# MAGIC %md
# MAGIC ## Save Results

print("\n\n" + "=" * 80)
print("💾 SAVING QUERY RESULTS")
print("=" * 80)

# Prepare results for storage
results_list = [result1, result2, result3]

# Save to Delta table
results_df = spark.createDataFrame([
    {
        'query': r['query'],
        'num_retrieved_chunks': len(r['retrieved_chunks']),
        'answer': r['answer'][:1000],  # Limit to 1000 chars
        'timestamp': r['timestamp']
    }
    for r in results_list
])

results_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("rag_query_results")

print("✅ Saved results to rag_query_results table")

# MAGIC %md
# MAGIC ## Summary

print("\n\n" + "=" * 80)
print("✅ RAG SYSTEM OPERATIONAL")
print("=" * 80)
print(f"""
📊 SYSTEM STATS:
   • Total Documents: {len(chunks_df)}
   • Total Embeddings: {len(embeddings_array)}
   • Query Model: all-MiniLM-L6-v2
   • Retrieval Method: Cosine Similarity
   
🎯 EXAMPLE QUERIES EXECUTED:
   • Query 1: Machine Learning Overview
   • Query 2: Transformers in NLP
   • Query 3: Neural Networks Applications
   
📁 DATA TABLES:
   ✅ chunk_embeddings - Vector embeddings
   ✅ processed_chunks - Text chunks
   ✅ rag_query_results - Query results
   
🚀 NEXT STEPS:
   1. Run custom queries using rag_query() function
   2. Integrate with web API (Flask/FastAPI)
   3. Add to Pinecone for production use
   4. Create dashboard for queries
   
""")
print("=" * 80)