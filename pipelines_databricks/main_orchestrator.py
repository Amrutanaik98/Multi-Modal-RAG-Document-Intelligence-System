# Databricks notebook source
# Databricks notebook source

# MAGIC %md
# MAGIC # RAG Pipeline - Fixed Orchestrator (No Logging Schema Issues)
# MAGIC Educational RAG System - Without problematic logging

# MAGIC %md
# MAGIC ## Stage 0: Install All Dependencies

print("=" * 80)
print("📦 INSTALLING ALL REQUIRED DEPENDENCIES")
print("=" * 80)

%pip install wikipedia python-dotenv arxiv feedparser requests beautifulsoup4 lxml selenium sentence-transformers torch pinecone-client youtube-transcript-api

print("\n✅ All dependencies installed!")

# MAGIC %md
# MAGIC ## Import Libraries & Setup

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add repo to path
sys.path.insert(0, '/Workspace/rag_project/scripts')

print("=" * 80)
print("🚀 RAG PIPELINE STARTED")
print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# Initialize counters
all_documents = []
chunk_count = 0
embedding_count = 0
uploaded_count = 0
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

# MAGIC %md
# MAGIC ## Stage 1: Scraping Documents

print("\n" + "=" * 80)
print("📥 STAGE 1: SCRAPING DOCUMENTS")
print("=" * 80)

# ============================================================================
# WIKIPEDIA SCRAPER
# ============================================================================
try:
    print("\n🔍 Scraping Wikipedia...")
    from scrapers.wikipedia_scraper import WikipediaScraper
    
    wiki_scraper = WikipediaScraper()
    wikipedia_docs = wiki_scraper.get_multiple_articles(titles=[
        'Machine Learning',
        'Artificial Intelligence',
        'Deep Learning',
        'Natural Language Processing',
        'Neural Networks',
        'Transformers',
        'Data Science'
    ])
    
    if wikipedia_docs and len(wikipedia_docs) > 0:
        all_documents.extend(wikipedia_docs)
        print(f"✅ Wikipedia: {len(wikipedia_docs)} documents scraped")
        wiki_status = 'success'
        wiki_count = len(wikipedia_docs)
    else:
        print("⚠️ Wikipedia: No documents returned")
        wiki_status = 'warning'
        wiki_count = 0
    
except Exception as e:
    print(f"❌ Wikipedia failed: {str(e)}")
    logger.error(f"Wikipedia error: {e}", exc_info=True)
    wiki_status = 'failed'
    wiki_count = 0

# ============================================================================
# ARXIV SCRAPER
# ============================================================================
try:
    print("\n🔍 Scraping arXiv...")
    from scrapers.arxiv_scraper import ArxivScraper
    
    arxiv_scraper = ArxivScraper()
    arxiv_docs = arxiv_scraper.get_multiple_papers(queries=[
        'machine learning',
        'deep learning',
        'natural language processing'
    ])
    
    if arxiv_docs and len(arxiv_docs) > 0:
        all_documents.extend(arxiv_docs)
        print(f"✅ arXiv: {len(arxiv_docs)} documents scraped")
        arxiv_status = 'success'
        arxiv_count = len(arxiv_docs)
    else:
        print("⚠️ arXiv: No documents returned")
        arxiv_status = 'warning'
        arxiv_count = 0
    
except Exception as e:
    print(f"❌ arXiv failed: {str(e)}")
    logger.error(f"arXiv error: {e}", exc_info=True)
    arxiv_status = 'failed'
    arxiv_count = 0

# ============================================================================
# HUGGINGFACE SCRAPER
# ============================================================================
try:
    print("\n🔍 Scraping HuggingFace...")
    from scrapers.huggingface_scraper import HuggingFaceScraper
    
    hf_scraper = HuggingFaceScraper()
    hf_docs = hf_scraper.get_multiple_docs(libraries=['transformers', 'datasets'])
    
    if hf_docs and len(hf_docs) > 0:
        all_documents.extend(hf_docs)
        print(f"✅ HuggingFace: {len(hf_docs)} documents scraped")
        hf_status = 'success'
        hf_count = len(hf_docs)
    else:
        print("⚠️ HuggingFace: No documents returned")
        hf_status = 'warning'
        hf_count = 0
    
except Exception as e:
    print(f"❌ HuggingFace failed: {str(e)}")
    logger.error(f"HuggingFace error: {e}", exc_info=True)
    hf_status = 'failed'
    hf_count = 0

# ============================================================================
# NORMALIZE AND CLEAN DOCUMENTS
# ============================================================================
print(f"\n📊 Total documents scraped: {len(all_documents)}")

if all_documents and len(all_documents) > 0:
    try:
        # Normalize all documents to consistent schema
        normalized_docs = []
        
        for doc in all_documents:
            try:
                # Ensure all required fields exist
                normalized_doc = {
                    'document_id': str(doc.get('title', 'unknown')[:100]).replace(' ', '_'),
                    'title': str(doc.get('title', 'No Title'))[:255],
                    'content': str(doc.get('content', ''))[:100000],
                    'url': str(doc.get('url', ''))[:500],
                    'source': str(doc.get('source', 'unknown'))[:100],
                    'scraped_date': datetime.now(),
                    'content_length': int(len(str(doc.get('content', '')))),
                    'created_at': datetime.now()
                }
                normalized_docs.append(normalized_doc)
            except Exception as e:
                logger.warning(f"Could not normalize document: {e}")
                continue
        
        if normalized_docs:
            # Create DataFrame with explicit schema
            df_raw = spark.createDataFrame(normalized_docs)
            
            # Write to Delta table - OVERWRITE to avoid schema conflicts
            df_raw.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .saveAsTable("raw_documents")
            
            print(f"✅ Saved {len(normalized_docs)} documents to raw_documents table")
            
        else:
            print("⚠️ No normalized documents to save")
        
    except Exception as e:
        print(f"❌ Failed to save raw documents: {str(e)}")
        logger.error(f"Raw save error: {e}", exc_info=True)
else:
    print("⚠️ No documents scraped - cannot proceed to next stage")

# MAGIC %md
# MAGIC ## Stage 2: Text Processing & Chunking

print("\n" + "=" * 80)
print("⚙️ STAGE 2: PROCESSING & CHUNKING")
print("=" * 80)

try:
    print("\n📝 Processing documents...")
    
    # Import text processing functions
    import re
    
    def clean_text(text):
        """Clean and normalize text"""
        if not text:
            return ""
        try:
            text = str(text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            # Remove special characters
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\:\;]', '', text)
            # Remove newlines and tabs
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Remove extra spaces
            text = ' '.join(text.split())
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    def chunk_text(text, chunk_size=512, overlap=50):
        """Split text into overlapping chunks"""
        if not text:
            return []
        try:
            text = str(text)
            words = text.split()
            chunks = []
            step = chunk_size - overlap
            
            for i in range(0, len(words), step):
                chunk = ' '.join(words[i:i+chunk_size])
                if len(chunk.strip()) > 0:
                    chunks.append(chunk)
            
            return chunks if chunks else [text]
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return [text]
    
    # Read raw documents
    raw_df = spark.table("raw_documents")
    raw_count = raw_df.count()
    print(f"📖 Read {raw_count} raw documents")
    
    if raw_count > 0:
        raw_pandas = raw_df.toPandas()
        processed_chunks = []
        
        for idx, row in raw_pandas.iterrows():
            try:
                content = str(row.get('content', ''))
                if not content or len(content.strip()) < 10:
                    continue
                
                # Clean text
                cleaned_text = clean_text(content)
                
                # Chunk text
                chunks = chunk_text(cleaned_text, chunk_size=512, overlap=50)
                
                # Create chunk records
                for chunk_idx, chunk_text_val in enumerate(chunks):
                    if len(chunk_text_val.strip()) > 0:
                        chunk_record = {
                            'chunk_id': f"{str(row.get('document_id', idx))[:50]}_{chunk_idx}",
                            'document_id': str(row.get('document_id', idx))[:100],
                            'chunk_text': str(chunk_text_val)[:100000],
                            'chunk_number': int(chunk_idx + 1),
                            'topic': 'ai',
                            'difficulty_level': 'intermediate',
                            'content_type': str(row.get('source', 'unknown'))[:100],
                            'keywords': '',
                            'text_length': int(len(chunk_text_val)),
                            'word_count': int(len(chunk_text_val.split())),
                            'created_at': datetime.now()
                        }
                        processed_chunks.append(chunk_record)
            except Exception as e:
                print(f"⚠️ Error processing document {idx}: {str(e)}")
                continue
        
        if processed_chunks:
            # Create DataFrame and write
            chunks_df = spark.createDataFrame(processed_chunks)
            chunks_df.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .saveAsTable("processed_chunks")
            
            chunk_count = len(processed_chunks)
            print(f"✅ Created {chunk_count} chunks from {raw_count} documents")
        else:
            print("⚠️ No chunks created!")
    else:
        print("⚠️ No raw documents to process!")
    
except Exception as e:
    print(f"❌ Processing failed: {str(e)}")
    logger.error(f"Processing error: {e}", exc_info=True)
    chunk_count = 0

# MAGIC %md
# MAGIC ## Stage 3: Generate Embeddings

print("\n" + "=" * 80)
print("🔢 STAGE 3: GENERATING EMBEDDINGS")
print("=" * 80)

try:
    from sentence_transformers import SentenceTransformer
    
    print("\n🔄 Reading processed chunks...")
    chunks_df = spark.table("processed_chunks")
    chunks_count = chunks_df.count()
    print(f"📦 Read {chunks_count} chunks")
    
    if chunks_count > 0:
        chunks_pandas = chunks_df.toPandas()
        embeddings_list = []
        
        print("\n🧠 Loading embedding model...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            model = None
        
        if model:
            print("\n🧠 Generating embeddings...")
            for idx, row in chunks_pandas.iterrows():
                try:
                    chunk_text_val = str(row.get('chunk_text', ''))
                    
                    if not chunk_text_val or len(chunk_text_val.strip()) == 0:
                        continue
                    
                    # Generate embedding
                    embedding = model.encode(chunk_text_val, convert_to_tensor=False)
                    
                    if embedding is not None:
                        embedding_record = {
                            'chunk_id': str(row['chunk_id'])[:100],
                            'embedding': embedding.tolist(),
                            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                            'embedding_dimension': int(len(embedding)),
                            'created_at': datetime.now()
                        }
                        embeddings_list.append(embedding_record)
                        
                        if (idx + 1) % 100 == 0:
                            print(f"   ✓ Generated {idx + 1} embeddings...")
                except Exception as e:
                    print(f"⚠️ Error at chunk {idx}: {str(e)}")
                    continue
            
            if embeddings_list:
                embeddings_df = spark.createDataFrame(embeddings_list)
                embeddings_df.write \
                    .format("delta") \
                    .mode("overwrite") \
                    .option("overwriteSchema", "true") \
                    .saveAsTable("chunk_embeddings")
                
                embedding_count = len(embeddings_list)
                print(f"✅ Generated embeddings for {embedding_count} chunks")
            else:
                print("⚠️ No embeddings generated!")
        else:
            print("⚠️ Could not load embedding model - skipping embedding generation")
    else:
        print("⚠️ No chunks to embed!")
    
except Exception as e:
    print(f"❌ Embedding generation failed: {str(e)}")
    logger.error(f"Embedding error: {e}", exc_info=True)
    embedding_count = 0

# MAGIC %md
# MAGIC ## Summary & Final Report

print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETED!")
print("=" * 80)

try:
    # Get final counts from tables
    try:
        raw_count = spark.table("raw_documents").count()
    except:
        raw_count = 0
    
    try:
        chunk_count_final = spark.table("processed_chunks").count()
    except:
        chunk_count_final = 0
    
    try:
        embedding_count_final = spark.table("chunk_embeddings").count()
    except:
        embedding_count_final = 0
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   SCRAPING SOURCES:")
    print(f"     • Wikipedia: {wiki_count} documents ({wiki_status})")
    print(f"     • arXiv: {arxiv_count} documents ({arxiv_status})")
    print(f"     • HuggingFace: {hf_count} documents ({hf_status})")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   PIPELINE RESULTS:")
    print(f"     • Raw documents saved: {raw_count}")
    print(f"     • Chunks created: {chunk_count_final}")
    print(f"     • Embeddings generated: {embedding_count_final}")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
except Exception as e:
    print(f"Error getting final counts: {e}")
    logger.error(f"Final summary error: {e}", exc_info=True)

print("\n" + "=" * 80)
print(f"⏰ Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)
print("🎉 RAG pipeline execution finished!")
print("=" * 80)
print("\n📊 DATA TABLES CREATED:")
print("   ✅ raw_documents - All scraped documents")
print("   ✅ processed_chunks - Text chunks for embeddings")
print("   ✅ chunk_embeddings - Vector embeddings for RAG")
print("=" * 80)