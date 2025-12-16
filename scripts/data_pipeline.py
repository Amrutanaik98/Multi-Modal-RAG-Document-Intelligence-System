"""
PHASE 1 WEEK 2: Mass Data Collection Pipeline
Scrape, clean, chunk, and validate 500+ educational documents
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scrapers.general_scraper import GeneralWebScraper
from scripts.scrapers.wikipedia_scraper import WikipediaScraper
from scripts.processing.text_cleaner import TextCleaner
from scripts.processing.text_chunker import TextChunker
from scripts.utils import setup_logger
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EDUCATIONAL_TOPICS,
    MIN_DOCUMENT_LENGTH, MAX_DOCUMENT_LENGTH
)

logger = setup_logger(__name__, log_file="data_pipeline.log")


class DataCollectionPipeline:
    """Complete pipeline for scraping, cleaning, and chunking data"""
    
    def __init__(self):
        """Initialize all components"""
        self.general_scraper = GeneralWebScraper()
        self.wiki_scraper = WikipediaScraper()
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker()
        
        self.raw_documents = []
        self.cleaned_documents = []
        self.chunks = []
        
        logger.info("[OK] Data Collection Pipeline initialized")
    
    def scrape_all_topics(self) -> List[Dict]:
        """Scrape all educational topics"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: SCRAPING ALL TOPICS")
        logger.info("="*70)
        
        all_results = []
        
        for topic, urls in EDUCATIONAL_TOPICS.items():
            logger.info(f"\n[TOPIC] Scraping: {topic}")
            logger.info(f"URLs to scrape: {len(urls)}")
            
            results = self.general_scraper.scrape_multiple_urls(urls)
            
            # Add topic to each result
            for result in results:
                result['topic'] = topic
            
            all_results.extend(results)
            logger.info(f"[OK] Scraped {len(results)} URLs for topic: {topic}")
        
        logger.info(f"\n[SUMMARY] Total documents scraped: {len(all_results)}")
        return all_results
    
    def scrape_wikipedia_topics(self) -> List[Dict]:
        """Scrape Wikipedia articles for each topic"""
        logger.info("\n" + "="*70)
        logger.info("BONUS: SCRAPING WIKIPEDIA ARTICLES")
        logger.info("="*70)
        
        all_results = []
        
        # Simple topic titles for Wikipedia
        wiki_topics = {
            'machine_learning': ["Machine learning", "Supervised learning", "Unsupervised learning"],
            'deep_learning': ["Deep learning", "Artificial neural network"],
            'nlp': ["Natural language processing", "Word embedding"],
            'llm': ["Large language model"],
            'rag': ["Information retrieval"],
            'ai_basics': ["Artificial intelligence"]
        }
        
        for topic, titles in wiki_topics.items():
            logger.info(f"\n[WIKIPEDIA] Scraping: {topic}")
            
            results = self.wiki_scraper.get_multiple_articles(titles)
            
            # Add topic to each result
            for result in results:
                result['topic'] = topic
                result['source'] = 'wikipedia'
            
            all_results.extend(results)
            logger.info(f"[OK] Scraped {len(results)} Wikipedia articles for topic: {topic}")
        
        logger.info(f"\n[SUMMARY] Total Wikipedia articles scraped: {len(all_results)}")
        return all_results
    
    def filter_documents(self, documents: List[Dict]) -> List[Dict]:
        """Filter documents by quality criteria"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: FILTERING DOCUMENTS")
        logger.info("="*70)
        
        filtered = []
        removed = 0
        
        for doc in documents:
            if doc['status'] != 'success':
                removed += 1
                continue
            
            content = doc.get('content', '')
            content_length = len(content)
            
            # Check length requirements
            if content_length < MIN_DOCUMENT_LENGTH:
                logger.warning(f"[TOO_SHORT] {doc.get('title', 'Unknown')}: {content_length} chars")
                removed += 1
                continue
            
            if content_length > MAX_DOCUMENT_LENGTH:
                logger.warning(f"[TOO_LONG] {doc.get('title', 'Unknown')}: {content_length} chars (truncating)")
                doc['content'] = content[:MAX_DOCUMENT_LENGTH]
            
            filtered.append(doc)
        
        logger.info(f"[SUMMARY] Kept: {len(filtered)}, Removed: {removed}")
        return filtered
    
    def clean_all_documents(self, documents: List[Dict]) -> List[Dict]:
        """Clean all documents"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: CLEANING DOCUMENTS")
        logger.info("="*70)
        
        cleaned = []
        
        for i, doc in enumerate(documents, 1):
            if i % 10 == 0:
                logger.info(f"[PROGRESS] Cleaned {i}/{len(documents)} documents")
            
            # Clean the content
            cleaned_content = self.text_cleaner.clean_text(doc.get('content', ''))
            
            # Create cleaned document
            cleaned_doc = {
                'title': doc.get('title', 'Unknown'),
                'url': doc.get('url', ''),
                'original_content': doc.get('content', ''),
                'cleaned_content': cleaned_content,
                'topic': doc.get('topic', 'unknown'),
                'source': doc.get('source', 'general'),
                'status': 'cleaned',
                'original_length': len(doc.get('content', '')),
                'cleaned_length': len(cleaned_content)
            }
            
            cleaned.append(cleaned_doc)
        
        logger.info(f"[OK] Cleaned {len(cleaned)} documents")
        return cleaned
    
    def chunk_all_documents(self, documents: List[Dict]) -> Dict:
        """Chunk all documents"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 4: CHUNKING DOCUMENTS")
        logger.info("="*70)
        
        chunks_by_topic = {}
        total_chunks = 0
        
        for i, doc in enumerate(documents, 1):
            if i % 10 == 0:
                logger.info(f"[PROGRESS] Chunked {i}/{len(documents)} documents")
            
            # Chunk the cleaned content
            text_chunks = self.text_chunker.chunk_text(doc['cleaned_content'])
            
            # Create chunk objects with metadata
            doc_chunks = []
            for chunk_idx, chunk in enumerate(text_chunks):
                chunk_obj = {
                    'id': f"{doc['title'].replace(' ', '_')}_{chunk_idx}",
                    'title': doc['title'],
                    'topic': doc['topic'],
                    'source': doc['source'],
                    'url': doc['url'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks),
                    'content': chunk,
                    'content_length': len(chunk)
                }
                doc_chunks.append(chunk_obj)
            
            # Organize by topic
            topic = doc['topic']
            if topic not in chunks_by_topic:
                chunks_by_topic[topic] = []
            
            chunks_by_topic[topic].extend(doc_chunks)
            total_chunks += len(doc_chunks)
        
        logger.info(f"[OK] Created {total_chunks} chunks from {len(documents)} documents")
        return chunks_by_topic
    
    def save_raw_documents(self, documents: List[Dict]):
        """Save raw scraped documents"""
        output_file = RAW_DATA_DIR / "raw_documents.json"
        
        logger.info(f"\n[SAVING] Raw documents to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_documents': len(documents),
                    'successful': sum(1 for d in documents if d['status'] == 'success')
                },
                'documents': documents
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Saved {len(documents)} raw documents")
    
    def save_cleaned_documents(self, documents: List[Dict]):
        """Save cleaned documents"""
        output_file = PROCESSED_DATA_DIR / "cleaned_documents.json"
        
        logger.info(f"\n[SAVING] Cleaned documents to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_documents': len(documents)
                },
                'documents': documents
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Saved {len(documents)} cleaned documents")
    
    def save_chunks(self, chunks_by_topic: Dict):
        """Save chunks by topic"""
        logger.info(f"\n[SAVING] Chunks by topic to: {PROCESSED_DATA_DIR}")
        
        total_chunks = 0
        
        for topic, chunks in chunks_by_topic.items():
            output_file = PROCESSED_DATA_DIR / f"chunks_{topic}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'topic': topic,
                        'total_chunks': len(chunks)
                    },
                    'chunks': chunks
                }, f, indent=2, ensure_ascii=False)
            
            total_chunks += len(chunks)
            logger.info(f"[OK] Saved {len(chunks)} chunks for topic: {topic}")
        
        logger.info(f"\n[SUMMARY] Total chunks saved: {total_chunks}")
    
    def generate_report(self, 
                       raw_docs: List[Dict],
                       cleaned_docs: List[Dict],
                       chunks_by_topic: Dict):
        """Generate detailed report"""
        logger.info("\n" + "="*70)
        logger.info("FINAL REPORT")
        logger.info("="*70)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'scraping': {
                'total_scraped': len(raw_docs),
                'successful': sum(1 for d in raw_docs if d['status'] == 'success'),
                'failed': sum(1 for d in raw_docs if d['status'] != 'success')
            },
            'filtering': {
                'total_kept': len(cleaned_docs),
                'total_removed': len(raw_docs) - len(cleaned_docs)
            },
            'chunking': {
                'total_documents': len(cleaned_docs),
                'total_chunks': sum(len(c) for c in chunks_by_topic.values()),
                'chunks_per_document': sum(len(c) for c in chunks_by_topic.values()) / max(len(cleaned_docs), 1)
            },
            'topics': {
                topic: len(chunks) for topic, chunks in chunks_by_topic.items()
            }
        }
        
        print("\n" + "="*70)
        print("DATA COLLECTION PIPELINE REPORT")
        print("="*70)
        print(f"\nSCRAPED: {report['scraping']['successful']}/{report['scraping']['total_scraped']} documents")
        print(f"FILTERED: {report['filtering']['total_kept']} documents kept")
        print(f"CHUNKED: {report['chunking']['total_chunks']} chunks created")
        print(f"CHUNKS PER DOC: {report['chunking']['chunks_per_document']:.1f}")
        print(f"\nTOPICS:")
        for topic, count in report['topics'].items():
            print(f"  - {topic}: {count} chunks")
        print("\n" + "="*70)
        
        # Save report
        report_file = OUTPUT_DIR / "pipeline_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[OK] Report saved to: {report_file}")
        
        return report
    
    def run_full_pipeline(self, use_wikipedia: bool = True):
        """Run complete pipeline"""
        logger.info("\n" + "="*70)
        logger.info("STARTING DATA COLLECTION PIPELINE")
        logger.info("="*70)
        
        # Step 1: Scrape
        raw_documents = self.scrape_all_topics()
        
        if use_wikipedia:
            wiki_documents = self.scrape_wikipedia_topics()
            raw_documents.extend(wiki_documents)
        
        # Step 2: Filter
        filtered_documents = self.filter_documents(raw_documents)
        
        # Step 3: Clean
        cleaned_documents = self.clean_all_documents(filtered_documents)
        
        # Step 4: Chunk
        chunks_by_topic = self.chunk_all_documents(cleaned_documents)
        
        # Step 5: Save
        self.save_raw_documents(raw_documents)
        self.save_cleaned_documents(cleaned_documents)
        self.save_chunks(chunks_by_topic)
        
        # Step 6: Report
        report = self.generate_report(raw_documents, cleaned_documents, chunks_by_topic)
        
        logger.info("\n[OK] PIPELINE COMPLETE!")
        
        return report


# Need to import OUTPUT_DIR
from config import OUTPUT_DIR


def main():
    """Run the pipeline"""
    print("\n" + "="*70)
    print("PHASE 1 WEEK 2: MASS DATA COLLECTION PIPELINE")
    print("="*70)
    
    # Create and run pipeline
    pipeline = DataCollectionPipeline()
    report = pipeline.run_full_pipeline(use_wikipedia=True)
    
    print("\n[OK] All data saved to:")
    print(f"    Raw: {RAW_DATA_DIR}")
    print(f"    Processed: {PROCESSED_DATA_DIR}")
    print(f"    Report: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()