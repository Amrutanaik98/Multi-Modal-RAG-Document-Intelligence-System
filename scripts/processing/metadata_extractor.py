"""
Metadata Extraction Pipeline
Extract metadata from chunks (keywords, topics, difficulty level, etc.)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger
from config import PROCESSED_DATA_DIR, OUTPUT_DIR

logger = setup_logger(__name__, log_file="metadata_extraction.log")


class MetadataExtractor:
    """Extract metadata from text chunks"""
    
    def __init__(self):
        """Initialize metadata extractor"""
        logger.info("[OK] Metadata Extractor initialized")
        print("[OK] Metadata Extractor initialized\n")
        
        # Define topic keywords
        self.topic_keywords = {
            'nlp': ['nlp', 'text', 'language', 'tokenization', 'bert', 'gpt', 
                   'transformer', 'sentiment', 'translation', 'parsing', 'lexicon'],
            'cv': ['computer vision', 'image', 'cnn', 'object detection', 
                  'segmentation', 'opencv', 'yolo', 'rcnn', 'convolutional'],
            'llm': ['large language model', 'gpt', 'llama', 'generative', 
                   'prompt', 'fine-tune', 'instruction', 'completion'],
            'rag': ['retrieval', 'augmented', 'generation', 'vector', 
                   'embedding', 'semantic search', 'rag'],
            'ml': ['machine learning', 'supervised', 'unsupervised', 
                  'classification', 'regression', 'clustering', 'feature'],
            'dl': ['deep learning', 'neural network', 'gradient', 'backprop', 
                  'layer', 'activation', 'loss function', 'optimization']
        }
        
        # Define difficulty indicators
        self.advanced_terms = {
            'architecture', 'optimization', 'hyperparameter', 'gradient',
            'backpropagation', 'neural', 'tensor', 'algorithm', 'convergence',
            'regularization', 'activation', 'convolution', 'transformer',
            'attention', 'fourier', 'eigenvalue', 'matrix', 'calculus'
        }
        
        self.beginner_terms = {
            'basic', 'introduction', 'simple', 'example', 'tutorial',
            'learn', 'basics', 'getting started', 'beginner', 'overview',
            'what is', 'how to', 'explained', 'for dummies', 'introduction to'
        }
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """
        Extract keywords from text using simple frequency analysis
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
        
        Returns:
            List of keywords
        """
        # Common stopwords to exclude
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        # Extract words
        words = text.lower().split()
        
        # Filter words
        filtered_words = [
            w.strip('.,!?;:-') for w in words 
            if w.lower().strip('.,!?;:-') not in stopwords 
            and len(w.strip('.,!?;:-')) > 3
        ]
        
        # Get most frequent
        word_freq = Counter(filtered_words)
        keywords = [word for word, _ in word_freq.most_common(num_keywords)]
        
        return keywords
    
    def detect_difficulty_level(self, text: str) -> str:
        """
        Detect difficulty level of content
        
        Args:
            text: Input text
        
        Returns:
            'beginner', 'intermediate', or 'advanced'
        """
        text_lower = text.lower()
        
        # Count indicators
        advanced_count = sum(1 for term in self.advanced_terms if term in text_lower)
        beginner_count = sum(1 for term in self.beginner_terms if term in text_lower)
        
        # Determine difficulty
        if advanced_count > beginner_count:
            return 'advanced'
        elif beginner_count > advanced_count / 2:
            return 'beginner'
        else:
            return 'intermediate'
    
    def detect_content_type(self, text: str) -> str:
        """
        Detect type of content
        
        Args:
            text: Input text
        
        Returns:
            Content type
        """
        text_lower = text.lower()
        
        # Check for code indicators
        if '```' in text or 'def ' in text or 'import ' in text or 'class ' in text:
            return 'code_example'
        
        # Check for research paper indicators
        if 'abstract' in text_lower or 'conclusion' in text_lower or 'introduction' in text_lower:
            if 'arxiv' in text_lower or 'paper' in text_lower:
                return 'research_paper'
        
        # Check for documentation indicators
        if 'function' in text_lower or 'parameter' in text_lower or 'return' in text_lower:
            if 'api' in text_lower or 'documentation' in text_lower:
                return 'documentation'
        
        # Check for tutorial indicators
        if 'step' in text_lower or 'tutorial' in text_lower or 'guide' in text_lower:
            return 'tutorial'
        
        # Check for blog/article
        if 'article' in text_lower or 'blog' in text_lower or 'post' in text_lower:
            return 'blog_article'
        
        # Default
        return 'general_content'
    
    def detect_topic(self, text: str) -> str:
        """
        Detect topic of content
        
        Args:
            text: Input text
        
        Returns:
            Topic: 'nlp', 'cv', 'ml', 'dl', 'rag', 'llm', or 'other'
        """
        text_lower = text.lower()
        
        # Count matches for each topic
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            topic_scores[topic] = score
        
        # Return topic with highest score
        if max(topic_scores.values()) > 0:
            return max(topic_scores, key=topic_scores.get)
        else:
            return 'other'
    
    def extract_metadata(self, chunk: Dict) -> Dict:
        """
        Extract all metadata for a chunk
        
        Args:
            chunk: Input chunk dictionary with 'content' key
        
        Returns:
            Chunk with added metadata
        """
        text = chunk.get('content', '')
        
        # Extract metadata
        metadata = {
            'keywords': self.extract_keywords(text, num_keywords=5),
            'difficulty_level': self.detect_difficulty_level(text),
            'content_type': self.detect_content_type(text),
            'topic': self.detect_topic(text),
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        # Add to chunk
        chunk['metadata'] = metadata
        return chunk
    
    def process_chunks_from_files(self) -> Dict:
        """
        Process all chunks from JSON files and extract metadata
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info("\n" + "="*70)
        logger.info("METADATA EXTRACTION PIPELINE")
        logger.info("="*70)
        
        # Find all chunk files
        chunk_files = list(PROCESSED_DATA_DIR.glob("chunks_*.json"))
        
        if not chunk_files:
            logger.warning("[WARNING] No chunk files found in: {PROCESSED_DATA_DIR}")
            print(f"[WARNING] No chunk files found in: {PROCESSED_DATA_DIR}\n")
            return {}
        
        total_chunks = 0
        stats = {
            'topics': {},
            'difficulty_levels': {},
            'content_types': {},
            'total_chunks': 0
        }
        
        # Process each chunk file
        for chunk_file in chunk_files:
            logger.info(f"\n[PROCESSING] {chunk_file.name}")
            print(f"\n[PROCESSING] {chunk_file.name}")
            
            try:
                # Load chunks
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chunks = data.get('chunks', [])
                logger.info(f"[OK] Loaded {len(chunks)} chunks")
                print(f"[OK] Loaded {len(chunks)} chunks")
                
                # Extract metadata for each chunk
                for chunk in chunks:
                    chunk = self.extract_metadata(chunk)
                    total_chunks += 1
                    
                    # Update stats
                    topic = chunk['metadata']['topic']
                    difficulty = chunk['metadata']['difficulty_level']
                    content_type = chunk['metadata']['content_type']
                    
                    stats['topics'][topic] = stats['topics'].get(topic, 0) + 1
                    stats['difficulty_levels'][difficulty] = stats['difficulty_levels'].get(difficulty, 0) + 1
                    stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
                
                # Save enhanced chunks
                output_file = PROCESSED_DATA_DIR / f"chunks_with_metadata_{chunk_file.stem.split('_')[1]}.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'metadata': {
                            'timestamp': datetime.now().isoformat(),
                            'total_chunks': len(chunks)
                        },
                        'chunks': chunks
                    }, f, ensure_ascii=False, indent=2)
                
                logger.info(f"[OK] Saved {len(chunks)} chunks with metadata to: {output_file}")
                print(f"[OK] Saved {len(chunks)} chunks with metadata")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to process {chunk_file.name}: {e}")
                print(f"[ERROR] Failed to process {chunk_file.name}: {e}")
        
        stats['total_chunks'] = total_chunks
        return stats
    
    def generate_report(self, stats: Dict) -> Dict:
        """Generate metadata extraction report"""
        logger.info("\n[GENERATING] Metadata extraction report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_chunks_processed': stats['total_chunks'],
            'topics': stats['topics'],
            'difficulty_levels': stats['difficulty_levels'],
            'content_types': stats['content_types']
        }
        
        return report
    
    def run_full_pipeline(self):
        """Run complete metadata extraction pipeline"""
        logger.info("\n" + "="*70)
        logger.info("STARTING METADATA EXTRACTION PIPELINE")
        logger.info("="*70)
        
        # Process chunks and extract metadata
        stats = self.process_chunks_from_files()
        
        # Generate report
        report = self.generate_report(stats)
        
        # Save report
        report_file = OUTPUT_DIR / "metadata_extraction_report.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[OK] Report saved to: {report_file}")
        
        return report


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("METADATA EXTRACTION PIPELINE")
    print("="*70 + "\n")
    
    # Create extractor and run pipeline
    extractor = MetadataExtractor()
    report = extractor.run_full_pipeline()
    
    # Print summary
    print("\n" + "="*70)
    print("METADATA EXTRACTION SUMMARY")
    print("="*70)
    
    if report:
        print(f"\nTotal Chunks Processed: {report['total_chunks_processed']}")
        
        if report['topics']:
            print(f"\nTopics Distribution:")
            for topic, count in sorted(report['topics'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / report['total_chunks_processed'] * 100) if report['total_chunks_processed'] > 0 else 0
                print(f"  {topic}: {count} ({percentage:.1f}%)")
        
        if report['difficulty_levels']:
            print(f"\nDifficulty Levels:")
            for level, count in sorted(report['difficulty_levels'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / report['total_chunks_processed'] * 100) if report['total_chunks_processed'] > 0 else 0
                print(f"  {level}: {count} ({percentage:.1f}%)")
        
        if report['content_types']:
            print(f"\nContent Types:")
            for content_type, count in sorted(report['content_types'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / report['total_chunks_processed'] * 100) if report['total_chunks_processed'] > 0 else 0
                print(f"  {content_type}: {count} ({percentage:.1f}%)")
        
        print("\n" + "="*70 + "\n")
    else:
        print("\n[WARNING] No chunks to process!\n")


if __name__ == "__main__":
    main()