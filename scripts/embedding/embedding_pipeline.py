"""
PHASE 1 WEEK 3: Embedding Pipeline
Convert text chunks to vector embeddings using Hugging Face
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from scripts.utils import setup_logger
from config import (
    PROCESSED_DATA_DIR, OUTPUT_DIR, EMBEDDING_MODEL,
    MAX_EMBEDDING_BATCH_SIZE
)

logger = setup_logger(__name__, log_file="embedding_pipeline.log")


class EmbeddingPipeline:
    """Convert text chunks to embeddings"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize embedding model"""
        logger.info(f"[LOADING] Embedding model: {model_name}")
        print(f"\n[LOADING] Embedding model: {model_name}")
        print("This may take a minute on first run...\n")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Get embedding dimension
        test_embedding = self.model.encode("test")
        self.embedding_dimension = len(test_embedding)
        
        logger.info(f"[OK] Model loaded: {model_name}")
        logger.info(f"[OK] Embedding dimension: {self.embedding_dimension}")
        print(f"[OK] Model loaded!")
        print(f"[OK] Embedding dimension: {self.embedding_dimension}\n")
    
    def load_chunks(self) -> Dict[str, List[Dict]]:
        """Load all chunks from processed directory"""
        logger.info("\n[LOADING] Chunks from processed directory")
        
        chunks_by_topic = {}
        chunk_files = PROCESSED_DATA_DIR.glob("chunks_*.json")
        
        for chunk_file in chunk_files:
            topic = chunk_file.stem.replace("chunks_", "")
            
            with open(chunk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks_by_topic[topic] = data['chunks']
            
            logger.info(f"[OK] Loaded {len(chunks_by_topic[topic])} chunks for topic: {topic}")
        
        total_chunks = sum(len(chunks) for chunks in chunks_by_topic.values())
        logger.info(f"[TOTAL] Loaded {total_chunks} chunks from {len(chunks_by_topic)} topics")
        
        return chunks_by_topic
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to embeddings
        
        Args:
            texts: List of text strings
        
        Returns:
            Array of embeddings (shape: [num_texts, embedding_dim])
        """
        logger.info(f"[EMBEDDING] Converting {len(texts)} texts to vectors")
        
        # Show progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=MAX_EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"[OK] Created {len(embeddings)} embeddings")
        return embeddings
    
    def embed_chunks(self, chunks_by_topic: Dict[str, List[Dict]]) -> Dict:
        """
        Embed all chunks organized by topic
        
        Args:
            chunks_by_topic: Dictionary of chunks organized by topic
        
        Returns:
            Dictionary with embeddings and metadata
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 1 WEEK 3: EMBEDDING ALL CHUNKS")
        logger.info("="*70)
        
        embeddings_by_topic = {}
        total_embedded = 0
        
        for topic, chunks in chunks_by_topic.items():
            print(f"\n[TOPIC] Embedding: {topic}")
            logger.info(f"\n[TOPIC] Embedding chunks for: {topic}")
            logger.info(f"Total chunks: {len(chunks)}")
            
            # Extract texts from chunks
            texts = [chunk['content'] for chunk in chunks]
            
            # Create embeddings
            embeddings = self.embed_texts(texts)
            
            # Create embedding objects with metadata
            topic_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                embedding_obj = {
                    'id': chunk['id'],
                    'title': chunk['title'],
                    'topic': chunk['topic'],
                    'source': chunk['source'],
                    'url': chunk['url'],
                    'content': chunk['content'],
                    'content_length': chunk['content_length'],
                    'embedding': embedding.tolist(),  # Convert numpy array to list
                    'embedding_model': self.model_name,
                    'embedding_dimension': self.embedding_dimension
                }
                topic_embeddings.append(embedding_obj)
            
            embeddings_by_topic[topic] = topic_embeddings
            total_embedded += len(topic_embeddings)
            
            logger.info(f"[OK] Embedded {len(topic_embeddings)} chunks for topic: {topic}")
            print(f"[OK] Embedded {len(topic_embeddings)} chunks")
        
        logger.info(f"\n[TOTAL] Embedded {total_embedded} chunks across {len(embeddings_by_topic)} topics")
        
        return embeddings_by_topic
    
    def save_embeddings(self, embeddings_by_topic: Dict):
        """Save embeddings to files"""
        logger.info("\n[SAVING] Embeddings to processed directory")
        
        total_saved = 0
        
        for topic, embeddings in embeddings_by_topic.items():
            output_file = PROCESSED_DATA_DIR / f"embeddings_{topic}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'topic': topic,
                        'total_embeddings': len(embeddings),
                        'embedding_model': self.model_name,
                        'embedding_dimension': self.embedding_dimension
                    },
                    'embeddings': embeddings
                }, f, ensure_ascii=False)
            
            total_saved += len(embeddings)
            logger.info(f"[OK] Saved {len(embeddings)} embeddings for topic: {topic}")
            print(f"[OK] Saved {len(embeddings)} embeddings for topic: {topic}")
        
        logger.info(f"[TOTAL] Saved {total_saved} embeddings")
        return total_saved
    
    def generate_embedding_report(self, embeddings_by_topic: Dict) -> Dict:
        """Generate embedding statistics report"""
        logger.info("\n[GENERATING] Embedding report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'topics': {},
            'statistics': {}
        }
        
        total_embeddings = 0
        all_embedding_norms = []
        
        for topic, embeddings in embeddings_by_topic.items():
            topic_count = len(embeddings)
            total_embeddings += topic_count
            
            # Calculate statistics for this topic
            embedding_vectors = np.array([e['embedding'] for e in embeddings])
            norms = np.linalg.norm(embedding_vectors, axis=1)
            
            all_embedding_norms.extend(norms.tolist())
            
            report['topics'][topic] = {
                'count': topic_count,
                'avg_embedding_norm': float(np.mean(norms)),
                'min_embedding_norm': float(np.min(norms)),
                'max_embedding_norm': float(np.max(norms))
            }
        
        # Global statistics
        all_norms = np.array(all_embedding_norms)
        report['statistics'] = {
            'total_embeddings': total_embeddings,
            'avg_embedding_norm': float(np.mean(all_norms)),
            'min_embedding_norm': float(np.min(all_norms)),
            'max_embedding_norm': float(np.max(all_norms))
        }
        
        return report
    
    def run_full_pipeline(self):
        """Run complete embedding pipeline"""
        logger.info("\n" + "="*70)
        logger.info("STARTING EMBEDDING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Load chunks
        chunks_by_topic = self.load_chunks()
        
        # Step 2: Embed all chunks
        embeddings_by_topic = self.embed_chunks(chunks_by_topic)
        
        # Step 3: Save embeddings
        total_saved = self.save_embeddings(embeddings_by_topic)
        
        # Step 4: Generate report
        report = self.generate_embedding_report(embeddings_by_topic)
        
        # Save report
        report_file = OUTPUT_DIR / "embedding_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[OK] Embedding report saved to: {report_file}")
        
        return report


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("PHASE 1 WEEK 3: EMBEDDING PIPELINE")
    print("="*70)
    
    # Create and run pipeline
    pipeline = EmbeddingPipeline()
    report = pipeline.run_full_pipeline()
    
    # Print report
    print("\n" + "="*70)
    print("EMBEDDING REPORT")
    print("="*70)
    print(f"\nModel: {report['model']}")
    print(f"Embedding Dimension: {report['embedding_dimension']}")
    print(f"Total Embeddings: {report['statistics']['total_embeddings']}")
    print(f"\nEmbedding Statistics:")
    print(f"  Average Norm: {report['statistics']['avg_embedding_norm']:.4f}")
    print(f"  Min Norm: {report['statistics']['min_embedding_norm']:.4f}")
    print(f"  Max Norm: {report['statistics']['max_embedding_norm']:.4f}")
    
    print(f"\nEmbeddings by Topic:")
    for topic, stats in report['topics'].items():
        print(f"  {topic}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg Norm: {stats['avg_embedding_norm']:.4f}")
    
    print("\n" + "="*70)
    print(f"[OK] Embeddings saved to: {PROCESSED_DATA_DIR}")
    print(f"[OK] Report saved to: {OUTPUT_DIR}/embedding_report.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()