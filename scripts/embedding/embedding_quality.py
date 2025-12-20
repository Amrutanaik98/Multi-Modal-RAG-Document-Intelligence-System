"""
Embedding Quality Analysis
Analyze and validate the quality of generated embeddings
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger
from config import PROCESSED_DATA_DIR, OUTPUT_DIR

logger = setup_logger(__name__, log_file="embedding_quality.log")


class EmbeddingQualityAnalyzer:
    """Analyze quality of embeddings"""
    
    def __init__(self):
        """Initialize quality analyzer"""
        logger.info("[OK] Embedding Quality Analyzer initialized")
        print("[OK] Embedding Quality Analyzer initialized\n")
    
    def load_embeddings(self) -> Dict[str, List[Dict]]:
        """Load all embeddings from processed directory"""
        logger.info("\n[LOADING] Embeddings from processed directory")
        print("[LOADING] Embeddings from processed directory")
        
        embeddings_by_topic = {}
        embedding_files = list(PROCESSED_DATA_DIR.glob("embeddings_*.json"))
        
        if not embedding_files:
            logger.warning(f"[WARNING] No embedding files found in: {PROCESSED_DATA_DIR}")
            print(f"[WARNING] No embedding files found in: {PROCESSED_DATA_DIR}\n")
            return embeddings_by_topic
        
        for embedding_file in embedding_files:
            try:
                topic = embedding_file.stem.replace("embeddings_", "")
                
                with open(embedding_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    embeddings_by_topic[topic] = data['embeddings']
                
                logger.info(f"[OK] Loaded {len(embeddings_by_topic[topic])} embeddings for: {topic}")
                print(f"[OK] Loaded {len(embeddings_by_topic[topic])} embeddings for: {topic}")
            
            except Exception as e:
                logger.error(f"[ERROR] Failed to load {embedding_file.name}: {e}")
                print(f"[ERROR] Failed to load {embedding_file.name}: {e}")
        
        total_embeddings = sum(len(embs) for embs in embeddings_by_topic.values())
        logger.info(f"[TOTAL] Loaded {total_embeddings} embeddings from {len(embeddings_by_topic)} topics")
        print(f"[TOTAL] Loaded {total_embeddings} embeddings from {len(embeddings_by_topic)} topics\n")
        
        return embeddings_by_topic
    
    def analyze_vector_properties(self, embeddings_by_topic: Dict) -> Dict:
        """
        Analyze vector properties (norms, dimensions, etc.)
        
        Args:
            embeddings_by_topic: Dictionary of embeddings organized by topic
        
        Returns:
            Analysis results
        """
        logger.info("\n[ANALYZING] Vector properties")
        print("[ANALYZING] Vector properties")
        
        analysis = {}
        
        for topic, embeddings in embeddings_by_topic.items():
            try:
                # Extract vectors
                vectors = np.array([e['embedding'] for e in embeddings])
                
                # Calculate norms
                norms = np.linalg.norm(vectors, axis=1)
                
                # Calculate similarities (for subset)
                if len(vectors) > 100:
                    subset = vectors[:100]
                else:
                    subset = vectors
                
                similarities = self._calculate_cosine_similarities(subset)
                
                # Store analysis
                analysis[topic] = {
                    'count': len(embeddings),
                    'dimension': len(embeddings[0]['embedding']) if embeddings else 0,
                    'norm_stats': {
                        'mean': float(np.mean(norms)),
                        'std': float(np.std(norms)),
                        'min': float(np.min(norms)),
                        'max': float(np.max(norms))
                    },
                    'similarity_stats': {
                        'mean': float(np.mean(similarities)) if len(similarities) > 0 else 0,
                        'std': float(np.std(similarities)) if len(similarities) > 0 else 0,
                        'min': float(np.min(similarities)) if len(similarities) > 0 else 0,
                        'max': float(np.max(similarities)) if len(similarities) > 0 else 0
                    }
                }
                
                logger.info(f"[OK] Analyzed {len(embeddings)} embeddings for: {topic}")
            
            except Exception as e:
                logger.error(f"[ERROR] Failed to analyze {topic}: {e}")
                print(f"[ERROR] Failed to analyze {topic}: {e}")
        
        return analysis
    
    def _calculate_cosine_similarities(self, vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between vectors"""
        try:
            # Normalize vectors
            normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            
            # Calculate cosine similarity matrix
            similarities = np.dot(normalized, normalized.T)
            
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(similarities), k=1).astype(bool)
            return similarities[mask]
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to calculate similarities: {e}")
            return np.array([])
    
    def detect_outliers(self, embeddings_by_topic: Dict) -> Dict:
        """
        Detect outlier embeddings
        
        Args:
            embeddings_by_topic: Dictionary of embeddings organized by topic
        
        Returns:
            Dictionary of outliers per topic
        """
        logger.info("\n[DETECTING] Outlier embeddings")
        print("[DETECTING] Outlier embeddings")
        
        outliers = {}
        
        for topic, embeddings in embeddings_by_topic.items():
            try:
                vectors = np.array([e['embedding'] for e in embeddings])
                norms = np.linalg.norm(vectors, axis=1)
                
                # Calculate z-score
                mean_norm = np.mean(norms)
                std_norm = np.std(norms)
                z_scores = np.abs((norms - mean_norm) / (std_norm + 1e-8))
                
                # Find outliers (z-score > 3)
                outlier_indices = np.where(z_scores > 3)[0]
                
                outliers[topic] = {
                    'count': int(len(outlier_indices)),
                    'percentage': float(len(outlier_indices) / len(embeddings) * 100) if len(embeddings) > 0 else 0,
                    'outlier_indices': outlier_indices.tolist()[:10]  # Top 10
                }
                
                logger.info(f"[OK] Found {len(outlier_indices)} outliers in: {topic}")
            
            except Exception as e:
                logger.error(f"[ERROR] Failed to detect outliers for {topic}: {e}")
                print(f"[ERROR] Failed to detect outliers for {topic}: {e}")
        
        return outliers
    
    def check_embedding_diversity(self, embeddings_by_topic: Dict) -> Dict:
        """
        Check diversity of embeddings
        
        Args:
            embeddings_by_topic: Dictionary of embeddings organized by topic
        
        Returns:
            Diversity metrics
        """
        logger.info("\n[CHECKING] Embedding diversity")
        print("[CHECKING] Embedding diversity")
        
        diversity = {}
        
        for topic, embeddings in embeddings_by_topic.items():
            try:
                if len(embeddings) < 2:
                    diversity[topic] = {'status': 'insufficient_data'}
                    continue
                
                vectors = np.array([e['embedding'] for e in embeddings])
                
                # Calculate pairwise distances
                from scipy.spatial.distance import pdist
                distances = pdist(vectors, metric='cosine')
                
                diversity[topic] = {
                    'mean_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances)),
                    'min_distance': float(np.min(distances)),
                    'max_distance': float(np.max(distances))
                }
                
                logger.info(f"[OK] Calculated diversity for: {topic}")
            
            except Exception as e:
                logger.error(f"[ERROR] Failed to check diversity for {topic}: {e}")
                print(f"[ERROR] Failed to check diversity for {topic}: {e}")
        
        return diversity
    
    def generate_quality_report(self, 
                               analysis: Dict, 
                               outliers: Dict, 
                               diversity: Dict) -> Dict:
        """Generate comprehensive quality report"""
        logger.info("\n[GENERATING] Quality report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'vector_analysis': analysis,
            'outlier_detection': outliers,
            'diversity_metrics': diversity,
            'quality_summary': self._summarize_quality(analysis, outliers)
        }
        
        return report
    
    def _summarize_quality(self, analysis: Dict, outliers: Dict) -> Dict:
        """Summarize quality metrics"""
        
        total_embeddings = sum(a['count'] for a in analysis.values())
        total_outliers = sum(o['count'] for o in outliers.values())
        
        # Determine quality status
        if total_embeddings == 0:
            outlier_percentage = 0
            quality_status = 'no_data'
        else:
            outlier_percentage = (total_outliers / total_embeddings * 100)
            if outlier_percentage < 1:
                quality_status = 'excellent'
            elif outlier_percentage < 5:
                quality_status = 'good'
            elif outlier_percentage < 10:
                quality_status = 'acceptable'
            else:
                quality_status = 'needs_attention'
        
        summary = {
            'total_embeddings': total_embeddings,
            'total_outliers': total_outliers,
            'outlier_percentage': float(outlier_percentage),
            'quality_status': quality_status
        }
        
        return summary
    
    def run_full_analysis(self) -> Dict:
        """Run complete embedding quality analysis"""
        logger.info("\n" + "="*70)
        logger.info("EMBEDDING QUALITY ANALYSIS")
        logger.info("="*70)
        
        # Step 1: Load embeddings
        embeddings_by_topic = self.load_embeddings()
        
        if not embeddings_by_topic:
            logger.warning("[WARNING] No embeddings to analyze")
            print("[WARNING] No embeddings to analyze\n")
            return {}
        
        # Step 2: Analyze vector properties
        analysis = self.analyze_vector_properties(embeddings_by_topic)
        
        # Step 3: Detect outliers
        outliers = self.detect_outliers(embeddings_by_topic)
        
        # Step 4: Check diversity
        diversity = self.check_embedding_diversity(embeddings_by_topic)
        
        # Step 5: Generate report
        report = self.generate_quality_report(analysis, outliers, diversity)
        
        # Save report
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_file = OUTPUT_DIR / "embedding_quality_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[OK] Quality report saved to: {report_file}")
        
        return report


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("EMBEDDING QUALITY ANALYSIS")
    print("="*70 + "\n")
    
    analyzer = EmbeddingQualityAnalyzer()
    report = analyzer.run_full_analysis()
    
    if report:
        # Print summary
        print("\n" + "="*70)
        print("QUALITY SUMMARY")
        print("="*70)
        
        summary = report['quality_summary']
        print(f"\nTotal Embeddings: {summary['total_embeddings']}")
        print(f"Total Outliers: {summary['total_outliers']}")
        print(f"Outlier Percentage: {summary['outlier_percentage']:.2f}%")
        print(f"Quality Status: {summary['quality_status'].upper()}")
        
        # Print vector analysis
        if report['vector_analysis']:
            print(f"\nVector Analysis by Topic:")
            for topic, analysis in report['vector_analysis'].items():
                print(f"\n  {topic.upper()}:")
                print(f"    Count: {analysis['count']}")
                print(f"    Dimension: {analysis['dimension']}")
                print(f"    Norm Mean: {analysis['norm_stats']['mean']:.4f}")
                print(f"    Norm Std: {analysis['norm_stats']['std']:.4f}")
                if 'mean' in analysis['similarity_stats'] and analysis['similarity_stats']['mean'] > 0:
                    print(f"    Similarity Mean: {analysis['similarity_stats']['mean']:.4f}")
        
        # Print outliers
        if report['outlier_detection']:
            print(f"\nOutliers by Topic:")
            for topic, outlier_info in report['outlier_detection'].items():
                if 'percentage' in outlier_info:
                    print(f"  {topic}: {outlier_info['count']} ({outlier_info['percentage']:.2f}%)")
        
        print("\n" + "="*70)
        print(f"[OK] Report saved to: {OUTPUT_DIR}/embedding_quality_report.json")
        print("="*70 + "\n")
    else:
        print("\n[WARNING] No report generated!\n")


if __name__ == "__main__":
    main()