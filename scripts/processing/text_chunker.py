"""
Text Chunking Pipeline
Split text into chunks for embedding
"""

from typing import List
import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = setup_logger(__name__, log_file="text_processing.log")


class TextChunker:
    """Split text into chunks"""
    
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"✅ TextChunker initialized (size: {chunk_size}, overlap: {chunk_overlap})")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
        
        Returns:
            List of text chunks
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        if len(text) < self.chunk_size:
            logger.info(f"Text shorter than chunk size, returning as single chunk")
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # End position
            end = start + self.chunk_size
            
            # Get chunk
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start with overlap
            start += self.chunk_size - self.chunk_overlap
        
        logger.info(f"✅ Created {len(chunks)} chunks from text")
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Split by sentences (better for readability)
        
        Args:
            text: Input text
        
        Returns:
            List of sentence chunks
        """
        # Split by common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"✅ Created {len(chunks)} sentence chunks")
        return chunks


def main():
    """Example usage"""
    chunker = TextChunker()
    
    # Sample text
    sample_text = """
    Machine learning is a subset of artificial intelligence (AI) that provides 
    systems the ability to automatically learn and improve from experience without 
    being explicitly programmed. It focuses on the development of computer programs 
    that can access data and use it to learn for themselves.
    
    The process of learning begins with observations or data, such as examples, 
    direct experience, or instruction, in order to look for patterns in data and 
    make better decisions in the future based on the examples that we provide.
    
    Deep learning is a subset of machine learning that uses neural networks with 
    multiple layers. These networks can learn hierarchical representations of data, 
    making them powerful for complex tasks like image recognition and natural 
    language processing.
    """
    
    print("ORIGINAL TEXT:")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    # Chunk it
    chunks = chunker.chunk_text(sample_text)
    
    print(f"TEXT CHUNKS ({len(chunks)} total):\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(chunk)
        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    main()