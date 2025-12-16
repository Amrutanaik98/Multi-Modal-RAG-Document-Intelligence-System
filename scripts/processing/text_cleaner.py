"""
Text Cleaning Pipeline
Cleans and normalizes text for processing
"""

import re
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger

logger = setup_logger(__name__, log_file="text_processing.log")


class TextCleaner:
    """Clean and normalize text"""
    
    def __init__(self):
        logger.info("✅ Text Cleaner initialized")
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra spaces, tabs, newlines"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\n+', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        pattern = r'https?://\S+|www\.\S+'
        return re.sub(pattern, '', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses"""
        pattern = r'\S+@\S+'
        return re.sub(pattern, '', text)
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters"""
        if keep_punctuation:
            # Keep letters, numbers, spaces, and basic punctuation
            pattern = r'[^a-zA-Z0-9\s.,!?;:-]'
        else:
            # Remove everything except letters and numbers
            pattern = r'[^a-zA-Z0-9\s]'
        
        return re.sub(pattern, '', text)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text - main cleaning function"""
        logger.info("🧹 Cleaning text...")
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove emails
        text = self.remove_emails(text)
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Convert to lowercase (optional - comment out if you want to keep case)
        # text = text.lower()
        
        logger.info("✅ Text cleaned successfully")
        return text
    
    def clean_text(self, text: str) -> str:
        """Main method to clean text"""
        return self.normalize_text(text)


def main():
    """Example usage"""
    cleaner = TextCleaner()
    
    # Sample dirty text
    sample_text = """
    
    
    Machine Learning is awesome!!!
    
    Check this out: https://example.com/ml-tutorial
    
    Contact me at john@example.com for more info.
    
    
    Deep Learning    is    a    subset.
    
    """
    
    print("BEFORE:")
    print(repr(sample_text))
    print("\n" + "="*60 + "\n")
    
    cleaned = cleaner.clean_text(sample_text)
    
    print("AFTER:")
    print(repr(cleaned))


if __name__ == "__main__":
    main()