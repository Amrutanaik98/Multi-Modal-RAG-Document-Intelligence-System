"""
Wikipedia-Specific Scraper
Optimized for scraping Wikipedia articles
Uses Wikipedia Python library for easier access
"""

import wikipedia
from typing import Dict, List
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger

logger = setup_logger(__name__, log_file="wikipedia_scraping.log")


class WikipediaScraper:
    """Scrape educational content from Wikipedia"""
    
    def __init__(self):
        """Initialize Wikipedia scraper"""
        wikipedia.set_lang("en")
        logger.info("[OK] Wikipedia Scraper initialized")
    
    def get_article(self, title: str) -> Dict:
        """
        Get Wikipedia article
        
        Args:
            title: Wikipedia article title
        
        Returns:
            Dictionary with article content
        """
        try:
            logger.info(f"[FETCHING] Wikipedia: {title}")
            
            # Fetch the page with auto_suggest to handle typos and variations
            page = wikipedia.page(title, auto_suggest=True)
            
            result = {
                'title': page.title,
                'url': page.url,
                'content': page.content,
                'summary': page.summary[:500],  # First 500 chars of summary
                'links': page.links[:20],  # First 20 links
                'source': 'wikipedia',
                'status': 'success',
                'content_length': len(page.content)
            }
            
            logger.info(f"[OK] Got Wikipedia article: {page.title} ({len(page.content)} chars)")
            return result
        
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"[DISAMBIGUATION] Page has multiple meanings: {title}")
            logger.info(f"[INFO] Options: {e.options[:3]}")
            
            # Try to get the first suggestion
            try:
                if e.options:
                    first_option = e.options[0]
                    logger.info(f"[RETRY] Trying first option: {first_option}")
                    return self.get_article(first_option)
            except:
                pass
            
            return {
                'title': title,
                'status': 'disambiguation',
                'error': 'Page has multiple meanings'
            }
        
        except wikipedia.exceptions.PageError:
            logger.error(f"[NOT_FOUND] Page not found: {title}")
            return {
                'title': title,
                'status': 'not_found',
                'error': 'Page not found'
            }
        
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error for {title}: {e}")
            return {
                'title': title,
                'status': 'error',
                'error': str(e)
            }
    
    def get_multiple_articles(self, titles: List[str]) -> List[Dict]:
        """
        Get multiple Wikipedia articles with deduplication
        
        Args:
            titles: List of article titles
        
        Returns:
            List of article dictionaries (deduplicated)
        """
        results = []
        seen_urls = set()  # Track URLs to avoid duplicates
        
        for i, title in enumerate(titles, 1):
            logger.info(f"\n[{i}/{len(titles)}] Fetching: {title}")
            result = self.get_article(title)
            
            # Check if already scraped this article
            if result['status'] == 'success':
                url = result.get('url', '')
                if url in seen_urls:
                    logger.warning(f"[DUPLICATE] Already scraped: {url}")
                    continue  # Skip duplicates
                seen_urls.add(url)
            
            results.append(result)
            time.sleep(0.5)  # Small delay between requests
        
        return results


def print_results(results: List[Dict]):
    """Pretty print scraping results"""
    print(f"\n{'='*70}")
    print("WIKIPEDIA SCRAPING SUMMARY")
    print(f"{'='*70}\n")
    
    for i, result in enumerate(results, 1):
        status = result['status']
        title = result.get('title', 'Unknown')
        content_length = result.get('content_length', 0)
        
        if status == 'success':
            print(f"{i}. [SUCCESS] {title}")
            print(f"   Content: {content_length} characters")
            print(f"   URL: {result.get('url', 'N/A')}")
        else:
            print(f"{i}. [{status.upper()}] {title}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print()
    
    print(f"{'='*70}")
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"Total: {successful}/{len(results)} successful")
    print(f"{'='*70}\n")


def main():
    """Main execution - scrape educational topics"""
    
    print("\n" + "="*70)
    print("WIKIPEDIA SCRAPER - EDUCATIONAL RAG")
    print("="*70 + "\n")
    
    # Create scraper
    scraper = WikipediaScraper()
    
    # Topics to scrape - SIMPLE TITLES THAT WORK
    topics = [
        "Machine learning",
        "Deep learning",
        "Artificial intelligence",
        "Supervised learning",
        "Unsupervised learning",
        "Reinforcement learning"
    ]
    
    print(f"Scraping {len(topics)} Wikipedia articles...\n")
    
    # Scrape articles
    results = scraper.get_multiple_articles(topics)
    
    # Print results
    print_results(results)
    
    # Show sample content from first successful article
    for result in results:
        if result['status'] == 'success':
            print(f"\nSAMPLE CONTENT FROM: {result['title']}\n")
            print("-" * 70)
            print(result['content'][:500] + "...\n")
            print("-" * 70)
            break


if __name__ == "__main__":
    main()