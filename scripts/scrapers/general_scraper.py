"""
General Web Scraper for Educational Content
Scrapes text from any website
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List
from datetime import datetime
import sys
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger
from config import RAW_DATA_DIR, SCRAPER_TIMEOUT, SCRAPER_RETRIES, SCRAPER_USER_AGENT

logger = setup_logger(__name__, log_file="scraping.log")


class GeneralWebScraper:
    """Scrape educational content from websites"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': SCRAPER_USER_AGENT
        }
        logger.info("✅ Web Scraper initialized")
    
    def scrape_url(self, url: str) -> Dict:
        """
        Scrape content from a single URL
        
        Args:
            url: Website URL
        
        Returns:
            Dictionary with title, content, source
        """
        try:
            logger.info(f"🌐 Scraping: {url}")
            
            # Make request with retries
            for attempt in range(SCRAPER_RETRIES):
                try:
                    response = requests.get(
                        url, 
                        headers=self.headers, 
                        timeout=SCRAPER_TIMEOUT
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < SCRAPER_RETRIES - 1:
                        logger.warning(f"Retry {attempt + 1}/{SCRAPER_RETRIES}: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get title
            title = soup.title.string if soup.title else "No Title"
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            result = {
                'title': title,
                'url': url,
                'content': text,
                'scraped_at': datetime.now().isoformat(),
                'status': 'success',
                'content_length': len(text)
            }
            
            logger.info(f"✅ Successfully scraped: {title} ({len(text)} chars)")
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error scraping {url}: {e}")
            return {
                'title': 'Error',
                'url': url,
                'content': '',
                'status': 'failed',
                'error': str(e),
                'scraped_at': datetime.now().isoformat()
            }
    
    def scrape_multiple_urls(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs"""
        results = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"\n[{i}/{len(urls)}] Processing: {url}")
            result = self.scrape_url(url)
            results.append(result)
            time.sleep(1)  # Rate limiting
        
        logger.info(f"\n✅ Scraped {len(results)} URLs")
        return results


def main():
    """Example usage"""
    scraper = GeneralWebScraper()
    
    # Example URLs
    urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning",
    ]
    
    # Scrape them
    results = scraper.scrape_multiple_urls(urls)
    
    # Print results
    for result in results:
        print(f"\n{'='*60}")
        print(f"Title: {result['title']}")
        print(f"Status: {result['status']}")
        print(f"Content Length: {result.get('content_length', 0)} chars")
        if result['status'] == 'success':
            content_preview = result['content'][:200] + "..."
            print(f"Content Preview: {content_preview}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()