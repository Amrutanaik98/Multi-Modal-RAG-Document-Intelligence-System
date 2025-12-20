import requests
from bs4 import BeautifulSoup
from typing import Dict, List
from datetime import datetime
import sys
from pathlib import Path
import json
import time
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger
from config import RAW_DATA_DIR, SCRAPER_TIMEOUT, SCRAPER_USER_AGENT

logger = setup_logger(__name__, log_file="medium_scraping.log")


class MediumScraper:
    """Scrape educational articles from Medium"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': SCRAPER_USER_AGENT
        }
        logger.info("[OK] Medium Scraper initialized")
    
    def scrape_article(self, url: str) -> Dict:
        """
        Scrape a Medium article
        
        Args:
            url: Medium article URL
        
        Returns:
            Dictionary with article content
        """
        try:
            logger.info(f"[SCRAPING] Medium article: {url}")
            
            response = requests.get(
                url,
                headers=self.headers,
                timeout=SCRAPER_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get title
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "No Title"
            
            # Get article content
            article_elem = soup.find('article')
            if article_elem:
                content = article_elem.get_text(separator='\n', strip=True)
            else:
                content = soup.get_text(separator='\n', strip=True)
            
            result = {
                'title': title,
                'url': url,
                'content': content,
                'source': 'medium',
                'scraped_at': datetime.now().isoformat(),
                'status': 'success',
                'content_length': len(content)
            }
            
            logger.info(f"[OK] Scraped: {title} ({len(content)} chars)")
            return result
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to scrape {url}: {e}")
            return {
                'url': url,
                'status': 'failed',
                'error': str(e),
                'scraped_at': datetime.now().isoformat()
            }
    
    def scrape_search_results(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Scrape Medium search results
        Note: Medium has anti-scraping measures, so this is limited
        
        Args:
            query: Search query
            num_results: Number of results to fetch
        
        Returns:
            List of article dictionaries
        """
        try:
            logger.info(f"[SEARCHING] Medium for: {query}")
            
            # Medium search URL
            search_url = f"https://medium.com/search?q={query}"
            
            response = requests.get(
                search_url,
                headers=self.headers,
                timeout=SCRAPER_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            articles = []
            links = soup.find_all('a', limit=num_results)
            
            article_urls = set()
            for link in links:
                href = link.get('href', '')
                if 'medium.com' in href and '/p/' in href:
                    if href not in article_urls:
                        article_urls.add(href)
                        articles.append(href)
                
                if len(articles) >= num_results:
                    break
            
            logger.info(f"[OK] Found {len(articles)} articles for: {query}")
            return articles
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to search Medium: {e}")
            return []
    
    def get_multiple_articles(self, topics: List[str], articles_per_topic: int = 5) -> List[Dict]:
        """
        Search and scrape multiple topics
        
        Args:
            topics: List of search topics
            articles_per_topic: Articles to scrape per topic
        
        Returns:
            List of scraped articles
        """
        all_articles = []
        seen_urls = set()
        
        for topic in topics:
            logger.info(f"\n[TOPIC] Searching: {topic}")
            
            # Find articles for this topic
            article_urls = self.scrape_search_results(topic, articles_per_topic)
            
            for url in article_urls:
                if url not in seen_urls:
                    result = self.scrape_article(url)
                    
                    if result['status'] == 'success':
                        all_articles.append(result)
                        seen_urls.add(url)
                    
                    time.sleep(1)  # Rate limiting
            
            time.sleep(2)
        
        logger.info(f"[TOTAL] Scraped {len(all_articles)} articles")
        return all_articles


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("MEDIUM SCRAPER - EDUCATIONAL RAG")
    print("="*70 + "\n")
    
    scraper = MediumScraper()
    
    # Alternative: Use predefined article URLs
    article_urls = [
        "https://medium.com/towards-data-science/machine-learning-basics-1eee2d7f1697",
        "https://medium.com/towards-data-science/the-complete-guide-to-deep-learning-1e100b9dbb78",
    ]
    
    articles = []
    for url in article_urls:
        result = scraper.scrape_article(url)
        articles.append(result)
        time.sleep(1)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Total Articles Scraped: {len(articles)}")
    for article in articles[:3]:
        if article['status'] == 'success':
            print(f"\n  • {article['title'][:60]}")
    print(f"{'='*70}\n")
    
    # =====================================================
    # NEW: SAVE TO DATABRICKS INSTEAD OF JSON
    # =====================================================
    
    try:
        # Prepare data for Databricks
        df_data = []
        successful_count = 0
        
        for article in articles:
            if article['status'] == 'success':
                df_data.append({
                    'document_id': article.get('url', '').replace('/', '_')[:100],
                    'title': article.get('title', '')[:255],
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': 'medium',
                    'scraped_date': datetime.now(),
                    'content_length': article.get('content_length', 0),
                    'created_at': datetime.now()
                })
                successful_count += 1
        
        # Save to Databricks
        if df_data:
            df = pd.DataFrame(df_data)
            spark.createDataFrame(df).write.mode("append").saveAsTable("raw_documents")
            
            print(f"✅ Saved {successful_count} Medium articles to Databricks")
            logger.info(f"[OK] Saved {successful_count} Medium articles to Databricks")
        
    except Exception as e:
        print(f"❌ Error saving to Databricks: {e}")
        logger.error(f"[ERROR] Failed to save to Databricks: {e}")
        raise


if __name__ == "__main__":
    main()