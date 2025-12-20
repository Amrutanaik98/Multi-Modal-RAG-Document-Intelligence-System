"""
YouTube Transcript Scraper
Scrapes transcripts from educational YouTube videos
"""

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
from config import RAW_DATA_DIR

logger = setup_logger(__name__, log_file="youtube_scraping.log")


class YouTubeTranscriptScraper:
    """Scrape transcripts from YouTube videos"""
    
    def __init__(self):
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            self.api = YouTubeTranscriptApi
            logger.info("[OK] YouTube Transcript Scraper initialized")
        except ImportError:
            logger.error("[ERROR] youtube-transcript-api not installed")
            logger.info("[INFO] Install with: pip install youtube-transcript-api")
            raise
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        import re
        
        # Handle different YouTube URL formats
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def get_transcript(self, video_url: str, title: str = None) -> Dict:
        """
        Get transcript from a YouTube video
        
        Args:
            video_url: YouTube video URL
            title: Video title (optional)
        
        Returns:
            Dictionary with transcript
        """
        try:
            video_id = self.extract_video_id(video_url)
            if not video_id:
                raise ValueError(f"Invalid YouTube URL: {video_url}")
            
            logger.info(f"[FETCHING] YouTube transcript: {video_url}")
            
            # Get transcript
            transcript_list = self.api.get_transcript(video_id)
            
            # Combine transcript entries
            full_transcript = ' '.join([item['text'] for item in transcript_list])
            
            result = {
                'title': title or f"YouTube Video: {video_id}",
                'url': video_url,
                'video_id': video_id,
                'content': full_transcript,
                'source': 'youtube',
                'scraped_at': datetime.now().isoformat(),
                'status': 'success',
                'content_length': len(full_transcript)
            }
            
            logger.info(f"[OK] Got transcript: {len(full_transcript)} chars")
            return result
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to get transcript: {e}")
            return {
                'url': video_url,
                'status': 'failed',
                'error': str(e),
                'scraped_at': datetime.now().isoformat()
            }
    
    def get_multiple_transcripts(self, video_urls: List[tuple]) -> List[Dict]:
        """
        Get transcripts from multiple videos
        
        Args:
            video_urls: List of (url, title) tuples
        
        Returns:
            List of transcripts
        """
        transcripts = []
        
        for i, (url, title) in enumerate(video_urls, 1):
            logger.info(f"\n[{i}/{len(video_urls)}] Processing: {title}")
            result = self.get_transcript(url, title)
            transcripts.append(result)
            time.sleep(1)  # Rate limiting
        
        logger.info(f"[TOTAL] Got {len(transcripts)} transcripts")
        return transcripts


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("YOUTUBE TRANSCRIPT SCRAPER - EDUCATIONAL RAG")
    print("="*70 + "\n")
    
    try:
        scraper = YouTubeTranscriptScraper()
        
        # Educational video URLs with titles
        videos = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "Sample Video 1"),
            # Add more video URLs here
        ]
        
        if not videos or len(videos) == 1:
            print("[INFO] No videos configured. Add YouTube URLs to the 'videos' list.")
            print("[INFO] Example: https://www.youtube.com/watch?v=VIDEO_ID")
            print("[INFO] Make sure videos have transcripts available.\n")
            return
        
        print(f"Scraping {len(videos)} YouTube videos...\n")
        
        # Scrape transcripts
        transcripts = scraper.get_multiple_transcripts(videos)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Total Videos Processed: {len(transcripts)}")
        for transcript in transcripts[:3]:
            if transcript['status'] == 'success':
                print(f"\n  • {transcript['title']}")
                print(f"    Transcript: {len(transcript['content'])} chars")
        print(f"{'='*70}\n")
        
        # =====================================================
        # NEW: SAVE TO DATABRICKS INSTEAD OF JSON
        # =====================================================
        
        try:
            # Prepare data for Databricks
            df_data = []
            successful_count = 0
            
            for transcript in transcripts:
                if transcript['status'] == 'success':
                    df_data.append({
                        'document_id': transcript.get('video_id', '')[:100],
                        'title': transcript.get('title', '')[:255],
                        'content': transcript.get('content', ''),
                        'url': transcript.get('url', ''),
                        'source': 'youtube',
                        'scraped_date': datetime.now(),
                        'content_length': transcript.get('content_length', 0),
                        'created_at': datetime.now()
                    })
                    successful_count += 1
            
            # Save to Databricks
            if df_data:
                df = pd.DataFrame(df_data)
                spark.createDataFrame(df).write.mode("append").saveAsTable("raw_documents")
                
                print(f"✅ Saved {successful_count} YouTube transcripts to Databricks")
                logger.info(f"[OK] Saved {successful_count} YouTube transcripts to Databricks")
            
        except Exception as e:
            print(f"❌ Error saving to Databricks: {e}")
            logger.error(f"[ERROR] Failed to save to Databricks: {e}")
            raise
    
    except ImportError:
        print("[ERROR] youtube-transcript-api is not installed")
        print("[INFO] Install it with:")
        print("       pip install youtube-transcript-api")
        print("\n[INFO] Then run this script again.\n")


if __name__ == "__main__":
    main()