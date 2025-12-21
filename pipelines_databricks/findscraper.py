# Databricks notebook source
# Databricks notebook source

# MAGIC %md
# MAGIC # Step 1: Install All Dependencies First

%pip install wikipedia python-dotenv arxiv feedparser requests beautifulsoup4 lxml selenium sentence-transformers torch pinecone-client youtube-transcript-api

# MAGIC %md
# MAGIC # Step 2: Restart Python and Inspect Scrapers

import sys
sys.path.insert(0, '/Workspace/rag_project/scripts')

print("=" * 80)
print("🔍 INSPECTING YOUR ACTUAL SCRAPER METHODS")
print("=" * 80)

import inspect

# Wikipedia
print("\n📖 WikipediaScraper:")
print("   " + "-" * 50)
try:
    from scrapers.wikipedia_scraper import WikipediaScraper
    wiki = WikipediaScraper()
    
    # Get all methods (excluding private/magic)
    all_attrs = [attr for attr in dir(wiki) if not attr.startswith('_')]
    
    # Filter to only callable methods
    methods = [m for m in all_attrs if callable(getattr(wiki, m))]
    
    print(f"   All non-private attributes: {all_attrs}")
    print(f"   Callable methods: {methods}")
    
    # Try to get the main scraping method
    for method_name in ['run', 'scrape', 'execute', 'fetch', 'get_documents', 'start', 'main']:
        if hasattr(wiki, method_name):
            print(f"   ✅ FOUND METHOD: {method_name}()")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# arXiv
print("\n📄 ArxivScraper:")
print("   " + "-" * 50)
try:
    from scrapers.arxiv_scraper import ArxivScraper
    arxiv = ArxivScraper()
    
    all_attrs = [attr for attr in dir(arxiv) if not attr.startswith('_')]
    methods = [m for m in all_attrs if callable(getattr(arxiv, m))]
    
    print(f"   All non-private attributes: {all_attrs}")
    print(f"   Callable methods: {methods}")
    
    for method_name in ['run', 'scrape', 'execute', 'fetch', 'get_documents', 'start', 'main']:
        if hasattr(arxiv, method_name):
            print(f"   ✅ FOUND METHOD: {method_name}()")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# Medium
print("\n✍️ MediumScraper:")
print("   " + "-" * 50)
try:
    from scrapers.medium_scraper import MediumScraper
    medium = MediumScraper()
    
    all_attrs = [attr for attr in dir(medium) if not attr.startswith('_')]
    methods = [m for m in all_attrs if callable(getattr(medium, m))]
    
    print(f"   All non-private attributes: {all_attrs}")
    print(f"   Callable methods: {methods}")
    
    for method_name in ['run', 'scrape', 'execute', 'fetch', 'get_documents', 'start', 'main']:
        if hasattr(medium, method_name):
            print(f"   ✅ FOUND METHOD: {method_name}()")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# HuggingFace
print("\n🤗 HuggingFaceScraper:")
print("   " + "-" * 50)
try:
    from scrapers.huggingface_scraper import HuggingFaceScraper
    hf = HuggingFaceScraper()
    
    all_attrs = [attr for attr in dir(hf) if not attr.startswith('_')]
    methods = [m for m in all_attrs if callable(getattr(hf, m))]
    
    print(f"   All non-private attributes: {all_attrs}")
    print(f"   Callable methods: {methods}")
    
    for method_name in ['run', 'scrape', 'execute', 'fetch', 'get_documents', 'start', 'main']:
        if hasattr(hf, method_name):
            print(f"   ✅ FOUND METHOD: {method_name}()")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# YouTube
print("\n🎥 YouTubeTranscriptScraper:")
print("   " + "-" * 50)
try:
    from scrapers.youtube_scraper import YouTubeTranscriptScraper
    yt = YouTubeTranscriptScraper()
    
    all_attrs = [attr for attr in dir(yt) if not attr.startswith('_')]
    methods = [m for m in all_attrs if callable(getattr(yt, m))]
    
    print(f"   All non-private attributes: {all_attrs}")
    print(f"   Callable methods: {methods}")
    
    for method_name in ['run', 'scrape', 'execute', 'fetch', 'get_documents', 'start', 'main']:
        if hasattr(yt, method_name):
            print(f"   ✅ FOUND METHOD: {method_name}()")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ LOOK AT THE '✅ FOUND METHOD:' LINES ABOVE")
print("Tell me what methods were found for each scraper!")
print("=" * 80)