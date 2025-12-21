# Databricks notebook source
# Databricks notebook source

# MAGIC %md
# MAGIC # Discover Actual Function Names in Your Scrapers

print("=" * 80)
print("📦 INSTALLING REQUIRED PACKAGES")
print("=" * 80)

%pip install wikipedia python-dotenv arxiv feedparser requests beautifulsoup4 lxml selenium

print("\n✅ Packages installed! Restarting Python...")
dbutils.library.restartPython()

# MAGIC %md
# MAGIC # Discover Functions (After Restart)

import sys
sys.path.insert(0, '/Workspace/rag_project/scripts')

print("=" * 80)
print("🔍 DISCOVERING ACTUAL FUNCTION NAMES")
print("=" * 80)

# Check Wikipedia
print("\n📖 Wikipedia Scraper:")
try:
    import scrapers.wikipedia_scraper as wiki
    functions = [item for item in dir(wiki) if not item.startswith('_') and callable(getattr(wiki, item))]
    print(f"   Available functions: {functions}")
except Exception as e:
    print(f"   Error: {e}")

# Check arXiv
print("\n📄 arXiv Scraper:")
try:
    import scrapers.arxiv_scraper as arxiv
    functions = [item for item in dir(arxiv) if not item.startswith('_') and callable(getattr(arxiv, item))]
    print(f"   Available functions: {functions}")
except Exception as e:
    print(f"   Error: {e}")

# Check Medium
print("\n✍️ Medium Scraper:")
try:
    import scrapers.medium_scraper as medium
    functions = [item for item in dir(medium) if not item.startswith('_') and callable(getattr(medium, item))]
    print(f"   Available functions: {functions}")
except Exception as e:
    print(f"   Error: {e}")

# Check HuggingFace
print("\n🤗 HuggingFace Scraper:")
try:
    import scrapers.huggingface_scraper as hf
    functions = [item for item in dir(hf) if not item.startswith('_') and callable(getattr(hf, item))]
    print(f"   Available functions: {functions}")
except Exception as e:
    print(f"   Error: {e}")

# Check YouTube
print("\n🎥 YouTube Scraper:")
try:
    import scrapers.youtube_scraper as youtube
    functions = [item for item in dir(youtube) if not item.startswith('_') and callable(getattr(youtube, item))]
    print(f"   Available functions: {functions}")
except Exception as e:
    print(f"   Error: {e}")

# Check General Scraper
print("\n🌐 General Scraper:")
try:
    import scrapers.general_scraper as general
    functions = [item for item in dir(general) if not item.startswith('_') and callable(getattr(general, item))]
    print(f"   Available functions: {functions}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 80)
print("✅ Copy the FUNCTION NAMES from above!")
print("Example: if you see 'main' or 'scrape' or 'run', tell me that!")
print("=" * 80)