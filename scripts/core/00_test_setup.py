"""
Test script to verify setup is correct
Run this first to make sure all libraries are installed
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger

logger = setup_logger(__name__, log_file="test_setup.log")

def test_imports():
    """Test that all critical libraries can be imported"""
    
    tests = [
        ("pdf2image", "PDF to Image conversion"),
        ("pytesseract", "Tesseract OCR wrapper"),
        ("cv2", "OpenCV for image processing"),
        ("PIL", "Pillow image library"),
        ("pandas", "Data processing"),
        ("numpy", "Numerical operations"),
        ("dotenv", "Environment variable loading"),
        ("torch", "PyTorch (deep learning)"),
    ]
    
    passed = 0
    failed = 0
    
    logger.info("=" * 60)
    logger.info("Testing Library Imports...")
    logger.info("=" * 60)
    
    for module_name, description in tests:
        try:
            __import__(module_name)
            logger.info(f"✅ {module_name:20s} - {description}")
            passed += 1
        except ImportError as e:
            logger.error(f"❌ {module_name:20s} - {description}")
            logger.error(f"   Error: {e}")
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    return failed == 0

def test_paths():
    """Test that directory structure is correct"""
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR
    
    logger.info("\nTesting Directory Structure...")
    logger.info("=" * 60)
    
    paths_to_check = [
        (RAW_DATA_DIR, "Raw data"),
        (PROCESSED_DATA_DIR, "Processed data"),
        (OUTPUT_DIR, "Output"),
    ]
    
    for path, name in paths_to_check:
        if path.exists():
            logger.info(f"✅ {name:20s} - {path}")
        else:
            logger.error(f"❌ {name:20s} - Missing: {path}")
    
    logger.info("=" * 60)

def test_env():
    """Test that .env file is loaded"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logger.info("\nTesting Environment Variables...")
    logger.info("=" * 60)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if openai_key and openai_key != "your_key_here":
        logger.info("✅ OPENAI_API_KEY is set")
    else:
        logger.warning("⚠️  OPENAI_API_KEY not configured in .env")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        imports_ok = test_imports()
        test_paths()
        test_env()
        
        if imports_ok:
            logger.info("\n🎉 All tests passed! Ready to start working.")
        else:
            logger.error("\n⚠️  Some imports failed. Install missing packages:")
            logger.error("   pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)