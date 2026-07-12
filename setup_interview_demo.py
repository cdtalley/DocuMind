#!/usr/bin/env python3
"""
Quick Setup Script for RAG Interview Demo

This script helps prepare the system for the interview demonstration by:
1. Checking system requirements
2. Verifying dependencies
3. Setting up basic configuration
4. Running a quick system test
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required. Current version: %s", sys.version)
        return False
    logger.info("✅ Python version: %s", sys.version.split()[0])
    return True


def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        'fastapi',
        'uvicorn', 
        'pydantic',
        'chromadb',
        'langchain_core',
        'langchain_text_splitters',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info("✅ %s is available", package)
        except ImportError:
            missing_packages.append(package)
            logger.warning("❌ %s is missing", package)
    
    if missing_packages:
        logger.error("Missing packages: %s", missing_packages)
        logger.info("Install with: pip install %s", ' '.join(missing_packages))
        return False
    
    return True


def check_file_structure():
    """Check if advanced RAG files are in place"""
    required_files = [
        'app/services/enhanced_rag_service.py',
        'app/services/agentic_rag.py', 
        'app/services/cross_encoder_reranker.py',
        'app/services/rag_evaluation.py',
        'app/services/rag_monitoring.py',
        'app/services/query_router.py',
        'app/services/semantic_chunker.py',
        'interview_rag_demo.py',
        'ADVANCED_RAG_FEATURES.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info("✅ %s exists", file_path)
        else:
            missing_files.append(file_path)
            logger.warning("❌ %s missing", file_path)
    
    if missing_files:
        logger.error("Missing files: %s", missing_files)
        return False
    
    return True


def create_basic_config():
    """Create basic configuration if needed"""
    env_file = Path('.env')
    
    if not env_file.exists():
        logger.info("Creating basic .env configuration...")
        
        basic_config = """
# Basic configuration for RAG demo
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3
EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./chroma_db
DEFAULT_LIBRARY=public
LOG_LEVEL=INFO

# Advanced features
ENABLE_ROUTING_LEARNING=true
ENABLE_RERANK_CACHING=true
MAX_RERANK_CANDIDATES=20
RERANK_TOP_K=10
""".strip()
        
        with open('.env', 'w') as f:
            f.write(basic_config)
        
        logger.info("✅ Created .env configuration")
    else:
        logger.info("✅ .env configuration exists")


def test_basic_imports():
    """Test basic imports to verify system readiness"""
    try:
        logger.info("Testing advanced RAG service imports...")
        
        # Test core imports
        from app.config import get_settings
        from app.utils.ollama_client import OllamaClient
        
        # Test advanced service imports
        from app.services.enhanced_rag_service import EnhancedRAGService
        from app.services.agentic_rag import AgenticRAGService
        from app.services.cross_encoder_reranker import RerankingService
        from app.services.rag_evaluation import RAGEvaluator
        from app.services.rag_monitoring import RAGMonitoringDashboard
        from app.services.query_router import IntelligentQueryRouter
        from app.services.semantic_chunker import SemanticChunkingService
        
        logger.info("✅ All advanced RAG services import successfully")
        return True
        
    except ImportError as e:
        logger.error("❌ Import failed: %s", e)
        return False
    except Exception as e:
        logger.error("❌ Unexpected error: %s", e)
        return False


def check_ollama_status():
    """Check if Ollama is available (optional)"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            logger.info("✅ Ollama is running and accessible")
            return True
        else:
            logger.warning("⚠️  Ollama responded with status: %s", response.status_code)
            return False
    except requests.RequestException:
        logger.warning("⚠️  Ollama is not running (demo will use simulated responses)")
        return False
    except ImportError:
        logger.warning("⚠️  requests library not available, cannot check Ollama")
        return False


def main():
    """Main setup function"""
    print("🚀 Setting up Advanced RAG System for Interview Demo")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 0
    
    # Required checks
    required_checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies), 
        ("File Structure", check_file_structure),
        ("Basic Imports", test_basic_imports)
    ]
    
    for name, check_func in required_checks:
        total_checks += 1
        print(f"\n🔍 Checking {name}...")
        
        if check_func():
            checks_passed += 1
        else:
            print(f"❌ {name} check failed!")
    
    # Optional setup
    print(f"\n🔧 Setting up configuration...")
    create_basic_config()
    
    # Optional checks
    print(f"\n🔍 Checking optional services...")
    check_ollama_status()
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 Setup Summary: {checks_passed}/{total_checks} required checks passed")
    
    if checks_passed == total_checks:
        print("✅ System is ready for interview demo!")
        print("\n🎯 Next steps:")
        print("1. Run the demo: python interview_rag_demo.py")
        print("2. Start the API server: python -m uvicorn app.main:app --reload")
        print("3. Review the features: cat ADVANCED_RAG_FEATURES.md")
        
        return True
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)