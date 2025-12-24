# backend/huggingface_integration.py
# HuggingFace Inference API Integration for LLM calls

import os
from huggingface_hub import InferenceClient
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# HUGGINGFACE LLM INTEGRATION
# ============================================================================

class HuggingFaceLLMIntegration:
    """
    Use HuggingFace Inference API for LLM calls
    
    Supports multiple models:
    - Mistral 7B (Fast & Accurate)
    - Zephyr 7B (Optimized for Chat)
    - Intel Neural Chat 7B
    - Llama 2 7B
    - Microsoft Phi 2 (Lightweight)
    
    Get API key from: https://huggingface.co/settings/tokens
    """
    
    def __init__(self, hf_api_key: str = None):
        """
        Initialize HuggingFace client
        
        Args:
            hf_api_key: HuggingFace API token (or from HF_API_TOKEN env var)
        """
        self.hf_api_key = hf_api_key or os.getenv("HF_API_TOKEN")
        
        if not self.hf_api_key:
            logger.warning("⚠️  HF_API_TOKEN not found in environment variables")
            logger.warning("Visit: https://huggingface.co/settings/tokens")
            self.client = None
            self.is_available = False
        else:
            try:
                self.client = InferenceClient(api_key=self.hf_api_key)
                self.is_available = True
                logger.info("✅ HuggingFace client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Error initializing HuggingFace: {e}")
                self.client = None
                self.is_available = False
        
        # Available models
        self.model_options = {
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
            'zephyr': 'HuggingFaceH4/zephyr-7b-beta',
            'neural_chat': 'Intel/neural-chat-7b-v3-1',
            'llama2': 'meta-llama/Llama-2-7b-chat-hf',
            'phi': 'microsoft/phi-2'
        }
        
        logger.info(f"Available models: {list(self.model_options.keys())}")
    
    def generate_answer_hf(self, 
                          query: str, 
                          context: str,
                          model: str = 'mistral',
                          max_tokens: int = 500,
                          temperature: float = 0.7,
                          top_p: float = 0.95) -> Dict:
        """
        Generate answer using HuggingFace model
        
        Args:
            query: User's question
            context: Retrieved document context
            model: Model name ('mistral', 'zephyr', 'phi', etc)
            max_tokens: Maximum tokens in response
            temperature: Creativity level (0.0-1.0)
            top_p: Diversity parameter (0.0-1.0)
        
        Returns:
            Dict with answer, model info, and metadata
        """
        
        # If client not available, use fallback
        if not self.client:
            return self._generate_fallback_answer(query, context, model)
        
        # Get model ID
        model_id = self.model_options.get(model, self.model_options['mistral'])
        
        # Create prompt
        prompt = f"""You are an expert AI assistant. Use the provided context to answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        try:
            start_time = time.time()
            
            # Call HuggingFace API
            response = self.client.text_generation(
                prompt=prompt,
                model=model_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            
            inference_time = time.time() - start_time
            
            logger.info(f"✅ Generated answer using {model} in {inference_time:.2f}s")
            
            return {
                'status': 'success',
                'answer': response,
                'model': model_id,
                'model_name': model,
                'tokens': max_tokens,
                'inference_time': inference_time,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ HuggingFace API error: {e}")
            
            # Return fallback answer if API fails
            return self._generate_fallback_answer(query, context, model, error=str(e))
    
    def summarize_documents(self, 
                           documents: List[str],
                           summary_length: str = 'medium',
                           model: str = 'mistral') -> Dict:
        """
        Summarize multiple documents
        
        Args:
            documents: List of document texts to summarize
            summary_length: 'short' (2-3 sentences), 'medium' (1 paragraph), or 'long' (2-3 paragraphs)
            model: Which model to use
        
        Returns:
            Dict with summary and metadata
        """
        
        if not self.client:
            return {'status': 'error', 'error': 'HuggingFace client not available', 'summary': None}
        
        # Combine documents (limit to first 3)
        combined_text = "\n\n".join(documents[:3])
        
        # Length instructions
        length_instructions = {
            'short': 'in 2-3 sentences',
            'medium': 'in 1 paragraph',
            'long': 'in 2-3 paragraphs'
        }
        
        length_inst = length_instructions.get(summary_length, 'clearly')
        
        # Create prompt
        prompt = f"""Summarize the following text {length_inst}:

{combined_text}

SUMMARY:"""
        
        try:
            model_id = self.model_options.get(model, self.model_options['mistral'])
            
            response = self.client.text_generation(
                prompt=prompt,
                model=model_id,
                max_new_tokens=300,
                temperature=0.5
            )
            
            logger.info(f"✅ Generated summary using {model}")
            
            return {
                'status': 'success',
                'summary': response,
                'summary_length': summary_length,
                'model': model_id,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Summary generation error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'summary': None
            }
    
    def extract_keywords(self, text: str, model: str = 'mistral', num_keywords: int = 5) -> Dict:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            model: Which model to use
            num_keywords: Number of keywords to extract
        
        Returns:
            Dict with keywords
        """
        
        if not self.client:
            return {'status': 'error', 'error': 'Client not available', 'keywords': []}
        
        prompt = f"""Extract {num_keywords} important keywords from the following text:

{text}

KEYWORDS (comma-separated):"""
        
        try:
            model_id = self.model_options.get(model, self.model_options['mistral'])
            
            response = self.client.text_generation(
                prompt=prompt,
                model=model_id,
                max_new_tokens=50,
                temperature=0.3
            )
            
            # Parse keywords
            keywords = [k.strip() for k in response.split(',')]
            
            return {
                'status': 'success',
                'keywords': keywords,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Keyword extraction error: {e}")
            return {'status': 'error', 'error': str(e), 'keywords': []}
    
    def paraphrase_answer(self, answer: str, model: str = 'mistral') -> Dict:
        """
        Paraphrase answer in different style
        
        Args:
            answer: Original answer
            model: Which model to use
        
        Returns:
            Dict with paraphrased answer
        """
        
        if not self.client:
            return {'status': 'error', 'paraphrased': answer}
        
        prompt = f"""Rewrite the following answer in a simpler, more concise way:

{answer}

REWRITTEN:"""
        
        try:
            model_id = self.model_options.get(model, self.model_options['mistral'])
            
            response = self.client.text_generation(
                prompt=prompt,
                model=model_id,
                max_new_tokens=300,
                temperature=0.6
            )
            
            return {
                'status': 'success',
                'paraphrased': response,
                'original': answer,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Paraphrase error: {e}")
            return {'status': 'error', 'paraphrased': answer}
    
    # ========================================================================
    # FALLBACK METHODS (when API is not available)
    # ========================================================================
    
    @staticmethod
    def _generate_fallback_answer(query: str, context: str, model: str = 'unknown', error: str = None) -> Dict:
        """Generate fallback answer if HuggingFace API fails"""
        
        # Extract key phrases from context
        context_lines = context.split('\n')
        key_info = '\n'.join(context_lines[:5])
        
        answer = f"""Based on the retrieved documents:

{key_info}

This information is relevant to your query: "{query}"

Note: This is a fallback response. The main LLM service is temporarily unavailable.

To get a more detailed answer:
1. Check your HuggingFace API token in .env
2. Ensure you have the correct permissions on HuggingFace
3. Review the full documents above
4. Try rephrasing your query"""
        
        status = 'fallback'
        if error:
            logger.warning(f"Using fallback. Error: {error}")
        
        return {
            'status': status,
            'answer': answer,
            'model': f'{model} (fallback)',
            'error': error,
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# UTILITY FUNCTION
# ============================================================================

def test_huggingface_connection(api_key: str = None) -> Dict:
    """
    Test HuggingFace API connection
    
    Args:
        api_key: Optional API key to test
    
    Returns:
        Connection status and available models
    """
    
    llm = HuggingFaceLLMIntegration(api_key)
    
    if not llm.is_available:
        return {
            'status': 'error',
            'message': 'HuggingFace API not available',
            'available_models': []
        }
    
    try:
        # Try a simple test
        response = llm.generate_answer_hf(
            query="What is AI?",
            context="AI is artificial intelligence.",
            max_tokens=50
        )
        
        return {
            'status': 'success',
            'message': 'HuggingFace API connected successfully',
            'test_response': response['answer'],
            'available_models': list(llm.model_options.keys()),
            'response_time': response.get('inference_time', 'N/A')
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Connection test failed: {e}',
            'available_models': list(llm.model_options.keys())
        }

# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 TESTING HUGGINGFACE INTEGRATION")
    print("=" * 80 + "\n")
    
    # Test connection
    from dotenv import load_dotenv
    load_dotenv()
    
    result = test_huggingface_connection()
    
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Available models: {result['available_models']}")
    
    if result['status'] == 'success':
        print(f"\n✅ Connection successful!")
        print(f"Response time: {result.get('response_time', 'N/A')}s")
    else:
        print(f"\n❌ Connection failed!")
    
    print("\n" + "=" * 80 + "\n")