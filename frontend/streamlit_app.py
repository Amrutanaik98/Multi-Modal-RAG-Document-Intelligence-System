# frontend/streamlit_app.py
# RAG Intelligence System - Streamlit Frontend

import streamlit as st
import requests
from datetime import datetime
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="RAG Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - MODERN DESIGN
# ============================================================================

st.markdown("""
<style>
    /* Main Container */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Title Styling */
    .title-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Query Input Box */
    .query-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        border-left: 5px solid #667eea;
    }
    
    /* Answer Box */
    .answer-box {
        background: linear-gradient(135deg, #e7f3ff 0%, #d9e7ff 100%);
        border-left: 5px solid #2196F3;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        color: #000;
        font-size: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Chunk Container */
    .chunk-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #667eea;
        padding: 18px;
        margin: 12px 0;
        border-radius: 8px;
        color: #1a1a1a;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    
    .chunk-container:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    /* Status Indicators */
    .status-connected {
        padding: 12px;
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 6px;
        color: #155724;
        font-weight: 500;
    }
    
    .status-disconnected {
        padding: 12px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 6px;
        color: #721c24;
        font-weight: 500;
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #667eea, transparent);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Success Message */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-radius: 8px;
    }
    
    /* Error Message */
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border-radius: 8px;
    }
    
    /* Header Text */
    h1, h2, h3 {
        color: #2d3748;
        font-weight: 700;
    }
    
    /* Subheader */
    [data-testid="stMarkdownContainer"] h2 {
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE & HEADER
# ============================================================================

st.markdown("""
<div class="title-container">
    <h1>🧠 RAG Intelligence System</h1>
    <p style="font-size: 18px; margin: 10px 0;">Semantic AI-Powered Document Search & Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.markdown("**Powered by:** Sentence Transformers + HuggingFace | **Type:** Retrieval-Augmented Generation")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Configuration
    st.subheader("🔗 API Settings")
    api_url = st.text_input(
        "FastAPI Backend URL",
        value="http://localhost:8000",
        help="URL where your FastAPI backend is running"
    )
    
    st.divider()
    
    # Query Settings
    st.subheader("🔍 Search Settings")
    top_k = st.slider(
        "Results to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of documents to fetch (higher = more context)"
    )
    
    # LLM Model Selection
    llm_model = st.selectbox(
        "LLM Model",
        options=["mistral", "zephyr", "phi", "neural_chat", "llama2"],
        help="Choose which LLM model to use"
    )
    
    st.divider()
    
    # System Info
    st.subheader("ℹ️ About This System")
    st.info("""
    **RAG Document Query System**
    
    ✨ **Features:**
    - 📚 Semantic search using embeddings
    - 🧠 AI-powered question answering
    - 🎯 High accuracy retrieval
    - ⚡ Real-time responses
    
    **Architecture:**
    - Frontend: Streamlit
    - Backend: FastAPI
    - Models: Sentence Transformers + HuggingFace
    """)
    
    st.divider()
    
    st.markdown("**Version:** 4.0.0 | **Last Updated:** 2025")

# ============================================================================
# CHECK API HEALTH
# ============================================================================

st.divider()

st.subheader("📡 System Status", divider="rainbow")

try:
    health_response = requests.get(f"{api_url}/health", timeout=3)
    if health_response.status_code == 200:
        health_data = health_response.json()
        
        # Display Status Cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📚 Documents Loaded", health_data.get('documents_loaded', 'N/A'))
        
        with col2:
            st.metric("🧠 Embedding Dim", 
                     health_data.get('embedding_dimension', 384))
        
        with col3:
            st.metric("✅ Status", "Connected")
        
        st.markdown("""<div class="status-connected">✅ Successfully connected to FastAPI backend!</div>""", 
                   unsafe_allow_html=True)
    else:
        st.error(f"❌ API Error: Status code {health_response.status_code}")

except requests.exceptions.ConnectionError:
    st.markdown("""<div class="status-disconnected">❌ Cannot connect to FastAPI backend</div>""", 
               unsafe_allow_html=True)
    st.error(f"**Backend not running at: {api_url}**")
    st.info("""
    **How to start the backend:**
    ```bash
    python backend/fastapi_backend.py
    ```
    Then wait for: `INFO: Uvicorn running on http://0.0.0.0:8000`
    """)

except Exception as e:
    st.error(f"Error: {str(e)}")

st.divider()

# ============================================================================
# QUERY SECTION
# ============================================================================

st.subheader("🔍 Ask Your Question", divider="blue")

user_query = st.text_area(
    "Enter your question about the documents:",
    placeholder="Example: What is machine learning? How do transformers work? What is RAG?",
    height=120,
    label_visibility="collapsed"
)

# Search button
col1, col2, col3 = st.columns([2, 1, 1])

with col2:
    submit_button = st.button("🔍 Search", use_container_width=True, type="primary")

with col3:
    clear_button = st.button("🔄 Clear", use_container_width=True)

if clear_button:
    st.rerun()

# ============================================================================
# RESULTS SECTION
# ============================================================================

if submit_button:
    if not user_query or len(user_query.strip()) == 0:
        st.warning("⚠️ Please enter a question to proceed")
    else:
        with st.spinner("🔄 Searching documents and generating answer..."):
            try:
                # Make request to FastAPI backend
                response = requests.post(
                    f"{api_url}/query",
                    json={
                        "query": user_query,
                        "top_k": top_k,
                        "llm_model": llm_model,
                        "include_summary": False,
                        "max_answer_tokens": 500
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display Answer Section
                    st.subheader("📖 AI-Generated Answer", divider="green")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', 
                               unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Display Retrieved Documents
                    st.subheader("📚 Retrieved Documents", divider="orange")
                    
                    retrieved_chunks = result.get('retrieved_chunks', [])
                    
                    if retrieved_chunks:
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            similarity_pct = chunk['similarity_score'] * 100
                            
                            # Color coding based on similarity
                            if similarity_pct >= 80:
                                color = "🟢"
                            elif similarity_pct >= 60:
                                color = "🟡"
                            else:
                                color = "🔵"
                            
                            with st.expander(
                                f"{color} Document {i} - Match: {similarity_pct:.1f}%",
                                expanded=(i == 1)
                            ):
                                st.markdown(f'<div class="chunk-container">{chunk["chunk_text"]}</div>', 
                                          unsafe_allow_html=True)
                                
                                col_meta1, col_meta2, col_meta3 = st.columns(3)
                                with col_meta1:
                                    st.caption(f"📌 ID: {chunk['chunk_id']}")
                                with col_meta2:
                                    st.caption(f"📄 Doc: {chunk['document_id']}")
                                with col_meta3:
                                    st.caption(f"⭐ Score: {chunk['similarity_score']:.2%}")
                    else:
                        st.warning("No relevant documents found")
                    
                    st.divider()
                    
                    # Display Statistics
                    st.subheader("📊 Retrieval Statistics", divider="violet")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "📚 Retrieved",
                            len(retrieved_chunks)
                        )
                    
                    with col2:
                        if retrieved_chunks:
                            best_match = retrieved_chunks[0]['similarity_score'] * 100
                            st.metric(
                                "🎯 Best Match",
                                f"{best_match:.1f}%"
                            )
                        else:
                            st.metric("🎯 Best Match", "N/A")
                    
                    with col3:
                        avg_sim = result.get('avg_similarity', 0) * 100
                        st.metric(
                            "📈 Avg Match",
                            f"{avg_sim:.1f}%"
                        )
                    
                    with col4:
                        response_time = result.get('response_time', 0)
                        st.metric(
                            "⏱️ Response Time",
                            f"{response_time:.2f}s"
                        )
                
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.code(response.text, language="json")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to FastAPI backend")
                st.info("""
                **To start the backend:**
                ```bash
                cd backend
                python fastapi_backend.py
                ```
                """)
            
            except requests.exceptions.Timeout:
                st.error("❌ Request timeout - backend took too long")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.divider()

# ============================================================================
# EXAMPLE QUERIES
# ============================================================================

st.subheader("💡 Try These Example Queries", divider="red")

col1, col2, col3 = st.columns(3)

examples = [
    ("What is machine learning?", "q1"),
    ("Explain transformers in NLP", "q2"),
    ("What is RAG?", "q3"),
]

for col, (query_text, key) in zip([col1, col2, col3], examples):
    with col:
        if st.button(query_text, use_container_width=True, key=key):
            st.session_state.user_query = query_text
            st.rerun()

st.divider()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
---
<div style='text-align: center; color: #666; padding: 20px 0;'>
    <h4>🚀 RAG Intelligence System v4.0</h4>
    <p>Built with <b>Streamlit</b> | Powered by <b>FastAPI</b> & <b>HuggingFace</b></p>
    
    <details>
    <summary><b>📋 System Architecture</b></summary>
    <p>
    Streamlit Frontend (Port 8501) ↓<br>
    HTTP POST /query ↓<br>
    FastAPI Backend (Port 8000) ↓<br>
    SentenceTransformer + HuggingFace LLM ↓<br>
    Answer + Relevant Sources
    </p>
    </details>
    
    <p style='font-size: 12px; margin-top: 10px;'>© 2025 | Retrieval-Augmented Generation System</p>
</div>
""", unsafe_allow_html=True)