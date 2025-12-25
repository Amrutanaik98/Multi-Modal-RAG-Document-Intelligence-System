"""
RAG Intelligence System - Streamlit Frontend
Fixed UI with proper contrast and visibility
"""

import streamlit as st
import requests
from datetime import datetime
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🧠 RAG Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - FIXED VISIBILITY
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
        background-color: #ffffff;
    }
    
    /* Sidebar - DARK BACKGROUND WITH LIGHT TEXT */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    
    /* Sidebar text - LIGHT AND VISIBLE */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Sidebar labels */
    [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    /* Sidebar input fields */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
    }
    
    /* Slider text in sidebar */
    [data-testid="stSidebar"] .stSlider {
        color: #ffffff !important;
    }
    
    /* Expander headers in sidebar */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border-left: 4px solid #667eea !important;
    }
    
    /* Main title */
    .title-container {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .title-container h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 800;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
        font-weight: 700;
        font-size: 1.1em;
    }
    
    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #e7f3ff 0%, #d9e7ff 100%);
        border-left: 6px solid #2196F3;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #000;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
    }
    
    /* Chunk container */
    .chunk-container {
        background: #f8f9fa;
        border-left: 6px solid #667eea;
        padding: 18px;
        margin: 12px 0;
        border-radius: 8px;
        color: #2d3748;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .chunk-container:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    /* Status indicators */
    .status-connected {
        padding: 15px 20px;
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        color: #155724;
        font-weight: 600;
    }
    
    .status-disconnected {
        padding: 15px 20px;
        background: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 8px;
        color: #721c24;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Text area */
    .stTextArea textarea {
        font-size: 15px;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE
# ============================================================================

st.markdown("""
<div class="title-container">
    <h1>🧠 RAG Intelligence System</h1>
    <p>Semantic AI-Powered Document Search & Question Answering</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ CONFIGURATION")
    
    # API Settings
    with st.expander("🔗 API Settings", expanded=True):
        api_url = st.text_input(
            "Backend URL",
            value="http://localhost:8000",
            help="FastAPI backend URL"
        )
    
    st.divider()
    
    # Search Settings
    with st.expander("🔍 Search Settings", expanded=True):
        top_k = st.slider(
            "Results to Retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of documents to fetch"
        )
        
        llm_model = st.selectbox(
            "LLM Model",
            options=["mistral", "zephyr", "phi", "neural_chat", "llama2"],
            help="Choose language model"
        )
        
        max_tokens = st.slider(
            "Answer Length",
            min_value=100,
            max_value=1000,
            value=500,
            step=100,
            help="Max response tokens"
        )
    
    st.divider()
    
    # System Info
    with st.expander("📊 System Info", expanded=False):
        st.info("""
        **RAG System Features:**
        - 📚 Semantic search
        - 🧠 AI-powered answers
        - 🎯 High accuracy
        - ⚡ Fast retrieval
        """)
    
    st.divider()
    st.caption("**Version:** 5.0.0 | **2025**")

# ============================================================================
# HEALTH CHECK
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    with st.spinner("🔄 Checking backend..."):
        try:
            response = requests.get(f"{api_url}/health", timeout=3)
            if response.status_code == 200:
                data = response.json()
                st.markdown(f"""
                <div class="status-connected">
                ✅ Connected | {data.get('documents', 0)} docs loaded
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-disconnected">
                ❌ Backend Error: {response.status_code}
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="status-disconnected">
            ❌ Backend Offline
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# QUERY SECTION
# ============================================================================

st.markdown('<div class="section-header">🔍 Ask Your Question</div>', unsafe_allow_html=True)

user_query = st.text_area(
    "Enter your question:",
    placeholder="Example: What is machine learning? How do transformers work?",
    height=100,
    label_visibility="collapsed"
)

col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

with col2:
    submit_button = st.button("🔍 Search", use_container_width=True, type="primary")

with col3:
    clear_button = st.button("🔄 Clear", use_container_width=True)

with col4:
    example_button = st.button("💡 Example", use_container_width=True)

if clear_button:
    st.rerun()

if example_button:
    st.session_state.query = "What is machine learning?"
    user_query = "What is machine learning?"
    st.rerun()

# ============================================================================
# RESULTS
# ============================================================================

if submit_button:
    if not user_query or len(user_query.strip()) == 0:
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("🔄 Searching and generating answer..."):
            try:
                response = requests.post(
                    f"{api_url}/query",
                    json={
                        "query": user_query,
                        "top_k": top_k,
                        "llm_model": llm_model,
                        "max_answer_tokens": max_tokens
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Answer
                    st.markdown('<div class="section-header">📖 AI Answer</div>', 
                               unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="answer-box">{result.get("answer", "No answer generated")}</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.divider()
                    
                    # Retrieved Documents
                    st.markdown('<div class="section-header">📚 Retrieved Documents</div>', 
                               unsafe_allow_html=True)
                    
                    chunks = result.get('retrieved_chunks', [])
                    
                    if chunks:
                        for i, chunk in enumerate(chunks, 1):
                            score = chunk.get('similarity_score', 0) * 100
                            
                            emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔵"
                            
                            with st.expander(
                                f"{emoji} Source {i} - {chunk.get('document_id', 'Unknown')[:30]}... ({score:.1f}%)",
                                expanded=(i == 1)
                            ):
                                st.markdown(
                                    f'<div class="chunk-container">{chunk.get("chunk_text", "No text")}</div>',
                                    unsafe_allow_html=True
                                )
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.caption(f"📌 {chunk.get('chunk_id', 'N/A')}")
                                with col2:
                                    st.caption(f"🏷️ {chunk.get('topic', 'unknown')}")
                                with col3:
                                    st.caption(f"⭐ {score:.1f}%")
                    
                    st.divider()
                    
                    # Statistics
                    st.markdown('<div class="section-header">📊 Statistics</div>', 
                               unsafe_allow_html=True)
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric("📚 Retrieved", len(chunks))
                    
                    with stat_col2:
                        best = chunks[0]['similarity_score'] * 100 if chunks else 0
                        st.metric("🎯 Best Match", f"{best:.1f}%")
                    
                    with stat_col3:
                        avg = result.get('avg_similarity', 0) * 100
                        st.metric("📈 Avg Match", f"{avg:.1f}%")
                    
                    with stat_col4:
                        time_taken = result.get('response_time', 0)
                        st.metric("⏱️ Time", f"{time_taken:.2f}s")
                
                else:
                    st.error(f"❌ API Error: {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend")
                st.info(f"Start backend: `python backend/fastapi_backend.py`")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# EXAMPLES
# ============================================================================

st.divider()

st.markdown('<div class="section-header">💡 Example Questions</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

examples = [
    "What is machine learning?",
    "Explain transformers in NLP",
    "What is RAG?"
]

for col, example in zip([col1, col2, col3], examples):
    with col:
        if st.button(example, use_container_width=True):
            user_query = example
            st.rerun()

st.divider()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
---
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>RAG Intelligence System v5.0</strong></p>
    <p>Streamlit + FastAPI + Embeddings + LLM</p>
    <p style="font-size: 12px; margin-top: 10px;">© 2025 | Retrieval-Augmented Generation</p>
</div>
""", unsafe_allow_html=True)