import streamlit as st
import json
import pickle
import numpy as np
import random
import time
import pandas as pd


st.set_page_config(
    page_title="Chatbot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
   
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f1016 0%, #090a0f 90%);
        color: #e0e0e0;
    }
    
    
    .stApp::before {
        content: '';
        position: absolute;
        top: -10%;
        left: -10%;
        width: 40%;
        height: 40%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        filter: blur(100px);
        z-index: 0;
        animation: floatOrb 10s infinite alternate ease-in-out;
    }
    
    @keyframes floatOrb {
        0% { transform: translate(0, 0); }
        100% { transform: translate(100px, 50px); }
    }

   
    [data-testid="stSidebar"] {
        background: rgba(13, 15, 23, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
    }
    
    .sidebar-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #fff;
        margin-bottom: 20px;
        margin-top: -20px; 
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -1px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .sidebar-header span {
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e2e8f0;
        line-height: 1.2;
    }
    
    .metric-sub {
        font-size: 0.75rem;
        color: #10b981;
        margin-left: 5px;
    }

   
    /* Hide default elements */
    /* Hide default elements (Removed to fix sidebar toggle) */
    /* header, footer { visibility: hidden !important; } */
    footer { visibility: hidden !important; }
    
    /* Input Container */
    .stChatInputContainer {
        padding-bottom: 40px;
        background: linear-gradient(to top, #090a0f 0%, transparent 100%);
    }
    
    [data-testid="stChatInput"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: #fff !important;
        backdrop-filter: blur(10px);
        padding: 0.8rem 1.2rem !important; 
    }
    
    [data-testid="stChatInput"]:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2) !important;
    }
    
    /* Welcome Title */
    .welcome-text {
        text-align: center;
        margin-top: 15vh;
        animation: fadeIn 0.8s ease-out;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 300;
    }
    
     /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Typewriter Cursor */
    .cursor {
        display: inline-block;
        width: 3px;
        background-color: #818cf8;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        50% { opacity: 0; }
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)


# BACKEND RESOURCES

@st.cache_resource
def load_resources():
    try:
        with open("sklearn_model.pickle", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pickle", "rb") as f:
            vectorizer = pickle.load(f)
        with open("label_encoder.pickle", "rb") as f:
            label_encoder = pickle.load(f)
        # Version 2
        with open("intents.json") as f:
            intents = json.load(f)
        return model, vectorizer, label_encoder, intents
    except:
        return None, None, None, None

model, vectorizer, label_encoder, intents = load_resources()

def predict(text):
    if not model: return None, 0.0
    vec = vectorizer.transform([text.lower()])
    idx = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    tag = label_encoder.inverse_transform([idx])[0]
    return tag, prob

def get_reply(tag):
    for i in intents:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm not sure."

# --------------------------------------------------
# STATE MANAGEMENT
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"confidence": 0.0, "last_intent": "None", "total_queries": 0}

# --------------------------------------------------
# SIDEBAR: THE NEURAL DASHBOARD
# --------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">ChatBot</div>', unsafe_allow_html=True)
    
    # System Status Pulse
    status_color = "#10b981" if model else "#ef4444"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:20px; background: rgba(255,255,255,0.03); padding:8px 12px; border-radius:30px; width:fit-content;">
        <div style="width:8px; height:8px; background:{status_color}; border-radius:50%; box-shadow: 0 0 10px {status_color}; animation: blink 2s infinite;"></div>
        <span style="font-size:0.8rem; color:#94a3b8; font-weight:500;">SYSTEM ONLINE</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Live Analysis")
    
    # Dynamic Metric Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{st.session_state.stats['confidence']:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Queries</div>
            <div class="metric-value">{st.session_state.stats['total_queries']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Detected Intent</div>
        <div class="metric-value" style="font-size: 1.1rem; color: #a78bfa;">{st.session_state.stats['last_intent'].upper()}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Neural Capabilities")
    # Dynamically list capabilities from intents.json if available
    if intents:
        available_tags = [i['tag'].capitalize() for i in intents]
        for tag in available_tags:
            st.markdown(f"🔹 {tag}")
    else:
        st.caption("Load model to see capabilities")

# MAIN INTERFACE

# --------------------------------------------------
# LOGIC: HANDLE USER INPUT
# --------------------------------------------------
def process_message(text):

    # ADDED: Store pending action
    st.session_state.pending_action = text
    st.rerun()

# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------

# WELCOME SCREEN & QUICK ACTIONS
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-text">
        <div class="hero-title">Hello, Human.</div>
        <div class="hero-subtitle">I am ready to process your intent.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Questions Grid
    st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
    if intents:
        if "welcome_suggestions" not in st.session_state:
             all_patterns = [p for i in intents for p in i['patterns']]
             st.session_state.welcome_suggestions = random.sample(all_patterns, min(4, len(all_patterns)))
        
        suggestions = st.session_state.welcome_suggestions
        
        cols = st.columns(len(suggestions))
        for idx, suggestion in enumerate(suggestions):
            with cols[idx]:
                if st.button(suggestion, use_container_width=True):
                    process_message(suggestion)

# --------------------------------------------------
# MAIN CHAT LOGIC
# --------------------------------------------------

# 1. Handle Pending Action (User Click/Type)
if "pending_action" in st.session_state and st.session_state.pending_action:
    action = st.session_state.pending_action
    
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": action})
    
    # Get Bot Response
    if model:
        tag, conf = predict(action)
        response = get_reply(tag)
        # Update Stats
        st.session_state.stats["confidence"] = conf
        st.session_state.stats["last_intent"] = tag
        st.session_state.stats["total_queries"] += 1
        
        # Add Bot Message (will be typed next)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.session_state.messages.append({"role": "assistant", "content": "System offline."})

    # Clear pending, set typing flag
    del st.session_state.pending_action
    st.session_state.typing_in_progress = True
    st.rerun()

# 2. Render History
for msg in st.session_state.messages:
    # Skip the last assistant message if we are about to type it
    if st.session_state.get("typing_in_progress") and msg == st.session_state.messages[-1] and msg["role"] == "assistant":
        continue
    
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 3. Handle Typing Animation (if active)
if st.session_state.get("typing_in_progress"):
    last_msg = st.session_state.messages[-1]
    response_text = last_msg["content"]
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for chunk in response_text.split():
            full_text += chunk + " "
            time.sleep(0.15) # Slower, more natural typing speed
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    
    # Typing done
    st.session_state.typing_in_progress = False
    st.rerun()

# FOLLOW-UP SUGGESTIONS
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    if intents:
        # Generate new suggestions only if message count changed (i.e. new bot reply)
        # We store them in 'current_suggestions' tied to the message length to keep them stable during interactions
        state_key = f"suggestions_{len(st.session_state.messages)}"
        
        if state_key not in st.session_state:
             all_patterns = [p for i in intents for p in i['patterns']]
             st.session_state[state_key] = random.sample(all_patterns, min(3, len(all_patterns)))
        
        suggestions = st.session_state[state_key]
        
        cols = st.columns(len(suggestions))
        for idx, suggestion in enumerate(suggestions):
            with cols[idx]:
                if st.button(suggestion, key=f"follow_{idx}_{len(st.session_state.messages)}", use_container_width=True):
                    process_message(suggestion)

# CHAT INPUT
if prompt := st.chat_input("Ask me anything..."):
    process_message(prompt)
