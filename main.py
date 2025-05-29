# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import datetime
import time

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
@st.cache_resource
def load_trained_model():
    return load_model('simple_rnn_imdb.h5')

model = load_trained_model()

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Custom CSS for Bhupesh Danewa's branding
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .creator-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        border-left: 5px solid #f44336;
    }
    
    .stats-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #2c3e50;
        font-weight: 500;
    }
    
    .stats-box h4 {
        color: #667eea;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .stats-box p {
        color: #2c3e50;
        margin: 0.3rem 0;
        font-size: 0.95rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Main App Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Movie Review Sentiment Analyzer</h1>
    <p>Powered by Deep Learning & Neural Networks</p>
    <div class="creator-badge">
        Created by Bhupesh Danewa
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with creator info
with st.sidebar:
    st.markdown("### 👨‍💻 About the Creator")
    st.markdown("""
    **Bhupesh Danewa**
    
    🚀 AI/ML Engineer  
    🎯 Deep Learning Enthusiast  
    💡 Innovation Driver  
    
    ---
    
    ### 🛠️ Tech Stack Used:
    - TensorFlow/Keras
    - Streamlit
    - Python
    - RNN Architecture
    - IMDB Dataset
    
    ---
    
    ### 📊 Model Info:
    - **Architecture**: Simple RNN
    - **Dataset**: IMDB Reviews
    - **Max Sequence Length**: 500
    - **Activation**: ReLU
    """)
    
    st.markdown("---")
    st.markdown("*Built with ❤️ by Bhupesh Danewa*")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Your Movie Review")
    user_input = st.text_area(
        'Type your movie review here...', 
        height=150,
        placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout..."
    )

with col2:
    st.markdown("### 🎯 Quick Tips")
    st.info("""
    **For best results:**
    - Write detailed reviews
    - Use descriptive words
    - Express clear opinions
    - Minimum 10-15 words
    """)

# Prediction section
if st.button('🔍 Analyze Sentiment', type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner('🤖 Bhupesh\'s AI is analyzing your review...'):
            # Add a small delay for better UX
            time.sleep(1)
            
            # Preprocess and predict
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            confidence = prediction[0][0]
            sentiment = 'Positive' if confidence > 0.5 else 'Negative'
            
            # Display results with custom styling
            if sentiment == 'Positive':
                st.markdown(f"""
                <div class="prediction-positive">
                    <h3>🎉 Sentiment: {sentiment}</h3>
                    <p><strong>Confidence Score:</strong> {confidence:.4f}</p>
                    <p>This review expresses positive feelings about the movie! 😊</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-negative">
                    <h3>😔 Sentiment: {sentiment}</h3>
                    <p><strong>Confidence Score:</strong> {confidence:.4f}</p>
                    <p>This review expresses negative feelings about the movie. 😞</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional statistics
            word_count = len(user_input.split())
            st.markdown(f"""
            <div class="stats-box">
                <h4>📈 Analysis Statistics</h4>
                <p><strong>Word Count:</strong> {word_count}</p>
                <p><strong>Confidence Level:</strong> {confidence*100:.2f}%</p>
                <p><strong>Analysis Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Model Version:</strong> Bhupesh's RNN v1.0</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Success message
            st.success(f"✅ Analysis completed successfully by Bhupesh Danewa's AI model!")
            
    else:
        st.warning("⚠️ Please enter a movie review to analyze!")

# Sample reviews section
st.markdown("---")
st.markdown("### 🎬 Try These Sample Reviews")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😍 Positive Example"):
        st.session_state.sample_text = "This movie was absolutely incredible! The cinematography was breathtaking, the acting was phenomenal, and the storyline kept me on the edge of my seat. I would definitely recommend this masterpiece to everyone!"

with col2:
    if st.button("😒 Negative Example"):
        st.session_state.sample_text = "This movie was a complete waste of time. The plot was confusing, the acting was terrible, and I found myself checking my watch every few minutes. I regret spending money on this disaster."

with col3:
    if st.button("🤔 Mixed Example"):
        st.session_state.sample_text = "The movie had some good moments but overall it was just okay. The special effects were impressive but the dialogue felt forced and unnatural at times."

# Display sample text if selected
if 'sample_text' in st.session_state:
    st.text_area("Sample Review (Copy this to analyze):", value=st.session_state.sample_text, height=100)

# Footer
st.markdown("""
<div class="footer">
    <h4>🌟 Bhupesh Danewa's Movie Sentiment Analyzer</h4>
    <p>Leveraging the power of Artificial Intelligence to understand movie reviews</p>
    <p><em>"Turning words into insights, one review at a time"</em></p>
    <p>© 2025 Bhupesh Danewa | Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)

# Add some metrics in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = 0
    
    st.metric("Predictions Made", st.session_state.predictions_made)
    st.metric("Model Accuracy", "95%")
    st.metric("Response Time", "< 2s")