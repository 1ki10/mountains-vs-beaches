import streamlit as st
import pandas as pd
import numpy as np
import os

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Vacation Preference AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple test - no model loading for now
st.success("🎉 App is starting successfully!")

# Check what files we have
st.write("📁 **Files in directory:**")
files = os.listdir('.')
for file in files:
    size = os.path.getsize(file) if os.path.isfile(file) else "Directory"
    st.write(f"- {file} ({size} bytes)")

# Test sklearn import
try:
    from sklearn.preprocessing import StandardScaler
    st.success("✅ Sklearn import successful")
except Exception as e:
    st.error(f"❌ Sklearn error: {e}")

# Check model file specifically
model_file = 'Vacation_Preference_XGBoost_Model.pkl'
if os.path.exists(model_file):
    file_size = os.path.getsize(model_file)
    st.success(f"✅ Model file found: {file_size:,} bytes")
    
    # Check if file size is reasonable
    if file_size > 100_000_000:  # 100MB
        st.warning("⚠️ Model file is very large - might cause memory issues")
    elif file_size == 0:
        st.error("❌ Model file is empty!")
    else:
        st.info(f"📊 Model file size looks good: {file_size/1024/1024:.2f} MB")
else:
    st.error("❌ Model file not found!")

# ===== HEADER =====
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; text-align: center; color: white; margin: 2rem 0;">
    <h1>🌍 AI-Powered Vacation Preference Predictor</h1>
    <h3>🚧 DIAGNOSTIC MODE 🚧</h3>
    <p>Testing deployment - Model loading temporarily disabled</p>
</div>
""", unsafe_allow_html=True)

# Simple form for testing
st.markdown("## 🧪 Basic Functionality Test")

with st.form("test_form"):
    st.write("Testing basic Streamlit components:")
    
    name = st.text_input("Your name:")
    age = st.number_input("Age:", min_value=18, max_value=100, value=30)
    preference = st.selectbox("Quick preference:", ["Mountains", "Beaches"])
    
    submitted = st.form_submit_button("Test Submit")
    
    if submitted:
        st.success(f"✅ Form works! Hello {name}, age {age}, you prefer {preference}")
        st.balloons()

# Memory usage check
st.markdown("---")
st.markdown("### 🔍 System Info")
st.write(f"Python version: {st.__version__}")
st.write("Memory usage: Testing basic operations...")

# Test creating data
test_data = pd.DataFrame({
    'col1': range(1000),
    'col2': np.random.randn(1000)
})
st.success(f"✅ Created test dataframe: {test_data.shape}")

st.markdown("---")
st.markdown("### 📋 Next Steps")
st.info("""
If this diagnostic version loads successfully:
1. ✅ Basic Streamlit functionality works
2. ✅ File system access works  
3. ✅ Dependencies are installed correctly

**Issue is likely with model loading/memory.**
""")

st.warning("""
**Temporary Solution:**
- Deploy this version first to confirm basic functionality
- Then gradually add back model loading with optimizations
""")

# Footer
st.markdown("---")
st.markdown("**🎓 Digital Skola Data Science Final Project - Group 10 - Diagnostic Mode**")
