import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Vacation Preference AI", 
    page_icon="🌍"
)

st.title("🌍 Vacation Preference AI")
st.success("✅ App deployed successfully!")

# Quick file check
st.write("📁 **Available Files:**")
files = [f for f in os.listdir('.') if f.endswith(('.py', '.pkl', '.txt', '.md'))]
for file in files:
    if os.path.isfile(file):
        size = os.path.getsize(file)
        st.write(f"- {file} ({size:,} bytes)")

# Check model file
model_file = 'Vacation_Preference_XGBoost_Model.pkl'
if os.path.exists(model_file):
    size = os.path.getsize(model_file)
    st.info(f"🎯 Model file found: {size/1024/1024:.1f} MB")
else:
    st.warning("⚠️ Model file not found")

# Simple test form
st.markdown("---")
st.markdown("## 🧪 Quick Test")

with st.form("test"):
    name = st.text_input("Your name:")
    preference = st.radio("Initial preference:", ["🏔️ Mountains", "🏖️ Beaches"])
    
    if st.form_submit_button("Test"):
        st.success(f"Hello {name}! You prefer {preference}")
        if "Mountains" in preference:
            st.balloons()
        else:
            st.snow()

st.markdown("---")
st.info("🔄 **Next:** Add full ML model functionality after basic deployment works")
st.markdown("**Digital Skola Data Science - Group 10**")
