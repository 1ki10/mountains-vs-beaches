import streamlit as st

st.set_page_config(page_title="Test App", page_icon="🚀")

st.title("🚀 Deployment Test")
st.success("App is working!")
st.write("If you see this, basic deployment works.")

# Test button
if st.button("Click me!"):
    st.balloons()
    st.write("✅ Interactivity works!")

st.info("This is a minimal test to verify Streamlit Cloud deployment.")
