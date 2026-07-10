import streamlit as st
import os
import base64
import requests
from dotenv import load_dotenv

# 1. Load the environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# 2. Check if the key exists before running
if not API_KEY:
    st.error("Missing API Key! Make sure your .env file has 'GROQ_API_KEY=your_key_here'")
    st.stop()

st.title("📸 Simple VQA Agent")

# 3. Interface
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Ask a question about the image")

# 4. Logic
if st.button("Get Answer"):
    if not uploaded_file:
        st.warning("Please upload an image first!")
    elif not question:
        st.warning("Please ask a question!")
    else:
        # Convert image to base64
        image_bytes = uploaded_file.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # API Details
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        }
        
        # Sending request
        try:
            with st.spinner("Thinking..."):
                response = requests.post(url, headers=headers, json=payload)
                
            if response.status_code == 200:
                data = response.json()
                answer = data['choices'][0]['message']['content']
                st.image(uploaded_file, caption="Uploaded Image")
                st.write("### Answer:")
                st.write(answer)
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")