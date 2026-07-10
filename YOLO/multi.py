"""
Multimodal Chatbot using Gradio and OpenRouter (Gemma Model)
Supports text input and image upload for visual understanding
"""

import os
import base64
import io
from pathlib import Path
from typing import Optional
from PIL import Image
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "ERROR: OPENROUTER_API_KEY not found in environment variables!\n"
        "Please create a .env file with: OPENROUTER_API_KEY=your_key_here\n"
        "Get your key from: https://openrouter.ai/keys"
    )

# Initialize OpenAI client pointing to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Model options available on OpenRouter
MODEL_OPTIONS = {
    "Gemma 4 31B": "google/gemma-4-31b-it",
    "Gemma 3 27B": "google/gemma-3-27b-it",
    "Gemma 3 12B": "google/gemma-3-12b-it",
}

DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, intelligent, and friendly assistant. "
    "You can see and analyze images provided by users. "
    "Respond clearly and concisely."
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def encode_image_to_base64(image) -> str:
    """
    Convert PIL Image to base64 string for API transmission
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    if image is None:
        return None
    
    # Convert PIL Image to bytes
    buffer = io.BytesIO()
    # Convert RGBA to RGB if necessary (some formats don't support RGBA)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    
    # Encode to base64
    b64_string = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_string}"


def build_messages_with_image(text: str, image, history: list) -> list:
    """
    Build message list with text and optional image for API call
    
    Args:
        text: User's text input
        image: PIL Image object (can be None)
        history: Previous chat history
        
    Returns:
        List of message dicts in OpenAI format
    """
    messages = []
    
    # Add historical messages (convert to proper format)
    for msg in history:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        else:
            messages.append({"role": "assistant", "content": msg["content"]})
    
    # Build current user message with text and/or image
    current_message = {"role": "user", "content": []}
    
    # Add text if provided
    if text and text.strip():
        current_message["content"].append({
            "type": "text",
            "text": text
        })
    
    # Add image if provided
    if image is not None:
        image_base64 = encode_image_to_base64(image)
        current_message["content"].append({
            "type": "image_url",
            "image_url": {"url": image_base64}
        })
    
    # Only add message if it has content
    if current_message["content"]:
        messages.append(current_message)
    
    return messages


def chat_with_gemma(
    message: dict,
    history: list,
    system_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    Send message to Gemma via OpenRouter API and get response
    
    Args:
        message: Dict with 'text' and 'files' (from Gradio MultimodalTextbox)
        history: Chat history list
        system_prompt: System prompt for the model
        model_name: Model identifier
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum response length
        
    Returns:
        Model's response text
    """
    try:
        # Extract text and image from Gradio message format
        user_text = message.get("text", "").strip() if message.get("text") else ""
        image_files = message.get("files", []) if message.get("files") else []
        
        # Load image if provided
        image = None
        if image_files:
            try:
                image = Image.open(image_files[0])
            except Exception as e:
                return f"Error loading image: {str(e)}"
        
        # Validate input
        if not user_text and image is None:
            return "Please enter text or upload an image."
        
        # Build messages array
        messages = build_messages_with_image(user_text, image, history)
        
        # Add system prompt
        messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Call OpenRouter API
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # Optional: add site information for OpenRouter rankings
            extra_headers={
                "HTTP-Referer": "http://localhost:7860",
                "X-Title": "Gemma Multimodal Chatbot"
            }
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create and configure Gradio interface"""
    
    with gr.Blocks(title="Gemma Multimodal Chatbot", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🤖 Gemma Multimodal Chatbot
            
            Chat with Google's Gemma model via OpenRouter API.
            Supports **text and image inputs** for multimodal understanding.
            
            **Features:**
            - Real-time chat interface
            - Upload images for visual questions
            - Adjustable model parameters
            - Chat history preservation
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat display area
                # Chat display area
             chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_label=True
    # type="messages"  <-- REMOVE OR DELETE THIS LINE
                )
                
            
            with gr.Column(scale=1):
                # Settings panel
                gr.Markdown("### ⚙️ Settings")
                
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value="Gemma 4 31B",
                    label="Model",
                    info="Select Gemma model version"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=256,
                    maximum=4096,
                    value=1024,
                    step=256,
                    label="Max Tokens",
                    info="Maximum response length"
                )
                
                system_prompt_text = gr.Textbox(
                    label="System Prompt",
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=5,
                    info="Define the AI's behavior"
                )
                
                clear_button = gr.ClearButton(
                    [chatbot],
                    value="Clear Chat",
                    scale=2
                )
        
        # Input area
        gr.Markdown("### 📝 Your Message")
        
        with gr.Row():
            # Multimodal input (text + images)
            multimodal_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="single",
                file_types=["image"],
                placeholder="Type your message here... or upload an image",
                show_label=False,
                min_width=400,
            )
            
            submit_button = gr.Button(
                "Send 📤",
                variant="primary",
                size="lg",
                scale=1,
                min_width=80
            )
        
        # Information box
        gr.Markdown(
            """
            ℹ️ **How to use:**
            1. Type your question in the text box
            2. (Optional) Upload an image to ask about it
            3. Adjust settings on the right if needed
            4. Click "Send" to get a response
            """
        )
        
        # ====================================================================
        # EVENT HANDLERS
        # ====================================================================
        
        def process_message(message, chat_history, model_select, temp, tokens, sys_prompt):
            """Process user message and get AI response"""
            
            # Get model ID from selection
            model_id = MODEL_OPTIONS[model_select]
            
            # Add user message to chat history
            if message.get("text"):
                user_display = message["text"]
                if message.get("files"):
                    user_display += " [Image attached]"
            else:
                user_display = "[Image attached]"
            
            chat_history.append({
                "role": "user",
                "content": user_display
            })
            
            # Get AI response
            ai_response = chat_with_gemma(
                message=message,
                history=chat_history,
                system_prompt=sys_prompt,
                model_name=model_id,
                temperature=temp,
                max_tokens=int(tokens)
            )
            
            # Add AI response to chat history
            chat_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            return chat_history, ""
        
        # Connect submit button
        submit_button.click(
            fn=process_message,
            inputs=[
                multimodal_input,
                chatbot,
                model_dropdown,
                temperature_slider,
                max_tokens_slider,
                system_prompt_text
            ],
            outputs=[chatbot, multimodal_input]
        )
        
        # Allow Enter key to submit (for text input)
        multimodal_input.submit(
            fn=process_message,
            inputs=[
                multimodal_input,
                chatbot,
                model_dropdown,
                temperature_slider,
                max_tokens_slider,
                system_prompt_text
            ],
            outputs=[chatbot, multimodal_input]
        )
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    
    print("\n" + "="*60)
    print("🚀 Gemma Multimodal Chatbot")
    print("="*60)
    print("✓ Connected to OpenRouter API")
    print("✓ Model: Gemma")
    print("✓ Launching interface...")
    print("\nOpen your browser to: http://localhost:7860")
    print("="*60 + "\n")
    
    # Launch the app
    # share=False: runs locally only (set to True to create public link)
    # server_name="0.0.0.0": allows access from other machines on your network
    demo.launch(

        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )