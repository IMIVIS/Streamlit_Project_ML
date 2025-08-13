import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Page Config
st.set_page_config(page_title="🧑🏾‍🚀 Chatbot", page_icon="🧑🏾‍🚀", layout="wide")

st.title("🧑🏾‍🚀 Chatbot - Local Mode")
st.markdown("""
This chatbot runs **entirely on your machine** using the Hugging Face `microsoft/DialoGPT-medium` model.
No API keys or internet connection are required after the first download.
""")

# Load model and tokenizer only once
if "chatbot_pipeline" not in st.session_state:
    with st.spinner("Loading chatbot model... This might take 10–30 seconds."):
        model_name = "microsoft/DialoGPT-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        chatbot_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        st.session_state.chatbot_pipeline = chatbot_pipeline
        st.session_state.chat_history = []

# User input
user_input = st.text_input("💬 You:", placeholder="Type your message and press Enter")

# Send button
if st.button("Send") and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    # Build chat context from history
    chat_context = ""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_context += f"User: {msg['text']}\n"
        else:
            chat_context += f"Bot: {msg['text']}\n"

    # Generate bot reply
    with st.spinner("🧑🏾‍🚀 Thinking..."):
        bot_reply = st.session_state.chatbot_pipeline(
            chat_context + "Bot:",
            max_length=200,
            pad_token_id=st.session_state.chatbot_pipeline.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )[0]['generated_text']

    # Extract only the latest bot part
    bot_reply = bot_reply[len(chat_context + "Bot:"):].strip()
    st.session_state.chat_history.append({"role": "bot", "text": bot_reply})

# Show chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['text']}")
    else:
        st.markdown(f"🤖 **Bot:** {msg['text']}")

# Optional clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
