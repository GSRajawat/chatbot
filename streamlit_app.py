# streamlit_app.py
import streamlit as st
import torch
import os # To check for file existence

# Import your model and utility functions
from model import TinyGPT, EMBED_SIZE, BLOCK_SIZE, NUM_HEADS, NUM_LAYERS
from utils import load_vocabulary

# --- Configuration ---
CHAT_DATA_PATH = "chat_data.txt"
MODEL_PATH = "tiny_gpt_model.pth"
MAX_NEW_TOKENS = 100 # How many tokens the bot generates per turn
DEVICE = 'cpu' # Use 'cuda' if you have a GPU and want to use it

# --- Load Vocabulary ---
if not os.path.exists(CHAT_DATA_PATH):
    st.error(f"Error: chat_data.txt not found at {CHAT_DATA_PATH}. Please ensure it's in the same directory.")
    st.stop() # Stop the app if data file is missing

vocab_size, stoi, itos, encode, decode = load_vocabulary(CHAT_DATA_PATH)

# --- Load Model (Cached to prevent reloading on every interaction) ---
@st.cache_resource # Use st.cache_resource for models/large objects
def load_tinygpt_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model weights '{MODEL_PATH}' not found. Please train the model first and save its weights.")
        st.stop() # Stop the app if model file is missing

    model = TinyGPT(vocab_size=vocab_size,
                    embed_size=EMBED_SIZE,
                    block_size=BLOCK_SIZE,
                    num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model.eval() # Set model to evaluation mode
    st.success("TinyGPT model loaded successfully!")
    return model

model = load_tinygpt_model().to(DEVICE)

# --- Streamlit UI ---
st.set_page_config(page_title="TinyGPT Chatbot", layout="centered")
st.title("🗣️ TinyGPT Chatbot")
st.write("A simple character-level language model trained on custom chat data.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Say something to the bot..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            # Ensure the prompt starts with "User: " for consistency
            full_prompt = "User: " + prompt.strip() + "\nBot:"

            # Encode the prompt
            idx = torch.tensor(encode(full_prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)

            # Generate response
            generated_indices = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()
            raw_response = decode(generated_indices)

            # Extract only the bot's response, assuming the model continues the "Bot:" part
            # This is a heuristic and might need refinement based on your model's actual output
            try:
                bot_response_start_idx = raw_response.find("Bot:") + len("Bot:")
                # Find the next "User:" or end of string to delimit the bot's actual reply
                next_user_idx = raw_response.find("User:", bot_response_start_idx)
                if next_user_idx != -1:
                    bot_text = raw_response[bot_response_start_idx:next_user_idx].strip()
                else:
                    bot_text = raw_response[bot_response_start_idx:].strip()
                
                # Further refine: sometimes it generates the prompt again, remove it
                if bot_text.startswith(prompt.strip()):
                    bot_text = bot_text[len(prompt.strip()):].strip()
                
                # If the bot_text is empty or just whitespace, provide a fallback
                if not bot_text:
                    bot_text = "I'm sorry, I couldn't generate a clear response."

            except Exception as e:
                bot_text = f"Error processing response: {e}. Raw: {raw_response}"
                
            st.markdown(bot_text)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_text})

# Optional: Clear chat history button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun() # Rerun the app to clear the display
