import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
CHAT_DATA_PATH = "C:/Users/GOVT LAW COLLEGE 107/Documents/ai assistant/chat_data.txt"
GPT2_MODEL_NAME = "gpt2" # You can try "gpt2-medium" if you have more resources
MAX_NEW_TOKENS = 100 # How many tokens the bot generates per turn
DEVICE = 'cpu' # Use 'cuda' if you have a GPU and want to use it

# Adjusted generation parameters for potentially better coherence
GEN_TEMPERATURE = 0.8 # Slightly higher to encourage diversity, but not too high to get gibberish
GEN_TOP_K = 50       # Consider only top 50 likely tokens
GEN_TOP_P = 0.95     # Consider tokens whose cumulative probability sum up to 0.95

# --- Load GPT-2 Tokenizer and Model (Cached) ---
@st.cache_resource
def load_gpt2_model_and_tokenizer():
    st.info(f"Loading GPT-2 model '{GPT2_MODEL_NAME}' and tokenizer. This may take a moment...")
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME)

    # GPT-2 tokenizer doesn't have a pad_token by default, but it's good practice
    # to set it for generation tasks. Using eos_token_id is a common workaround.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval() # Set model to evaluation mode
    st.success(f"GPT-2 model '{GPT2_MODEL_NAME}' loaded successfully!")
    return tokenizer, model

tokenizer, model = load_gpt2_model_and_tokenizer()
model.to(DEVICE)

# --- Streamlit UI ---
st.set_page_config(page_title="GPT-2 Chatbot", layout="centered")
st.title("🗣️ GPT-2 Chatbot")
st.write(f"A chatbot powered by the pre-trained Hugging Face '{GPT2_MODEL_NAME}' model.")

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
            # Construct the conversation history for the model
            conversation_history = ""
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    conversation_history += f"User: {msg['content'].strip()}\n"
                else: # assistant
                    conversation_history += f"Bot: {msg['content'].strip()}\n"
            
            # Add the current prompt and prime the bot's turn
            full_prompt_for_generation = conversation_history + f"User: {prompt.strip()}\nBot:"

            input_ids = tokenizer.encode(
                full_prompt_for_generation,
                return_tensors='pt',
                truncation=True,
                max_length=tokenizer.model_max_length # Ensures input doesn't exceed 1024 tokens for GPT-2
            ).to(DEVICE)

            # Store the length of the input_ids so we can slice the output later
            prompt_length = input_ids.shape[1]

            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_k=GEN_TOP_K,
                top_p=GEN_TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )

            # Decode only the newly generated tokens (excluding the input prompt)
            newly_generated_tokens = output[0, prompt_length:]
            raw_bot_response = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True)
            
            # --- IMPROVED RESPONSE PARSING ---
            bot_text = "I'm currently unable to generate a coherent response. Please try asking something else." # Default fallback

            # Try to find the end of the bot's intended turn by looking for a new 'User:' turn
            next_user_idx = raw_bot_response.find("User:")
            if next_user_idx != -1:
                bot_text = raw_bot_response[:next_user_idx].strip()
            else:
                bot_text = raw_bot_response.strip()

            # Final check to prevent very short/empty/repetitive responses
            # Also, sometimes GPT-2 might just generate the original prompt text again.
            if not bot_text or \
               len(bot_text) < 3 or \
               bot_text.lower().strip() == prompt.lower().strip() or \
               "i'm having trouble understanding" in bot_text.lower() or \
               "i've never seen india" in bot_text.lower() or \
               "i'm an editor" in bot_text.lower(): # Added checks for the specific repetitive phrases

                bot_text = "I'm still learning and might not have a clear answer for that. Could you rephrase?"

            st.markdown(bot_text)
        st.session_state.messages.append({"role": "assistant", "content": bot_text})

# Optional: Clear chat history button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
