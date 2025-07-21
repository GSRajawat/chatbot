import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM # Import Hugging Face components

# --- Configuration ---
CHAT_DATA_PATH = "chat_data.txt" # Still useful for fine-tuning
GPT2_MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 100
DEVICE = 'cpu'
GEN_TEMPERATURE = 0.7
GEN_TOP_K = 50
GEN_TOP_P = 0.95

# --- Load GPT-2 Tokenizer and Model (Cached) ---
@st.cache_resource
def load_gpt2_model_and_tokenizer():
    st.info(f"Loading GPT-2 model '{GPT2_MODEL_NAME}' and tokenizer. This may take a moment...")
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Essential for generation

    model.eval()
    st.success(f"GPT-2 model '{GPT2_MODEL_NAME}' loaded successfully!")
    return tokenizer, model

tokenizer, model = load_gpt2_model_and_tokenizer()
model.to(DEVICE)

# --- Streamlit UI ---
st.set_page_config(page_title="GPT-2 Chatbot", layout="centered")
st.title("🗣️ GPT-2 Chatbot")
st.write(f"A chatbot powered by the pre-trained Hugging Face '{GPT2_MODEL_NAME}' model.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something to the bot..."):
    st.chat_message("user").markdown(prompt)
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

            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_k=GEN_TOP_K,
                top_p=GEN_TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Additional parameter for more controlled generation
                # This prevents the model from generating more tokens than current input_ids + max_new_tokens
                # (but max_new_tokens is already doing this)
                # Setting num_return_sequences=1 ensures we get only one generated sequence
                num_return_sequences=1
            )

            raw_generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # --- IMPROVED RESPONSE PARSING ---
            bot_text = "I'm having trouble understanding or responding to that." # Default fallback

            # The goal is to find the *first* complete "Bot:" turn *after* our input prompt.
            # We'll look for the last occurrence of "User: [user_prompt]\nBot:" and take everything after it.
            # Then, we'll try to stop at the next "User:" or the end of the generated text.

            # Find where our prompt ends and bot's response should begin in the *full generated text*
            expected_start_marker = full_prompt_for_generation
            
            # Find the index of the start marker in the raw generated text
            start_index = raw_generated_text.rfind(expected_start_marker)

            if start_index != -1:
                # Get the part of the text that comes *after* our exact prompt
                potential_response = raw_generated_text[start_index + len(expected_start_marker):].strip()

                # Find the next "User:" turn if the model decided to continue the conversation for us
                next_user_turn_idx = potential_response.find("User:")
                if next_user_turn_idx != -1:
                    bot_text = potential_response[:next_user_turn_idx].strip()
                else:
                    bot_text = potential_response.strip()
            else:
                # Fallback if the prompt structure isn't found (e.g., model generated something entirely different)
                # In this case, we might just show a snippet or the whole raw output.
                # For now, let's just use the default fallback message.
                pass # bot_text remains the fallback

            # Final check to prevent very short/empty/repetitive responses
            if not bot_text or bot_text.lower() in [prompt.lower().strip(), "bot:"]:
                bot_text = "I'm still learning, please try asking something different."

            st.markdown(bot_text)
        st.session_state.messages.append({"role": "assistant", "content": bot_text})

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
