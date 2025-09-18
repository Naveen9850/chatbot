import streamlit as st
import os
from groq import Groq  # Groq SDK

st.title("IC AI")

# Initialize Groq client with API key from environment variable
if 'client' not in st.session_state:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY environment variable not set.")
    else:
        st.session_state['client'] = Groq(api_key=groq_api_key)

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Sidebar for parameters
st.sidebar.title("Model Parameters")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider("Max number of tokens", min_value=1, max_value=4096, value=256)

# Display conversation history
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
if prompt := st.chat_input("Em kavali ra gundu"):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        client = st.session_state.get('client')
        if not client:
            st.error("Groq client is not initialized. Please set the GROQ_API_KEY environment variable.")
        else:
            # Call Groq API
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state['messages']
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            # Collect assistant reply from stream
            response_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content

            # Show full response once (nice paragraph format)
            st.markdown(response_text.strip(), unsafe_allow_html=True)

            # Save assistant response as plain string
            st.session_state['messages'].append({"role": "assistant", "content": response_text.strip()})