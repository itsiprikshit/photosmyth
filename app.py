import streamlit as st
import base64
from io import BytesIO
import json

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# st.set_page_config(layout = "wide")

st.subheader("Chat with your AI Assistant!")

def setup_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            if message["role"] == "assistant":
                st.markdown(message["content"]["query"])
                st.image(message["content"]["img"], width=300)

def create_payload(d):
    if d:
        d = {"prompt": d.get("messages", [{}])[0].get("content")}

    return d

def to_pil_img(d):
    img = d.response_metadata['b64_json']
    return BytesIO(base64.b64decode(img))

def initialize():
    llm = ChatNVIDIA(model="sdxl")

    llm.client.payload_fn = create_payload

    chain = llm | to_pil_img

    return chain

def main():
    user_input = st.chat_input("A cat with a ball.")

    chain = initialize()

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(user_input)

            img = chain.invoke(user_input)
            st.image(img, width=300)

        st.session_state.messages.append({"role": "assistant", "content": {"query": user_input, "img": img}})

if __name__ == "__main__":
    setup_chat()
    main()