import streamlit as st
import base64
from io import BytesIO
import time

from langchain_nvidia_ai_endpoints import ChatNVIDIA

# st.set_page_config(layout = "wide")

st.set_page_config(page_title='Photosmyth')
st.title("Hi! I am Photosmyth.")
st.subheader("I can synthesize photos for you ;)")
st.text("\n")
st.text("\n")

vectorstore = None
vector_store_path = "vectorstore.pkl"

def setup_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            if message["role"] == "assistant":
                st.markdown(f'You request is processed! {message["content"]["query"]}')
                st.image(message["content"]["img"], width=300)

def create_payload(data):
    payload = {'text_prompts': []}

    if 'messages' in data:
        for message in data['messages']:
            p = {'text': message['content']}
            payload['text_prompts'].append(p)

    return payload

def base64_to_img(data):
    # The role of the response is assistant
    artifacts = data.response_metadata['artifacts']
    img = artifacts[0]['base64']
    return BytesIO(base64.b64decode(img))

def initialize():
    llm = ChatNVIDIA(model="ai-sdxl-turbo")

    llm.client.payload_fn = create_payload

    chain = llm | base64_to_img

    return chain

def get_stream(message):

    def stream() :
        for word in message.split(" "):
            yield word + " "
            time.sleep(.1)

    return stream

def insert_styles():
    st.markdown("""
        <style>
            img {
                border-radius: 10px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
            }

            #MainMenu {
                display: None;
            }
        </style>
    """, unsafe_allow_html=True)

def main():
    user_input = st.chat_input("Generate an image of a cat with a ball.")

    chain = initialize()

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.write_stream(get_stream(f"Processing your request... {user_input}"))

            img = chain.invoke(user_input)
            message_placeholder.markdown(f"You request is processed! {user_input}")
            st.image(img, width=300)

        st.session_state.messages.append({"role": "assistant", "content": {"query": user_input, "img": img}})

if __name__ == "__main__":
    setup_chat()
    insert_styles()
    main()
