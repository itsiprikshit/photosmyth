import streamlit as st
import base64
from io import BytesIO
import time
import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from rag import get_vector_store, query_vs, create_vector_store

# st.set_page_config(layout = 'wide')

st.set_page_config(page_title='Photosmyth')
st.title('Hi! I am Photosmyth.')
st.subheader('I can synthesize photos for you ;)')
st.text('\n')

vectorstore = None
vector_store_path = 'vectorstore.pkl'

DOCS_DIR = './uploaded'

def setup_uploader():
    if 'noupload' not in st.session_state:
        st.session_state.noupload = False

    uploaded_files = st.file_uploader('Choose a PDF file', accept_multiple_files=True, label_visibility='collapsed', disabled=st.session_state.noupload)

    if len(uploaded_files) > 0:
        st.session_state.noupload = True

    for uploaded_file in uploaded_files:
        path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(path, 'wb') as f:
            f.write(uploaded_file.read())

    if len(uploaded_files) > 0:
        st.success(f'Upload successfull!')

    return True if len(uploaded_files) > 0 else False

def setup_chat():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message['role'] == 'user':
                st.markdown(message['content'])
            if message['role'] == 'assistant':
                st.markdown(f"You request is processed! {message['content']['query']}")
                st.image(message['content']['img'], width=300)

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
    llm = ChatNVIDIA(model='ai-sdxl-turbo')

    llm.client.payload_fn = create_payload

    chain = llm | base64_to_img

    return chain

def get_stream(message):

    def stream() :
        for word in message.split(' '):
            yield word + ' '
            time.sleep(.1)

    return stream

def insert_styles():
    st.markdown('''
        <style>
            img {
                border-radius: 10px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
            }

            #MainMenu {
                display: None;
            }
        </style>
    ''', unsafe_allow_html=True)

def main():
    user_input = st.chat_input('Generate an image of a cat with a ball.')

    chain = initialize()

    if user_input:
        with st.chat_message('user'):
            st.markdown(user_input)

        vs = get_vector_store()

        res = query_vs(vs, user_input)

        context = ''

        for i in range(len(res)):
            context += res[i].page_content

        query = f'Context: {context} \n\nQuestion: {user_input}'

        # print(query)

        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            message_placeholder.write_stream(get_stream(f'Processing your request... {user_input}'))

            img = chain.invoke(query)
            message_placeholder.markdown(f'You request is processed! {user_input}')
            st.image(img, width=300)

        st.session_state.messages.append({'role': 'user', 'content': user_input})
        st.session_state.messages.append({'role': 'assistant', 'content': {'query': user_input, 'img': img}})

if __name__ == '__main__':
    uploaded = setup_uploader()

    if uploaded:
        create_vector_store()

    setup_chat()
    insert_styles()

    main()
