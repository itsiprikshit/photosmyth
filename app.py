import streamlit as st
import base64
from io import BytesIO
import json
import pickle
import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate

# st.set_page_config(layout = "wide")

# {'Authorization': 'Bearer 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', 'Accept': 'application/json'}

st.title("Hi! I am Photosmyth.")
st.subheader("I can smith photos for you ;)")

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
                st.markdown(message["content"]["query"])
                st.image(message["content"]["img"], width=300)

def setup_uploader():
    DOCS_DIR = os.path.abspath("./uploaded")

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    with st.form("upload-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file!", accept_multiple_files = True)
        submitted = st.form_submit_button("Upload")

    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())

        raw_documents = DirectoryLoader(DOCS_DIR).load()

        if raw_documents:
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)

            document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")

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

def get_context(input):
    global vectorstore

    if vectorstore == None:
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)

    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    return context

def main():
    user_input = st.chat_input("A cat with a ball.")

    chain = initialize()

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        context = get_context(user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(user_input)

            augmented_user_input = "You are a helpful AI assistant named PhotoSmyth. You will reply to questions by generating images only based on the context that you are provided.  If something is out of context, generate an image of an exclamation mark. \n\n"

            augmented_user_input = augmented_user_input + "Context: " + context + "\n\nQuestion: " + user_input + "\n"

            img = chain.invoke({"input": augmented_user_input})
            st.image(img, width=300)

        st.session_state.messages.append({"role": "assistant", "content": {"query": user_input, "img": img}})

if __name__ == "__main__":
    setup_chat()
    setup_uploader()
    main()