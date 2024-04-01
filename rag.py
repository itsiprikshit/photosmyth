import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DOCS_DIR = './uploaded'

def get_raw_documents():
    raw_documents = DirectoryLoader(DOCS_DIR).load()
    return raw_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

def get_embedder():
    embedder = NVIDIAEmbeddings(model='nvolveqa_40k', type='passage')
    return embedder

def create_vector_store():
    '''
        This function creates a vector store of the embeddings.
        I used FAISS to create the vector store that uses nvolveqa_40k model to create the embeddings

        Finally, I save the vector store - This is also called indexing the vector store.
    '''
    embedder = get_embedder()
    documents = get_raw_documents()
    chunks = split_documents(documents)

    db = FAISS.from_documents(chunks, embedder)
    db.save_local('vector_store')
    return db

def get_vector_store():
    embedder = get_embedder()

    if os.path.exists('./vector_store'):
        db = FAISS.load_local('vector_store', embedder, allow_dangerous_deserialization=True)
        return db

    return None

def query_vs(vs, query):
    '''
        Now we can query the vector store to get the similar documents.
        There are few methods of doing this -
        - similarity_search
        - as a retriever
    '''

    retriever = vs.as_retriever()
    docs = retriever.get_relevant_documents(query)
    return docs

def create_prompt():
    chat_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are a helpful AI bot. Your name is Bumblebee.'\
                'You will reply to questions only based on the context that you are provided.'\
                'If something is out of context, you will refrain from replying and politely decline to respond to the user.'),
            ('human', '{input}')
        ]
    )

    return chat_template

def create_model():
    llm = ChatNVIDIA(model='mixtral_8x7b')
    return llm

def init_chain():
    '''
    To make it as easy as possible to create custom chains, we’ve implemented a “Runnable” protocol. The Runnable protocol is 
    implemented for most components. This is a standard interface, which makes it easy to define custom chains as well as invoke 
    them in a standard way. The standard interface includes: invoke, stream, batch.

    All runnables expose input and output schemas to inspect the inputs and outputs.

    Input Schema - chain.input_schema.schema()
    Output Schema - chain.output_schema.schema()
    '''

    prompt = create_prompt()
    llm = create_model()

    chain = prompt | llm | StrOutputParser()

    return chain


if __name__ == '__main__':

    input = 'What is Diag?'

    vs = get_vector_store()

    if vs == None:
        print('Vector store not available!')
        print('Creating vector store...')
        vs = create_vector_store()

    res = query_vs(vs, input)

    context = ''
    sources = []

    for i in range(len(res)):
        context += res[i].page_content
        source = {
            'content': res[i].page_content,
            'source': res[i].metadata['source']
        }

        sources.append(source)

    chain = init_chain()

    query = f'Context: {context} \n\nQuestion: {input}'

    ans = ''
    for response in chain.stream({'input': query}):
        ans += response

    print(ans)
    print(f'Sources: {sources}')