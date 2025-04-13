# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
import streamlit as st
from langchain_openai import OpenAIEmbeddings
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import ServerlessSpec
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

#from langchain_community.vectorstores import Chroma


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    # elif extension == '.csv':
    #     import pandas as pd
    #     from langchain_community.document_loaders.csv_loader import CSVLoader
    #     loader = CSVLoader(file_path=file)
    #     data = loader.load()
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data,chunk_size=500):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=0)
    chunks=text_splitter.split_documents(data)
    return chunks


#Delete the index for adding new index name
def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()
    
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes ... ')
        for index in indexes:
            pc.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def insert_or_fetch_embeddings(index_name, chunks):
    # importing the necessary libraries and initializing the Pinecone client

    
    pc = pinecone.Pinecone()
        
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well

    # loading from existing index
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index 
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
        ) 
        )

        # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
        # inserting the embeddings into the index and returning a new Pinecone vector store object. 
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer['result']


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.00002


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    #st.image('langchain.png')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        st.write("Enter your OpenAI API Key to continue.")
        api_key_openai = st.text_input('OpenAI API Key:', type='password')
        if api_key_openai:
            os.environ['OPENAI_API_KEY'] = api_key_openai 
            

        # Get Pinecone API Key from the user
        st.write("Enter your PineCone API Key to continue.")
        api_key_pinecone = st.text_input('Pinecone API Key:', type='password')
        if api_key_pinecone:
            os.environ['PINECONE_API_KEY'] = api_key_pinecone  # Set Pinecone API Key in environment

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt','csv'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')
                   
                #Delete the existing index
                delete_pinecone_index() 
                
                #Create the new index
                index_name = 'vectordb'
                vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)

                # creating the embeddings and returning the Chroma vector store

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            #st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)

            #st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            # the current question and answer
            value = f'Q: {q} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)