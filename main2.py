import nltk
import os
import tempfile
import pickle
import streamlit as st
import time
import openai
import langchain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

 # Get all environment variables
from dotenv import load_dotenv
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Web interface

st.title("Research Query Tool")
st.sidebar.subheader("Websites")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# File upload
st.sidebar.subheader('Upload documents')
uploaded_files = st.sidebar.file_uploader("Select files from your computer", type=['txt','pdf','docx'], accept_multiple_files=True)

process_url_clicked = st.sidebar.button("Get content")


# progress bar

main_placefolder = st.empty()


# Filepath location
file_path = "faiss_index\index.pkl"

 

# Initialize llm
llm = OpenAI(temperature=0.6, max_tokens=500)

if process_url_clicked:
    # load data from URLs

    data = []

    if any(urls):

        try:

            loader = UnstructuredURLLoader(urls=urls)

            main_placefolder.text("Loading your data. . . . ⏳")

            data = loader.load()

            st.write(f"Loaded {len(data)} documents from URLs")

        except Exception as e:

            st.error(f"Failed to load data from URLs: {e}")

 

   

    # load data from uploaded documents

    if uploaded_files:

        for file in uploaded_files:

            try:

                # create temporary file

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:

                    tmp_file.write(file.getbuffer())

                    tmp_file_path = tmp_file.name

           

                # temporary file path for UnstructuredFileLoader

                file_loader = UnstructuredFileLoader(tmp_file_path)

                main_placefolder.text(f"Loading file: {file.name}. . . . 📂")

                data.extend(file_loader.load())

 

                # delete temporary file after processing

                os.remove(tmp_file_path)

            except Exception as e:

                st.error(f"Failed to load file {file.name}: {e}")

 

    # check if data was loaded properly

    if not data:

        st.error("No data was loaded from the provided URLs or files. Please check the inputs")

        st.stop()

    else:

        st.write(f"Loaded {len(data)} documents")

 

    # split data

    text_splitter = RecursiveCharacterTextSplitter(

        separators=['\n\n', '\n', '.', ','],

        chunk_size=1000

    )

    main_placefolder.text("Splitting your data. . . . ✂")

    docs = text_splitter.split_documents(data)

 

    # validate text splitting

    if not docs:

        st.error("No documents were created after splitting. Ensure the data contains valid text")

        st.stop()

    else:

        st.write(f"Split into {len(docs)} chunks")

 

    # embeddings

    # store embeddings in FAISS index

    # and FAISS index in Streamlit session state memory

    embeddings = OpenAIEmbeddings()

    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    st.session_state['vectorstore'] = vectorstore_openai

    main_placefolder.text("Creating embedding vector. . . . 📚")

    time.sleep(2)

 

# Question box and submit button

with st.form("query_form"):

    query = st.text_input("What do you want to know?", "")

    enter_button = st.form_submit_button("Enter")

 

if enter_button:

    if query:

        # Load FAISS index from session state

        if 'vectorstore' in st.session_state:

            vectorstore = st.session_state['vectorstore']

 

            # Create retrieval QA chain

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            result = chain({"question":query}, return_only_outputs=True)

 

            # {"answer":" ", "sources":" "}

            st.header("Answer")

            st.write(result["answer"])

 

            # Display sources

            sources = result.get("sources", "")

            if sources:

                st.subheader("Sources:")

                sources_list = sources.split("\n")

                for source in sources_list:

                    st.write(source)

 

        else:

            st.error("FAISS index not found. Please process data first")