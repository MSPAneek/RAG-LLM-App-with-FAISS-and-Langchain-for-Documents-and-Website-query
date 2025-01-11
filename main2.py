import os

import tempfile

import pickle

import streamlit as st

import time

import langchain

from langchain.llms import OpenAI

from langchain.chains import RetrievalQAWithSourcesChain

from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader

from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import FAISS

import openai



# Get all environment variables (To make sure the key is secured)
from dotenv import load_dotenv
load_dotenv()
openai.api_key=st.secrets["OPENAI_API_KEY"]
# Web interface
st.title("WTF: What's the fact")
st.sidebar.title("URLs")


urls = []

for i in range(4):

    url = st.sidebar.text_input(f"URL {i+1}")

    urls.append(url)

 # File upload
st.sidebar.subheader('Upload Documents')
uploaded_files=st.sidebar.file_uploader("Upload files from your PC",type=["txt","pdf","docx"],accept_multiple_files=True)

process_url_clicked = st.sidebar.button("Fetch the News")

 

# progress bar

main_placefolder = st.empty()

 

# Filepath location

file_path = "faiss_index\index.pkl"

 

# Initialize llm

llm = OpenAI(temperature=0.6, max_tokens=500)

 

if process_url_clicked:

    # load data from websites
    data=[]
    if any (urls):
        loader = UnstructuredURLLoader(urls=urls)

        main_placefolder.text("Loading your data. . . . ⏳")

        data.extend(loader.load())
    
    # Load data from your PC
    if uploaded_files:
        for file in uploaded_files:

            # Create Temporary File
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.getbuffer())
                tmp_file_path=tmp_file.name
            # Temporary file path for UnstructuredFileLoader
            file_loader=UnstructuredFileLoader(tmp_file_path)
            main_placefolder.text(f"Loading your document:{file.name}......")
            data.extend(file_loader.load())

            # Delete Temporary File after Processing 
            os.remove(tmp_file_path)

    # split data

    text_splitter = RecursiveCharacterTextSplitter(

        separators=['\n\n', '\n', '.', ','],

        chunk_size=1000

    )

    main_placefolder.text("Splitting your data. . . . ✂")
    docs = text_splitter.split_documents(data)

    # embeddings
    # Store in FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Creating embedding vector. . . . 📚")
    time.sleep(2)

    # Directory to save FAISS index
    faiss_index_dir = "faiss_index"
    if not os.path.exists(faiss_index_dir):
        os.makedirs(faiss_index_dir)

    # Save faiss index on local computer
    vectorstore_openai.save_local(faiss_index_dir)
    print("FAISS index saved successfully")

# Question box
with st.form("Ask your Question "):
    query = st.text_input("What do you want to know?")
    enter_button=st.form_submit_button("Hit")

if enter_button:
    if query:
        # Load FAISS index from directory
        if os.path.exists("faiss_index"):
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully")
            
            # Create retrieval QA chain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question":query}, return_only_outputs=True)

            # {"answer":" ", "sources":" "}
            st.header("Answer")
            st.write(result["answer"])
            # Display Source
            sources=result.get("sources", "")
            if sources:
                st.subheader("sources:")
                source_list=sources.split("\n")
                for sources in source_list:
                    st.write(sources)


