import os
import tempfile
import pickle
import streamlit as st
import time
from dotenv import load_dotenv

import langchain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import openai
import faiss

# Load environment variables
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Web interface
st.title("WTF: What's the fact")
st.sidebar.title("URLs")

# URLs input
urls = []
for i in range(4):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# File upload
st.sidebar.subheader('Upload Documents')
uploaded_files = st.sidebar.file_uploader(
    "Upload files from your PC", type=["txt", "pdf", "docx"], accept_multiple_files=True
)

process_url_clicked = st.sidebar.button("Fetch the News")

# Progress placeholder
main_placefolder = st.empty()

# FAISS index directory
faiss_index_dir = "faiss_index"

# Initialize LLM
llm = OpenAI(temperature=0.6, max_tokens=500)

if process_url_clicked:
    # Load data from websites and uploaded files
    data = []

    if any(urls):
        loader = UnstructuredURLLoader(urls=urls)
        main_placefolder.text("Loading your data. . . . ⏳")
        data.extend(loader.load())

    if uploaded_files:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.getbuffer())
                tmp_file_path = tmp_file.name

            file_loader = UnstructuredFileLoader(tmp_file_path)
            main_placefolder.text(f"Loading your document: {file.name} . . .")
            data.extend(file_loader.load())

            os.remove(tmp_file_path)

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placefolder.text("Splitting your data. . . . ✂")
    docs = text_splitter.split_documents(data)

    # Generate embeddings and store in FAISS index
    embeddings = OpenAIEmbeddings()

    if docs:
        main_placefolder.text("Creating embedding vector. . . . 📚")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)

        # Save FAISS index
        if not os.path.exists(faiss_index_dir):
            os.makedirs(faiss_index_dir)

        vectorstore_openai.save_local(faiss_index_dir)
        main_placefolder.text("FAISS index saved successfully")
    else:
        main_placefolder.text("No valid data to process.")

# Question box
with st.form("Ask your Question"):
    query = st.text_input("What do you want to know?")
    enter_button = st.form_submit_button("Hit")

if enter_button:
    if query:
        # Load FAISS index
        if os.path.exists(faiss_index_dir):
            embeddings = OpenAIEmbeddings()
            try:
                vectorstore = FAISS.load_local(
                    faiss_index_dir, embeddings, allow_dangerous_deserialization=True
                )
                main_placefolder.text("FAISS index loaded successfully")

                # Create retrieval QA chain
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, retriever=vectorstore.as_retriever()
                )
                result = chain({"question": query}, return_only_outputs=True)

                # Display results
                st.header("Answer")
                st.write(result.get("answer", "No answer found."))

                # Display sources
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    source_list = sources.split("\n")
                    for source in source_list:
                        st.write(source)
            except Exception as e:
                st.error(f"Error loading FAISS index: {e}")
        else:
            st.error("FAISS index not found. Please process data first.")
