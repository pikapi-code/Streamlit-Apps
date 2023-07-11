import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

st.set_page_config(page_title="Document Comparison")

st.header(" :robot_face: Document comparison app :books:")

def get_pdf_text(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    return chunks

def create_vector_db(file1, file2, name1, name2):
    text1 = get_pdf_text(file1)
    text2 = get_pdf_text(file2)

    chunk1 = get_text_chunks(text1)
    chunk2 = get_text_chunks(text2)

    embeddings = OpenAIEmbeddings(chunk_size=1)
    persist_directory1 = name1
    vectorstore1 = Chroma.from_texts(chunk1, embeddings, persist_directory=persist_directory1)
    vectorstore1.persist()

    persist_directory2 = name2
    vectorstore2 = Chroma.from_texts(chunk2, embeddings, persist_directory=persist_directory2)
    vectorstore2.persist()

file1 = st.file_uploader("Upload the first document.")

file2 = st.file_uploader("Upload the second document")

if st.button("Process"):
    with st.spinner("Processing files!"):
        create_vector_db(file1, file2, file1.name, file2.name, )