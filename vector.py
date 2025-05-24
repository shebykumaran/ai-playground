import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import os

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    print(soup)
    return soup.get_text()

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def store_in_chroma(texts, embedding, persist_dir="chroma_store"):
    vectordb = Chroma.from_texts(texts, embedding=embedding, persist_directory=persist_dir)
    return vectordb
