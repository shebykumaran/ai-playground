#!/usr/bin/env python

import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import LLMChain
from vector import *

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Create system and user message templates
system_template = "Rewrite the text in {response_type} english"

# Wrap templates using LangChain prompt types
system_message_prompt = SystemMessagePromptTemplate.from_template("Using the information from {search_results}")
human_message_prompt = HumanMessagePromptTemplate.from_template("{user_input}")

# Combine into a ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
])

# Create vector data
url = "https://www.aljazeera.com/news/2025/5/16/five-key-takeaways-from-us-president-donald-trumps-middle-east-trip"
text = get_text_from_url(url)
chunks = split_text(text)
vectordb = store_in_chroma(chunks, embedding)

# Create the LLMChain
chain = chat_prompt | llm


# Run the chain
user_input = input("Ask something: ")

retriever = vectordb.as_retriever()
search_results = retriever.invoke(user_input)

print(search_results)

response = chain.invoke({"search_results": search_results, "user_input": user_input})
print(response.content)

