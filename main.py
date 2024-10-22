#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from bs4 import BeautifulSoup, SoupStrainer

# Load environment variables from .env file
load_dotenv()


## Config setup
CONFIG_MODEL="gpt-3.5-turbo-0125"
WEBSITE_CRAWL="https://github.com/nodejs/node/releases/latest"

## LLM Model
llm = ChatOpenAI(model=CONFIG_MODEL)

# Initialize the HuggingFace embeddings (if we want to use offline embeddings instead of the OpenAI one)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## Indexing
### Loaders

""" Load data from the website """
def load_data():
    # Define the USER_AGENT header
    user_agent = "Mozilla/5.0 (compatible; MyLLMCrawler/1.0; +http://mywebsite.com/bot-info)"
    headers = {
        "User-Agent": user_agent
    }

    # Examples for filters:
    # bs4_strainer = bs4.SoupStrainer(class_=("markdown-body my-3"))
    # bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

    bs4_strainer = SoupStrainer(lambda name, attrs: 
        (name == "div" and attrs.get('class') and "markdown-body" in attrs['class']) or 
        (name == "title")
    )

    loader = WebBaseLoader(
        web_paths=(WEBSITE_CRAWL,),
        bs_kwargs={"parse_only": bs4_strainer},
        header_template={
          'User-Agent': user_agent
      })
    docs = loader.load()

    # print(len(docs[0].page_content))
    # print(docs[0].page_content)

    return docs


""" Split the text into smaller chunks """
""" the returned chunks are Document objects """
def text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # print(len(all_splits))
    # print(all_splits[0])
    # print(all_splits[-1])

    return all_splits


""" Store the embeddings in a Vector store """
def store_embeddings(all_splits):
    # use the OpenAIEmbeddings to calculate the embeddings
    # vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    # or use the HuggingFace embeddings (offline and no API call needed) to calculate the embeddings
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf_embeddings)
    return vectorstore

def chat_prompt(vectorstore):
    ## Retrieve
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    ## Retrieve and Generate
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Use the following context in XML tags to answer the question in XML tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible. You don't need to wrap the answer in XML tags.

    <context>
    {context}
    </context>

    Question:

    <question>
    {question}
    </question>

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    for chunk in rag_chain.stream("What are the top 5 most significant changes in this release? (such as security implications, major API updates, or breaking changes)"):
        print(chunk, end="", flush=True)

documents_data = load_data()
documents_chunked = text_splitter(documents_data)
vectorstore = store_embeddings(documents_chunked)
chat_prompt(vectorstore)

# ---
# example
# simple chat prompt interface
# ---
# parser = StrOutputParser()
# system_template = "Translate the following into {language}:"
# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", "{text}")]
# )
# chain = prompt_template | llm | parser
# result = chain.invoke({"language": "italian", "text": "hi"})
# print(result)
