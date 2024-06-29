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
import bs4

# Load environment variables from .env file
load_dotenv()

## LLM Model
CONFIG_MODEL="gpt-3.5-turbo-0125"
llm = ChatOpenAI(model=CONFIG_MODEL)

# Initialize the HuggingFace embeddings (if we want to use offline embeddings instead of the OpenAI one)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## Indexing
### Loaders
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
# Define the USER_AGENT header
user_agent = "Mozilla/5.0 (compatible; MyLLMCrawler/1.0; +http://mywebsite.com/bot-info)"
headers = {
    "User-Agent": user_agent
}
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
    header_template={
      'User-Agent': user_agent
  })
docs = loader.load()

# print(len(docs[0].page_content))

### Splitters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))
# # also print the first split
# print(all_splits[0])
# print('-------------------')
# print('\n')
# print(all_splits[-1])

### Stores
# use the OpenAIEmbeddings to calculate the embeddings
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
# or use the HuggingFace embeddings (offline and no API call needed) to calculate the embeddings
vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf_embeddings)

## Retrieve
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
# len(retrieved_docs)
# print(retrieved_docs)


## Retrieve and Generate
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """Use the following context in XML tags to answer the question in XML tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible. You don't need to wrap the answer in XML tags.

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
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# result = rag_chain.invoke("What is Task Decomposition?")
# print(result)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)


# ---
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
