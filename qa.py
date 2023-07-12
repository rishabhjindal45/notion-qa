"""Ask a question to the notion database."""
# import faiss
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# import pickle
# import argparse
import faiss
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pickle
import argparse

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, do not produce an answer from your own knowledge base, just say I don't know. 

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), chain_type="stuff", retriever=store.as_retriever(),
chain_type_kwargs = chain_type_kwargs, return_source_documents=True)
result = chain({"query": args.question})
print(f"Answer: {result['result']}")

metadata_list = [doc.metadata for doc in result['source_documents']]

# Print the extracted metadata
for metadata in metadata_list:
    print(metadata)

# chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=store.as_retriever())
# result = chain({"question": args.question})
# print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")