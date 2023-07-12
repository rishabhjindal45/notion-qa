"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import pickle
import pathlib
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
# chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
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


# From here down is all the StreamLit UI.
st.set_page_config(page_title="Blendle HR QA Bot", page_icon=":robot:")
st.header("Blendle HR QA Bot")
st.subheader("Ask me anything about Blendle's HR Policies!")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    result = chain({"query": user_input})
    print(f"Answer: {result['result']}")
    sources = [doc.metadata for doc in result['source_documents']]
    # result = chain({"question": user_input})
    # output = f"Answer: {result['answer']}\nSources: {result['sources']}"
    output = f"Answer: {result['result']}\nSources: {str(sources[0]['source'])[10:]}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")