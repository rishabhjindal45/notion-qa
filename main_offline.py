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
pathlib.PosixPath = pathlib.WindowsPath

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
st.set_page_config(page_title="Blendle Notion QA Bot", page_icon=":robot:")
st.header("Blendle Notion QA Bot")

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
    sources = list(set([str(source['source'])[10:-3] for source in sources]))
    output = f"Answer:\n {result['result']}\n\nSources: \n" + '\n'.join(sources)
    # Create columns for the buttons
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns([1,1,1,1,1,1,1,1,1,1,1,1])

    # Add thumbs up and thumbs down buttons
    col1.button('👍')
    col2.button('👎')
    # output_history = f"Answer: {result['result']}\nSources: \n" + '\n'.join(sources)
    # answer_color = "green"
    # sources_color = "blue"

    # output = f'<p style="color:{answer_color};">Answer:<br>{result["result"]}</p>'
    # output += f'<p style="color:{sources_color};">Sources:<br>{"<br>".join(sources)}</p>'

    # # Using markdown to print HTML
    # st.markdown(output, unsafe_allow_html=True)


    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


with st.sidebar:
    st.markdown("""
    # Welcome to the Blendle Notion QA Bot! 
    This app let's you chat with Blendle's HR documentation through the power of Large Language Models (LLMs).
                
    You can ask it questions about:
    1. Blendle's values
    2. Blendle's social code
    3. Diversity and inclusion
    4. Hiring
    5. Perks and benefits
                
    And lots more! So what are you waiting for? Go ahead and ask your first question!
                
    **Please note that this is only a Proof of Concept system and may contain bugs or unfinished features.**
    """)