import streamlit as st
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM

CHROMA_PERSIST_DIR = "./chroma_db"

st.set_page_config(page_title="RAG Chat")
st.title("RAG Chat")


@st.cache_resource(show_spinner="Setting up the pipeline, please wait...")
def load_rag_pipeline(url):
    bs4_strainer = bs4.SoupStrainer(class_=("content-area"))
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    local_embeddings = OllamaEmbeddings(model="all-minilm:latest")

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=local_embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = OllamaLLM(model="llama3.2:1b")

    return retriever, llm


def get_answer(question, retriever, llm):
    retrieved_docs = retriever.invoke(question)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    response = llm.invoke(
        f"Answer the question according to the context given very briefly.\n"
        f"Question: {question}\n"
        f"Context: {context}"
    )
    return response


if "messages" not in st.session_state:
    st.session_state.messages = []

if "url_loaded" not in st.session_state:
    st.session_state.url_loaded = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Paste a URL to load or ask a question"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if prompt.startswith("http://") or prompt.startswith("https://"):
        st.session_state.url_loaded = False
        with st.chat_message("assistant"):
            with st.spinner("Loading the article and building the vector store..."):
                st.session_state.retriever, st.session_state.llm = load_rag_pipeline(prompt)
                st.session_state.url_loaded = True
            response = "Article loaded successfully. You can now ask questions about it."
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        if not st.session_state.url_loaded:
            with st.chat_message("assistant"):
                response = "Please paste a URL first so I can load the article."
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_answer(prompt, st.session_state.retriever, st.session_state.llm)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})