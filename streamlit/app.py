import streamlit as st
from llama_index.llms import AzureOpenAI
import openai
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)

from config import settings, Settings

from utils import build_sentence_window_index
from llama_index.chat_engine.types import BaseChatEngine


def get_sentence_window_chat_engine(
    sentence_index: VectorStoreIndex,
    similarity_top_k=6,
    rerank_top_n=2,
    chat_mode="condense_question",
) -> BaseChatEngine:
    # define postprocessors
    # postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_chat_engine = sentence_index.as_chat_engine(
        chat_mode=chat_mode,
        verbose=True,
        similarity_top_k=similarity_top_k,
        node_postprocessors=[rerank],
    )
    return sentence_window_chat_engine


st.set_page_config(
    page_title="Chat with the Crayon personal handbook",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat med Crayonite 💬🦙")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Jeg har tilgang til Crayon \
                sitt interne dokumentasjonssystem. \
                    Spør meg om hva som helst! 😊",
        }
    ]


def get_llm_model(cfg: Settings):
    openai.api_key = cfg.AZURE_OPENAI_API_KEY
    llm = AzureOpenAI(
        engine=cfg.AZURE_DEPLOYMENT_NAME,
        model=cfg.AZURE_DEPLOYMENT_MODEL,
        temperature=0.1,
        azure_endpoint=cfg.AZURE_OPENAI_API_ENDPOINT,
        azure_api_key=cfg.AZURE_OPENAI_API_KEY,
        api_version=cfg.AZURE_OPENAI_API_VERSION,
        system_prompt="Du er en HR-ekspert og jobber med å \
            svare på HR-spørsmål.\
        Anta at alle spørsmål er relatert til Crayon \
            sitt interne dokumentasjonssystem. \
        Hold svarene dine tekniske og basert på fakta – \
              ikke hallusiner funksjoner.",
    )
    return llm


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(
        text="Loading and indexing the Crayon – hang tight! This should take 1-2 minutes."
    ):
        azure_llm = get_llm_model(cfg=settings)
        # reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        # docs = reader.load_data()
        # service_context = ServiceContext.from_defaults(llm=azure_llm)
        # index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        index = build_sentence_window_index(
            document=SimpleDirectoryReader(
                input_dir="./data", recursive=True
            ).load_data(),
            llm=azure_llm,
        )
        return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    # st.session_state.chat_engine = index.as_chat_engine(
    #     chat_mode="condense_question", verbose=True
    # )
    # st.session_state.chat_engine = get_sentence_window_query_engine(index, 6, 3)
    st.session_state.chat_engine = get_sentence_window_chat_engine(index, 6, 3)

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            # response = st.session_state.chat_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
