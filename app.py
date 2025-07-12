import json
import streamlit as st
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableSequence
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()

# ------------------------
# Messenger-Style Chat CSS
# ------------------------
st.markdown("""
    <style>
    .chat-container {
        max-width: 700px;
        margin: auto;
        padding: 10px;
    }

    .chat-bubble {
        padding: 12px 15px;
        border-radius: 20px;
        margin: 8px 0;
        max-width: 80%;
        word-wrap: break-word;
        display: inline-block;
        font-size: 16px;
    }

    .user-bubble {
        background-color: #4f8bf9;
        color: white;
        margin-left: auto;
        text-align: right;
    }

    .bot-bubble {
        background-color: #f1f0f0;
        color: black;
        margin-right: auto;
        text-align: left;
    }

    .chat-row {
        display: flex;
        align-items: flex-end;
    }

    .chat-row.user {
        justify-content: flex-end;
    }

    .chat-row.bot {
        justify-content: flex-start;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("📋 Leave Policy Chatbot")
st.sidebar.markdown("Ask any questions about leave policies.")

# ------------------------
# Load and prepare data
# ------------------------
@st.cache_resource
def initialize_chain():
    with open('data.json', 'r') as f:
        data = json.load(f)

    docs = [
        Document(
            page_content=f"{entry['title']} {entry['description']}",
            metadata={'title': entry['title'], 'leave_type': entry['leave_type']}
        ) for entry in data
    ]

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorestore = Chroma.from_documents(docs, embedding_model)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    model = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation"
    )
    llm = ChatHuggingFace(llm=model, temperature=0)

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorestore,
        docstore=store,
        child_splitter=splitter,
    )
    retriever.add_documents(docs)

    prompt_template = """
You are an intelligent assistant designed to answer questions related to an organization's leave policies. Use the provided context from the company's official leave policy documents to generate accurate, concise, and helpful answers.

Context:
{context}

Question:
{question}

Instructions:
- Only use the information provided in the context to answer.
- If the context does not contain enough information, say: "The document does not contain information about this topic."
- Be specific and avoid vague responses.
- Use bullet points or formatting if it improves clarity.
- Do not make up policies or speculate.

Answer:
"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    parser = StrOutputParser()
    chain = RunnableSequence({"context": retriever, "question": RunnablePassthrough()}, prompt ,llm ,parser)
    return chain

# ------------------------
# Initialize chain
# ------------------------
chain = initialize_chain()

# ------------------------
# Chat UI
# ------------------------
st.title("💼 Leave Policy Chatbot")
st.write("Ask me anything about the company's leave policy!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.chat_input("Ask your leave policy question...")

# Response logic
if user_input:
    with st.spinner("Thinking..."):
        answer = chain.invoke(user_input)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", answer))

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for speaker, message in st.session_state.chat_history:
    bubble_class = "user-bubble" if speaker == "user" else "bot-bubble"
    row_class = "user" if speaker == "user" else "bot"

    st.markdown(f"""
        <div class="chat-row {row_class}">
            <div class="chat-bubble {bubble_class}">{message}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
