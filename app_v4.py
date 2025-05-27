import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import ChatPromptTemplate
import torch
import socket
import re
import logging

# --------------------------- Logging Setup ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logging.info("🚀 Chatbot started.")

# --------------------------- Utility Functions ---------------------------
def clean_text(text):
    logging.info("🔧 Cleaning extracted text.")
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def format_links(text):
    logging.info("🔗 Formatting links in response.")
    url_pattern = r"(https?://[^\s]+)"
    return re.sub(url_pattern, r'<a href="\1" target="_blank" style="color:#1a73e8;">\1</a>', text)

def check_connectivity():
    logging.info("🌐 Checking internet connectivity.")
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        logging.info("✅ Internet connection available.")
        return True
    except OSError:
        logging.warning("⚠️ No internet connection.")
        return False

# --------------------------- PDF Processing ---------------------------
def get_pdf_text(pdf_docs):
    logging.info("📄 Extracting text from uploaded PDFs.")
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    logging.info("✅ Text extraction complete.")
    return clean_text(text)

def get_text_chunks(text):
    logging.info("✂️ Splitting text into chunks.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    logging.info(f"🔢 Generated {len(chunks)} chunks.")
    return chunks

def get_vector_store(text_chunks):
    logging.info("📦 Generating embeddings and saving FAISS index.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    logging.info("✅ FAISS index saved locally.")

# --------------------------- Language Models ---------------------------
def get_conversational_chain():
    logging.info("🤖 Setting up Gemini conversational chain.")
    prompt_template = """
    You are an intelligent assistant answering questions based strictly on the given context. Use the information provided to answer comprehensively, accurately, and in the appropriate format.

    Guidelines:
    - Use only the given context to answer. If the answer is not found, respond: "Answer is not available in the context."
    - Never fabricate or assume facts not in the context.
    - Treat 'NITJ', 'nitj', 'institute', and 'Dr. B.R. Ambedkar National Institute of Technology' as referring to the same entity.
    - If a question involves steps, procedures, or processes, use clear bullet points.
    - For definitions or factual queries: provide concise, formal answers.
    - For lists: use bullet points.
    - For how-to or process questions: step-by-step format.
    - For comparisons: use tables or summaries.
    - Do NOT search externally; use only the provided context.
    - If the context is insufficient, say so clearly.
    - If counting not provided directly then count and give answer.

    Context:
    {context}

    Question:
    {question}
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    logging.info("✅ Conversational chain ready.")
    return chain

@st.cache_resource
def load_local_llm():
    logging.info("📥 Loading local Phi-2 model.")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float32,
        device_map="auto"
    )
    logging.info("✅ Local LLM loaded.")
    return tokenizer, model

def local_llm_response(context, question):
    logging.info(f"🧠 Generating local response for: {question}")
    tokenizer, model = load_local_llm()
    prompt = """
    You are an intelligent assistant answering questions based strictly on the given context. Use the information provided to answer comprehensively, accurately, and in the appropriate format.

    Guidelines:
    - Use only the given context to answer. If the answer is not found, respond: "Answer is not available in the context."
    - Never fabricate or assume facts not in the context.
    - Treat 'NITJ', 'nitj', 'institute', and 'Dr. B.R. Ambedkar National Institute of Technology' as referring to the same entity.
    - If a question involves steps, procedures, or processes, use clear bullet points.
    - For definitions or factual queries: provide concise, formal answers.
    - For lists: use bullet points.
    - For how-to or process questions: step-by-step format.
    - For comparisons: use tables or summaries.
    - Do NOT search externally; use only the provided context.
    - If the context is insufficient, say so clearly.
    - If counting not provided directly then count and give answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=300)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()
    logging.info("✅ Local LLM response generated.")
    return answer

# --------------------------- Chat Processing ---------------------------
def user_input2(user_question):
    logging.info(f"💬 User asked: {user_question}")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    logging.info("🔍 Performing vector similarity search.")
    retrieved_docs = vector_store.similarity_search(user_question, k=3)

    if not retrieved_docs:
        logging.warning("⚠️ No relevant documents found.")
        response_text = "Answer is not available in the context or ask question in different way."
    else:
        if check_connectivity():
            chain = get_conversational_chain()
            response = chain.invoke({"input_documents": retrieved_docs, "question": user_question})
            response_text = response
            logging.info("✅ Response generated using Gemini API.")
        else:
            context_text = "\n".join([doc.page_content for doc in retrieved_docs])
            response_text = local_llm_response(context_text, user_question)
            logging.info("✅ Response generated using local LLM.")

    st.session_state.chat_history.append(("🤖", response_text))
    return response_text

# --------------------------- Streamlit UI ---------------------------
def main():
    logging.info("📱 Streamlit app launched.")
    st.set_page_config(page_title="📚 NITJ Chatbot", layout="wide")
    st.title("📘 NITJ Academic Chatbot")
    st.markdown("Ask me anything about NITJ, academics, placements, policies, and more!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("🗂️ Upload Documents")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("🔄 Process Documents"):
            logging.info("📥 Document upload button clicked.")
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Documents processed successfully!")
                logging.info("📚 Document processing complete.")
            else:
                st.warning("⚠️ Please upload at least one PDF.")

        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            logging.info("🧹 Chat history cleared.")

    user_question = st.text_input("💬 Enter your question:")
    if user_question:
        logging.info(f"📩 Processing user input: {user_question}")
        with st.spinner("🤔 Thinking..."):
            response = user_input2(user_question)
            st.markdown(format_links(response), unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for i, (sender, message) in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**{sender}:** {format_links(message)}", unsafe_allow_html=True)

# --------------------------- Entry Point ---------------------------
if __name__ == "__main__":
    main()
