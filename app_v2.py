import streamlit as st
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import schedule
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Directory to store PDFs for auto-update
PDF_STORAGE_DIR = "pdf_storage"
if not os.path.exists(PDF_STORAGE_DIR):
    os.makedirs(PDF_STORAGE_DIR)

# 🔧 Function to clean text of surrogate/unicode errors
def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def format_links(text):
    # Detect URLs and wrap them in styled anchor tags
    url_pattern = r"(https?://[^\s]+)"
    return re.sub(url_pattern, r'<a href="\1" target="_blank" style="color:#1a73e8;">\1</a>', text)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return clean_text(text)

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    
    # Debugging output: Print cleaned chunks
    for i in range(min(20, len(chunks))):
        cleaned_chunk = clean_text(chunks[i])
        logger.info(f"Chunk {i+1}:\n{cleaned_chunk}\n{'-'*50}\n")
    
    return chunks

# Function to create FAISS index
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    logger.info("Storing the following chunks in FAISS:")
    for i, chunk in enumerate(text_chunks):
        cleaned_chunk = clean_text(chunk)
        logger.info(f"Chunk {i+1}:\n{cleaned_chunk}\n{'-'*50}\n")
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    logger.info("FAISS index updated successfully.")

# Function to process PDFs in storage directory
def process_stored_pdfs():
    pdf_files = [os.path.join(PDF_STORAGE_DIR, f) for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pdf")]
    if pdf_files:
        logger.info("Starting auto-update of FAISS index...")
        raw_text = ""
        for pdf_path in pdf_files:
            with open(pdf_path, "rb") as pdf_file:
                raw_text += get_pdf_text([pdf_file])
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            logger.info("Auto-update completed successfully.")
        else:
            logger.info("No text extracted from stored PDFs.")
    else:
        logger.info("No PDFs found in storage directory for auto-update.")

# Schedule auto-update at 2:00 AM daily
def schedule_auto_update():
    schedule.every().day.at("23:32").do(process_stored_pdfs)
    logger.info("Auto-update scheduled for 2:00 AM daily.")

# Background thread for scheduler
def run_scheduler():
    schedule_auto_update()
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Start scheduler in a separate thread
def start_background_scheduler():
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Background scheduler started.")

# Function to load the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant answering questions based strictly on the given context. Use the information provided to answer comprehensively, accurately, and in the appropriate format.

    Guidelines:
    - Use only the given context to answer. If the answer is not found, respond: "Answer is not available in the context."
    - Never fabricate or assume facts not in the context.
    - Treat 'NITJ', 'nitj', 'institute', and 'Dr. B.R. Ambedkar National Institute of Technology' as referring to the same entity.
    - If a question involves steps, procedures, or processes, use clear bullet points.
    - If a numerical answer is requested (e.g. how many clubs), and the number is not directly given, count based on the context.
    - Answer in the tone and format suitable for the question type:
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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return "FAISS index not found. Please process documents first."

    retrieved_docs = vector_store.similarity_search(user_question, k=3)
    
    if not retrieved_docs:
        response_text = "Answer is not available in the context or ask question in different way."
    else:
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": retrieved_docs, "question": user_question}, return_only_outputs=True)
        response_text = f"{response['output_text']}"

    st.session_state.chat_history.append(("🤖", response_text))
    return response_text

# Streamlit UI
def main():
    st.set_page_config(page_title="🎓 NITJ AI Chatbot", layout="wide")

    # Start the background scheduler
    start_background_scheduler()

    st.markdown("""
        <h1 style="text-align:center; color:#074791;">🎓 NITJ Academic Assistant 🤖</h1>
        <p style="text-align:center; font-size:18px; color:#555;">
        Your AI-powered guide for academic queries, admissions, research, and campus information.
        </p>
        <hr style="border:1px solid #002147;">
    """, unsafe_allow_html=True)

    user_question = st.text_input("🤔 Ask something about NITJ (admissions, academics, research, facilities)...")

    if user_question:
        st.session_state.chat_history.append(("🧑‍💻", f"{user_question}"))
        response = user_input(user_question)
        st.markdown(format_links(response), unsafe_allow_html=True)

    # Chat history section
    st.markdown("<h3 style='color:#074791;'>📜 Chat History</h3>", unsafe_allow_html=True)
    chat_container = st.container()

    with chat_container:
        for role, message in reversed(st.session_state.chat_history):  # Reverse the order
            st.markdown(f"<div style='padding:10px; border-radius:8px; background:#f1f1f1;color:black; margin-bottom:5px;'><strong>{role}</strong>: {format_links(message)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar menu
    with st.sidebar:
        st.markdown("<h2 style='color:#edf0f2;'>📂 Upload Academic PDFs</h2>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("📤 Upload relevant NITJ documents", accept_multiple_files=True)

        if st.button("🔄 Process Documents"):
            with st.spinner("⏳ Extracting and processing text..."):
                # Save uploaded PDFs to storage directory
                for pdf in pdf_docs:
                    pdf_path = os.path.join(PDF_STORAGE_DIR, pdf.name)
                    with open(pdf_path, "wb") as f:
                        f.write(pdf.read())
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Documents processed successfully!")

        # Clear Chat History Button
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

        st.markdown("<h3 style='color:#edf0f2;'>📌 About This Chatbot</h3>", unsafe_allow_html=True)
        st.info("""
        - This chatbot is powered by **Gemini AI** to assist students, faculty, and visitors with **NITJ-related information**.
        - Upload **academic PDFs** to enhance chatbot responses.
        - Auto-updates daily at 2:00 AM to refresh document data.
        - Ask about **admissions, research, scholarships, faculty, and more!**
        """)

if __name__ == "__main__":
    main()