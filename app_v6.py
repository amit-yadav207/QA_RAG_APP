# app_v5.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Remove after recreating environment
import streamlit as st
import re
import socket
import os.path as ospath
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logger = logging.getLogger('NITJChatbot')
logger.setLevel(logging.DEBUG)

# Create rotating file handler
handler = RotatingFileHandler('chatbot.log', maxBytes=1_000_000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Log app start
logger.info("Starting NITJ AI Chatbot application")

# Load API key from .env file
load_dotenv()
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
logger.debug("Google API key configured")


# --------------------------- Utility Functions ---------------------------

def clean_text(text):
    logger.debug("Cleaning text input")
    cleaned = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    logger.debug("Text cleaned successfully")
    return cleaned

def format_links(text):
    logger.debug("Formatting links in text")
    url_pattern = r"(https?://[^\s]+)"
    formatted = re.sub(url_pattern, r'<a href="\1" target="_blank" style="color:#1a73e8;">\1</a>', text)
    logger.debug("Links formatted successfully")
    return formatted

def check_connectivity():
    logger.info("Checking internet connectivity")
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        logger.info("Internet connectivity confirmed")
        return True
    except OSError as e:
        logger.error(f"Connectivity check failed: {e}")
        return False

def faiss_index_exists():
    logger.debug("Checking if FAISS index exists")
    exists = ospath.exists("faiss_index") and ospath.isfile(ospath.join("faiss_index", "index.faiss"))
    logger.debug(f"FAISS index exists: {exists}")
    return exists

# --------------------------- PDF Processing ----------------------------

def get_pdf_text(pdf_docs):
    logger.info("Extracting text from %d PDF documents", len(pdf_docs))
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
            logger.debug("Text extracted from PDF: %s", pdf.name)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf.name}: {e}")
    cleaned_text = clean_text(text)
    logger.info("PDF text extraction completed, length: %d characters", len(cleaned_text))
    return cleaned_text

def get_text_chunks(text):
    logger.info("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    logger.debug("Text split into %d chunks", len(chunks))
    return chunks

def get_vector_store(text_chunks):
    logger.info("Creating FAISS vector store with %d text chunks", len(text_chunks))
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        logger.info("FAISS vector store created and saved to 'faiss_index'")
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        raise

# --------------------------- Chat Functionality ----------------------------

def get_conversational_chain():
    logger.debug("Initializing conversational chain")
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

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt,
            document_variable_name="context"  # Map input_documents to context
        )
        logger.debug("Conversational chain initialized successfully")
        return chain
    except Exception as e:
        logger.error(f"Error initializing conversational chain: {e}")
        raise

@st.cache_resource
def load_local_llm():
    logger.info("Loading local LLM (microsoft/phi-2)")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float32,
            device_map="auto"
        )
        logger.info("Local LLM loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading local LLM: {e}")
        raise

def local_llm_response(context, question):
    logger.info("Generating local LLM response for question: %s", question[:50])
    try:
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
        inputs = tokenizer(prompt.format(context=context, question=question), return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.5,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded.split("Answer:")[-1].strip()
        logger.debug("Local LLM response generated: %s", answer[:50])
        return answer
    except Exception as e:
        logger.error(f"Error generating local LLM response: {e}")
        raise

@st.cache_resource
def load_vector_store():
    logger.info("Loading FAISS vector store")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS vector store loaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading FAISS vector store: {e}")
        raise

def user_input2(user_question):
    logger.info("Processing user question: %s", user_question[:50])
    try:
        vector_store = load_vector_store()
        retrieved_docs = vector_store.similarity_search(user_question, k=3)
        logger.debug("Retrieved %d documents for question", len(retrieved_docs))
        
        if not retrieved_docs:
            response_text = "Answer is not available in the context or ask question in different way."
            logger.info("No relevant documents found for question")
        else:
            if check_connectivity():
                chain = get_conversational_chain()
                response = chain.invoke({"context": retrieved_docs, "question": user_question})
                response_text = response
                logger.debug("Online response generated: %s", response_text[:50])
            else:
                context_text = "\n".join([doc.page_content for doc in retrieved_docs])
                response_text = local_llm_response(context_text, user_question)
                logger.debug("Offline response generated: %s", response_text[:50])

        st.session_state.chat_history.append(("🤖", response_text))
        logger.info("User question processed successfully")
        return response_text
    except Exception as e:
        logger.error(f"Error processing user question: {e}")
        raise

# --------------------------- Streamlit UI ----------------------------

def main():
    logger.info("Initializing Streamlit UI")
    try:
        st.set_page_config(page_title="🎓 NITJ AI Chatbot", layout="wide")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            logger.debug("Initialized chat_history in session state")

        if "offline_mode" not in st.session_state:
            st.session_state.offline_mode = False
            logger.debug("Initialized offline_mode in session state")

        if "use_existing_index" not in st.session_state:
            st.session_state.use_existing_index = False
            logger.debug("Initialized use_existing_index in session state")

        st.markdown("""
            <h1 style="text-align:center; color:#074791;margin-bottom:-10px;margin-top:-30px">🎓 NITJ Academic Assistant 🤖</h1>
            <p style="text-align:center; font-size:18px; color:#555;">
            Your AI-powered guide for academic queries, admissions, research, and campus information.
            </p>
            <hr style="border:1px solid #002147;">
        """, unsafe_allow_html=True)

        user_question = st.text_input("🤔 Ask something about NITJ (admissions, academics, research, facilities)...")

        if user_question:
            logger.debug("User submitted question: %s", user_question[:50])
            st.session_state.chat_history.append(("🧑‍💻", f"{user_question}"))
            response = user_input2(user_question)
            # st.markdown(format_links(response), unsafe_allow_html=True)

        st.markdown("<h3 style='color:#074791;'>📜 Chat History</h3>", unsafe_allow_html=True)
        chat_container = st.container()
        with chat_container:
            for role, message in reversed(st.session_state.chat_history):
                st.markdown(f"<div style='padding:10px; border-radius:8px; background:#f1f1f1;color:black; margin-bottom:5px;'><strong>{role}</strong>: {format_links(message)}</div>", unsafe_allow_html=True)

        
        with st.sidebar:
            st.markdown("<h2 style='color:#edf0f2;'>📂 Document Processing</h2>", unsafe_allow_html=True)
            
            # Checkbox to use existing FAISS index
            use_existing = st.checkbox("📁 Use existing FAISS index", value=faiss_index_exists(), disabled=not faiss_index_exists())
            st.session_state.use_existing_index = use_existing
            logger.debug("Use existing FAISS index set to: %s", use_existing)

            if not st.session_state.use_existing_index:
                pdf_docs = st.file_uploader("📤 Upload relevant NITJ documents", accept_multiple_files=True)
                if st.button("🔄 Process Documents", disabled=not pdf_docs):
                    logger.info("Processing uploaded documents")
                    with st.spinner("⏳ Extracting and processing text..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ Documents processed successfully!")
                        logger.info("Document processing completed")
            elif faiss_index_exists():
                st.info("Using existing FAISS index from 'faiss_index' folder.")
                logger.info("Using existing FAISS index")
            else:
                st.warning("No FAISS index found. Please upload and process PDFs.")
                logger.warning("No FAISS index found")

            st.markdown("<h3 style='color:#edf0f2;'>🌐 Connectivity Settings</h3>", unsafe_allow_html=True)
            auto_detected = not check_connectivity()
            st.session_state.offline_mode = st.toggle("📴 Offline Mode", value=auto_detected)
            logger.debug("Offline mode set to: %s", st.session_state.offline_mode)

            if st.session_state.offline_mode:
                st.warning("⚠️ You are in Offline Mode. Responses are based on local data only.")
            else:
                st.success("✅ Online Mode Enabled. Gemini API will be used.")

            if st.sidebar.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                logger.info("Chat history cleared")

            st.markdown("<h3 style='color:#edf0f2;'>📌 About This Chatbot</h3>", unsafe_allow_html=True)
            st.info("""
            - Powered by **Gemini AI** to assist students, faculty, and visitors with **NITJ-related information**.
            - Upload **academic PDFs** or use existing FAISS index to enhance responses.
            - Ask about **admissions, research, scholarships, exams, and more!**
            """)
        logger.debug("Streamlit UI rendered successfully")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()