import streamlit as st
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
import tempfile
import time

# Load environment variables
load_dotenv()

# Initialize ChromaDB
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def check_ollama_service():
    """Check if Ollama service is running and model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            # Check if llama2 model is available
            models = response.json().get("models", [])
            if not any(model["name"] == "llama2" for model in models):
                # Pull llama2 model if not available
                st.info("Downloading llama2 model... This may take a few minutes...")
                requests.post("http://localhost:11434/api/pull", json={"name": "llama2"})
            return True
    except requests.exceptions.ConnectionError:
        return False
    return False

def ensure_ollama_running():
    """Ensure Ollama service is running and provide instructions if it's not"""
    if not check_ollama_service():
        st.error("""
        Ollama service is not running! Please follow these steps:
        
        1. Make sure Ollama is installed:
           - Download from https://ollama.ai/
           - Run the installer
        
        2. Start Ollama:
           - Windows: Run Ollama from the Start Menu
           - Mac/Linux: Open terminal and run `ollama serve`
        
        3. After starting Ollama, refresh this page
        """)
        st.stop()

def scrape_constitution():
    """Scrape the Constitution text from the official website"""
    url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text()
        return content
    except Exception as e:
        st.error(f"Error scraping constitution: {str(e)}")
        return None

def process_document(content, source_type="constitution"):
    """Process document content and add it to the vector store"""
    # Ensure Ollama is running before processing
    ensure_ollama_running()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(content)

    # Create embeddings and store in ChromaDB using Ollama
    embeddings = OllamaEmbeddings(model="llama2")
    
    if st.session_state.vector_store is None:
        st.session_state.vector_store = Chroma.from_texts(
            chunks,
            embeddings,
            collection_metadata={"source": source_type}
        )
    else:
        st.session_state.vector_store.add_texts(chunks)

def process_uploaded_file(uploaded_file):
    """Process uploaded files based on their type"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    try:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)

        pages = loader.load()
        text_content = "\n".join([page.page_content for page in pages])
        process_document(text_content, source_type=uploaded_file.name)
        
    finally:
        os.unlink(file_path)

def get_conversation_chain():
    """Create a conversation chain using the vector store"""
    # Ensure Ollama is running before creating the chain
    ensure_ollama_running()
    
    llm = Ollama(model="llama2")
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vector_store.as_retriever(),
        return_source_documents=True
    )

def main():
    st.title("Kazakhstan Constitution AI Assistant")
    
    # Check Ollama service status at startup
    ensure_ollama_running()
    
    # Sidebar for document upload
    with st.sidebar:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload additional documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx"]
        )

    # Initialize vector store with constitution if not already done
    if st.session_state.vector_store is None:
        with st.spinner("Loading Constitution..."):
            constitution_text = scrape_constitution()
            if constitution_text:
                process_document(constitution_text)
                st.success("Constitution loaded successfully!")

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                process_uploaded_file(uploaded_file)
        st.success("All files processed successfully!")

    # Chat interface
    if st.session_state.vector_store is not None:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        user_question = st.chat_input("Ask a question about the Constitution of Kazakhstan:")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                chain = get_conversation_chain()
                response = chain({"question": user_question, "chat_history": []})
                
                st.write(response["answer"])
                st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

    else:
        st.warning("Loading the knowledge base... Please wait.")

if __name__ == "__main__":
    main() 