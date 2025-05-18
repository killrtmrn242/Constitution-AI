# Kazakhstan Constitution AI Assistant

An AI-powered assistant that can answer questions about the Constitution of the Republic of Kazakhstan and process additional documents for context-aware responses.

## Features

- Interactive chat interface using Streamlit
- Integration with Ollama (using Llama 2 model)
- Vector storage using ChromaDB for efficient document retrieval
- Support for multiple document formats (PDF, DOCX, TXT)
- Automatic scraping and processing of the Constitution text
- Chat history tracking
- Document upload functionality (single or multiple files)

## Prerequisites

1. Install Ollama:
   - Download from https://ollama.ai/
   - Run the installer
   - Start Ollama from the Start Menu (Windows) or run `ollama serve` (Mac/Linux)

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure Ollama is running on your system

2. Start the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

4. The application will automatically:
   - Download the Llama 2 model (first time only)
   - Load the Constitution of Kazakhstan
   - Process and store it in the vector database

## Using the Application

1. The application automatically loads the Constitution of Kazakhstan when started
2. You can upload additional documents through the sidebar:
   - Single file upload
   - Multiple files at once
   - Supported formats: PDF, DOCX, TXT
3. Ask questions in the chat interface about:
   - The Constitution
   - Any uploaded documents
4. The AI will provide context-aware answers based on all available documents
5. Chat history is maintained during the session

## Technical Details

- Uses Llama 2 through Ollama for:
  - Text embeddings
  - Question answering
- Implements ChromaDB as a vector store for efficient document retrieval
- Features recursive text splitting for optimal context management
- Uses LangChain for document processing and chat chain management

## File Structure

- `app.py` - Main application code
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `LICENSE` - MIT License

## Note

Make sure Ollama is running before starting the application. The first time you run the application, it will download the Llama 2 model, which may take several minutes depending on your internet connection. 