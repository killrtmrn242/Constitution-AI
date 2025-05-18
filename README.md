# Kazakhstan Constitution AI Assistant

An AI-powered assistant that can answer questions about the Constitution of the Republic of Kazakhstan and process additional documents for context-aware responses.

## Features

- Interactive chat interface using Streamlit
- Integration with OpenAI's GPT-3.5 Turbo model
- Vector storage using ChromaDB for efficient document retrieval
- Support for multiple document formats (PDF, DOCX, TXT)
- Automatic scraping and processing of the Constitution text
- Chat history tracking
- Document upload functionality (single or multiple files)

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

Alternatively, you can input your API key directly in the application's sidebar.

## Running the Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

3. If you haven't set the API key in the `.env` file, enter it in the sidebar

4. Start asking questions about the Constitution of Kazakhstan!

## Using the Application

1. The application automatically loads the Constitution of Kazakhstan when started
2. You can upload additional documents through the sidebar
3. Ask questions in the chat interface
4. The AI will provide answers based on the Constitution and any additional uploaded documents
5. Chat history is maintained during the session

## Supported File Types

- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Text files (`.txt`)

## Technical Details

- Uses LangChain for document processing and chat chain management
- Implements ChromaDB as a vector store for efficient document retrieval
- Utilizes OpenAI's embeddings for document vectorization
- Features recursive text splitting for optimal context management

## Note

Make sure you have a valid OpenAI API key with sufficient credits to use this application. The application uses the GPT-3.5 Turbo model for generating responses. 