# Renewable Energy Data Room Assistant 🌱

An AI-powered assistant designed to help analysts, developers, and investors efficiently analyze and extract critical insights from renewable energy project documents. Built using Python, Google Gemini, FAISS, and Streamlit.

## The Challenge
Professionals in the renewable energy sector often face the daunting task of reviewing extensive data rooms filled with technical specifications, permits, financial models, and legal agreements. Manually extracting key information is time-consuming, inefficient, and prone to oversight.

## Features ✨

- 📄 Multi-Format Document Upload: Process PDF, DOCX, and TXT files
- 🧠 AI-Powered Q&A: Leverage Google's Gemini 1.5 Pro
- 🔍 Context-Aware Retrieval (RAG): Using FAISS vector search
- 📚 Source Citations: Automatic citations with page numbers
- 💡 Dynamic Example Questions: Based on uploaded documents
- 🔄 Real-time Processing: Immediate document indexing
- ⚙️ Modular Codebase: Well-structured Python modules
- 💻 Interactive UI: Built with Streamlit

## Technology Stack 🛠️

- Core Language: Python 3.x
- LLM: Google Gemini 1.5 Pro (via google-generativeai)
- Embeddings: Google embedding-001
- Vector Store: FAISS (faiss-cpu)
- Document Parsing: PyMuPDF (fitz), docx2txt
- Text Splitting: RecursiveCharacterTextSplitter
- Web Framework: Streamlit
- Environment Management: python-dotenv

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd RenewableDataHelper
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your Gemini API key:
- Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Set it as an environment variable:
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload your renewable energy project documents:
   - Click "Upload Documents" in the sidebar
   - Select PDF, DOCX, or TXT files
   - Wait for processing to complete

4. Ask questions about your documents:
   - Type your question in the text area
   - Click "Submit Question"
   - View the answer with relevant citations

## Workflow ⚙️

1. **Upload:**
   - User uploads documents via Streamlit interface
   - Supports PDF, DOCX, and TXT formats

2. **Process:**
   - Text extraction using PyMuPDF/docx2txt
   - Content cleaning and preprocessing
   - Splitting into manageable chunks
   - Vector embedding generation
   - FAISS indexing with metadata

3. **Query:**
   - Question embedding and similarity search
   - Context retrieval from relevant chunks
   - Answer generation with Gemini LLM
   - Citation formatting and validation

## Project Structure

```
RenewableDataHelper/
├── app.py                 # Main Streamlit application
├── gemini_api.py         # Google Gemini API integration
├── document_processor.py  # Document parsing and chunking
├── document_store.py     # Vector storage and document management
├── qa_engine.py          # Question answering logic
├── database.py           # Database operations
└── utils.py             # Utility functions
```

## Configuration

Key parameters can be adjusted in their respective files:

- `qa_engine.py`: Adjust generation parameters
```python
generation_config = {
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2000,
}
```

- `document_processor.py`: Modify chunking parameters
```python
chunk_size=1000
chunk_overlap=200
```

## Dependencies

Major dependencies include:
- `streamlit`: Web interface
- `google-generativeai`: Gemini AI API
- `PyMuPDF`: PDF processing
- `faiss-cpu`: Vector similarity search
- `pymongo`: Database operations (optional)
- `docx2txt`: Word document processing

## Challenges & Limitations 🚧

- LLM Output Formatting: Requires careful prompt engineering
- Scalability: Limited by in-memory FAISS index
- Document Complexity: No support for scanned PDFs or complex tables
- Chunking Strategy: Performance varies with document structure
- API Dependency: Requires Gemini API key and internet connectivity

## Future Improvements 🔮

- Support for Excel and CSV files
- Integration with persistent vector databases
- Enhanced table and figure handling
- User feedback mechanism
- Advanced metadata extraction capabilities

## Error Handling

The application includes robust error handling for:
- Invalid API keys
- Document processing failures
- Network issues
- Invalid file types
- Memory constraints

## Limitations

- Currently optimized for text-based content
- Image analysis is not yet supported
- Maximum file size depends on available memory
- Requires stable internet connection for AI operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Team Contributors 👥

**Team Unthinkables**
- Kanishk
- Md Azlan

## License

This project is licensed under the MIT License - see the LICENSE file for details.
