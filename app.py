import os
import tempfile
import streamlit as st
from document_processor import extract_text_from_file, is_supported_file
from document_store import DocumentStore
from qa_engine import get_answer_with_citations
from utils import get_file_extension, create_session_state_if_not_exists, get_file_icon

# Set environment variable for Gemini API
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "AIzaSyCL7WcFojRcescuZfdGw4iK_syGD3YfG5E")

# Set page configuration
st.set_page_config(
    page_title="Renewable Energy Data Room Assistant",
    page_icon="🌱",
    layout="wide",
)

# Custom CSS for a more modern UI
st.markdown('''
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card-like styling for containers */
    div.stExpander, div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s;
    }
    div.stExpander:hover, div[data-testid="stExpander"]:hover {
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Citations styling */
    .citation-box {
        border-left: 4px solid #4CAF50;
        padding-left: 10px;
        margin: 10px 0;
        background-color: #f9f9f9;
        border-radius: 0 5px 5px 0;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        border: none;
        background-color: #4CAF50;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-family: "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
        color: #2E7D32;
    }
    h1 {
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        background-color: #f5f5f5;
        box-shadow: inset -2px 0 5px rgba(0,0,0,0.05);
        height: 100%;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        padding-left: 0.5rem;
    }
    
    /* File items styling */
    .file-item {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        background-color: white;
        border-radius: 8px;
        margin-bottom: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .file-icon {
        margin-right: 10px;
        font-size: 1.2rem;
    }
    .file-name {
        flex: 1;
    }
    .file-action {
        color: #d32f2f;
        cursor: pointer;
    }
    
    /* Document list container */
    .document-list {
        margin-top: 1rem;
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
</style>
''', unsafe_allow_html=True)

# Initialize session state variables
create_session_state_if_not_exists()

# Page title and description with a more attractive header
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <div style="font-size: 2.5rem; margin-right: 0.8rem;">🌱</div>
    <div>
        <h1 style="margin: 0; padding: 0;">Renewable Energy Data Room Assistant</h1>
        <p style="margin: 0; color: #666; font-size: 1.1rem;">Your intelligent guide to renewable energy project information</p>
    </div>
</div>

<div style="padding: 1.2rem; background: linear-gradient(90deg, #e3f2fd, #f1f8e9); border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
    <p style="margin: 0; font-size: 1.1rem;">This AI assistant helps you extract insights from renewable energy project documents. Upload your files and ask questions to get answers with specific citations to the source materials.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with document management
with st.sidebar:
    st.header("Document Management")
    
    # Upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or other document files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = get_file_extension(uploaded_file.name)
            
            if not is_supported_file(file_extension):
                st.error(f"Unsupported file type: {file_extension}")
                continue
                
            if uploaded_file.name not in st.session_state.processed_files:
                st.info(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name
                
                try:
                    # Extract text from the file
                    extracted_text = extract_text_from_file(temp_file_path, file_extension)
                    
                    # Add document to the document store
                    document_id = st.session_state.document_store.add_document(
                        filename=uploaded_file.name,
                        content=extracted_text
                    )
                    
                    # Update session state
                    st.session_state.processed_files[uploaded_file.name] = document_id
                    st.success(f"Successfully processed {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.remove(temp_file_path)
    
    # Document List
    if st.session_state.processed_files:
        st.subheader("Uploaded Documents")
        
        # Container for document list with scrollbar
        st.markdown('<div class="document-list">', unsafe_allow_html=True)
        
        for filename in st.session_state.processed_files.keys():
            file_extension = get_file_extension(filename)
            file_icon = get_file_icon(file_extension)
            
            # Create a stylish file item
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class="file-item">
                    <span class="file-icon">{file_icon}</span>
                    <span class="file-name">{filename}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("❌", key=f"remove_{filename}", help=f"Remove {filename}"):
                    document_id = st.session_state.processed_files[filename]
                    st.session_state.document_store.delete_document(document_id)
                    del st.session_state.processed_files[filename]
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 20px; text-align: center; border-radius: 10px; background-color: #f5f5f5;">
            <img src="https://img.icons8.com/ios-filled/50/2E7D32/document.png" style="width: 60px; margin-bottom: 10px;">
            <p>No documents uploaded yet. Add some files to get started!</p>
        </div>
        """, unsafe_allow_html=True)

    # Clear all documents button
    if st.session_state.processed_files:
        if st.button("Clear All Documents"):
            st.session_state.document_store.clear_all()
            st.session_state.processed_files = {}
            st.rerun()

# Main content area with a card-like container for the question section
st.markdown('''
<div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
    <h2 style="color: #2E7D32; margin-top: 0; display: flex; align-items: center;">
        <span style="margin-right: 10px;">💬</span> Ask a Question
    </h2>
</div>
''', unsafe_allow_html=True)

main_container = st.container()

with main_container:
    # Question input section
    if not st.session_state.processed_files:
        st.markdown('''
        <div style="background-color: #fff3e0; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff9800; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 0.7rem;">⚠️</span>
                <p style="margin: 0;">Please upload documents using the sidebar before asking questions.</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        # Create a stylish question input box
        user_question = st.text_area(
            "Enter your question about the renewable energy project documents:", 
            height=100,
            placeholder="Example: What is the total capacity of the wind farm project?")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.button("🔍 Submit Question", use_container_width=True)
        
        if submit_button:
            if not user_question:
                st.markdown('''
                <div style="background-color: #fff3e0; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff9800; margin: 1rem 0;">
                    <p style="margin: 0;">Please enter a question first.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                with st.spinner("Analyzing documents and generating answer..."):
                    try:
                        answer, citations = get_answer_with_citations(
                            question=user_question,
                            document_store=st.session_state.document_store
                        )
                        
                        # Display the answer in a nice card
                        st.markdown(f'''
                        <div style="background-color: #f1f8e9; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 1.5rem 0;">
                            <h3 style="color: #2E7D32; margin-top: 0; display: flex; align-items: center;">
                                <span style="margin-right: 10px;">💡</span> Answer
                            </h3>
                            <p style="font-size: 1.1rem; line-height: 1.6;">{answer}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Display citations
                        if citations:
                            st.markdown('''
                            <h3 style="color: #2E7D32; display: flex; align-items: center; margin-top: 2rem;">
                                <span style="margin-right: 10px;">📚</span> Citations
                            </h3>
                            ''', unsafe_allow_html=True)
                            for i, citation in enumerate(citations, 1):
                                with st.expander(f"Citation {i}: {citation['filename']} (Page {citation['page_number']})"):
                                    st.markdown('''
                                    <div class="citation-box">
                                    ''', unsafe_allow_html=True)
                                    st.write(citation['text'])
                                    st.markdown('''
                                    </div>
                                    ''', unsafe_allow_html=True)
                        else:
                            st.markdown('''
                            <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 8px; margin: 1rem 0; display: flex; align-items: center;">
                                <span style="font-size: 1.2rem; margin-right: 0.7rem;">ℹ️</span>
                                <p style="margin: 0;">No specific citations found for this answer.</p>
                            </div>
                            ''', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'''
                        <div style="background-color: #ffebee; padding: 1rem; border-radius: 8px; border-left: 4px solid #f44336; margin: 1rem 0;">
                            <p style="margin: 0;"><strong>Error: </strong>{str(e)}</p>
                        </div>
                        ''', unsafe_allow_html=True)
    
    # Example questions section
    with st.expander("Example Questions"):
        st.markdown("""
        Here are some example questions you can ask:
        
        - What is the total capacity of the wind farm project?
        - What are the main environmental considerations for this solar project?
        - What financing model is used for this renewable energy project?
        - What are the projected energy yields from this facility?
        - What permitting requirements are mentioned in the documents?
        """)

    # Information about the assistant
    with st.expander("About this Assistant"):
        st.markdown("""
        ### Renewable Energy Data Room Assistant
        
        This AI-powered assistant helps you analyze renewable energy project documents by:
        
        1. Extracting and processing text from uploaded documents
        2. Analyzing document content using advanced natural language processing
        3. Providing answers to your questions with specific citations to source documents
        
        **Supported File Types:**
        - PDF (.pdf)
        - Word Documents (.docx)
        - Text Files (.txt)
        
        **Limitations:**
        - The assistant can only answer based on the documents you upload
        - Very large documents might take longer to process
        - Highly technical or ambiguous questions may have varying accuracy
        """)
