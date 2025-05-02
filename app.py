import os
import tempfile
import streamlit as st
from document_processor import extract_text_from_file, is_supported_file
from document_store import DocumentStore
from qa_engine import get_answer_with_citations
from utils import get_file_extension, create_session_state_if_not_exists

# Set environment variable for Gemini API
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "AIzaSyCL7WcFojRcescuZfdGw4iK_syGD3YfG5E")

# Set page configuration
st.set_page_config(
    page_title="Renewable Energy Data Room Assistant",
    page_icon="🌱",
    layout="wide",
)

# Initialize session state variables
create_session_state_if_not_exists()

# Page title and description
st.title("🌱 Renewable Energy Data Room Assistant")
st.markdown("""
Upload your renewable energy project documents and ask questions to get answers 
with specific citations from the source materials.
""")

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
        for filename in st.session_state.processed_files.keys():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(filename)
            with col2:
                if st.button("❌", key=f"remove_{filename}"):
                    document_id = st.session_state.processed_files[filename]
                    st.session_state.document_store.delete_document(document_id)
                    del st.session_state.processed_files[filename]
                    st.rerun()
    else:
        st.info("No documents uploaded yet.")

    # Clear all documents button
    if st.session_state.processed_files:
        if st.button("Clear All Documents"):
            st.session_state.document_store.clear_all()
            st.session_state.processed_files = {}
            st.rerun()

# Main content area
main_container = st.container()

with main_container:
    # Question input
    st.subheader("Ask a Question")
    if not st.session_state.processed_files:
        st.warning("Please upload documents before asking questions.")
    else:
        user_question = st.text_input("Enter your question about the renewable energy project documents:")
        
        if st.button("Submit Question", use_container_width=True):
            if not user_question:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing documents and generating answer..."):
                    try:
                        answer, citations = get_answer_with_citations(
                            question=user_question,
                            document_store=st.session_state.document_store
                        )
                        
                        # Display the answer
                        st.subheader("Answer")
                        st.write(answer)
                        
                        # Display citations
                        if citations:
                            st.subheader("Citations")
                            for i, citation in enumerate(citations, 1):
                                with st.expander(f"Citation {i}: {citation['filename']} (Page {citation['page_number']})"):
                                    st.markdown(f"```\n{citation['text']}\n```")
                        else:
                            st.info("No specific citations found for this answer.")
                    except Exception as e:
                        st.error(f"Error processing your question: {str(e)}")
    
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
