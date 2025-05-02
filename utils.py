import os
import streamlit as st
from document_store import DocumentStore

def get_file_extension(filename):
    """Get the extension of a file."""
    return os.path.splitext(filename)[1][1:].lower()

def create_session_state_if_not_exists():
    """Initialize session state variables if they don't exist."""
    if 'document_store' not in st.session_state:
        st.session_state.document_store = DocumentStore()
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}  # Maps filename to document_id
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Store previous Q&A

def format_citation(citation):
    """Format a citation for display."""
    return f"{citation['filename']} (Page {citation['page_number']})"

def get_file_icon(file_extension):
    """Get an icon for a file based on its extension."""
    if file_extension.lower() == 'pdf':
        return "📄"
    elif file_extension.lower() == 'docx':
        return "📝"
    elif file_extension.lower() == 'txt':
        return "📋"
    else:
        return "📁"

def truncate_text(text, max_length=100):
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def get_mime_type(file_extension):
    """Get the MIME type for a file extension."""
    extension_map = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain',
    }
    return extension_map.get(file_extension.lower(), 'application/octet-stream')
