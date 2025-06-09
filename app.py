# app.py
# This file orchestrates the AI Copilot for Renewable Energy Data Rooms application.
# It implements the core Retrieval-Augmented Generation (RAG) pipeline
# as outlined in the project plan, with explicit document processing, decoupled display,
# and dynamic example question generation.

import os
import re 
from dotenv import load_dotenv  # Load environment variables from .env
load_dotenv()  # Load environment variables from .env
import tempfile
import streamlit as st
from collections import defaultdict 
try:
    from document_processor import extract_text_from_file, is_supported_file
    from document_store import DocumentStore 
    from qa_engine import get_answer_with_citations, generate_example_questions
    from utils import get_file_extension, create_session_state_if_not_exists, get_file_icon
except ImportError as e:
    st.error(f"Error importing required modules: {e}. Make sure document_processor.py, document_store.py, qa_engine.py, and utils.py are present.")
    st.stop() 

# --- Application Setup and Configuration ---

# Set environment variable for Gemini API
# This will work in Streamlit Cloud
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Or use this to handle both local .env and Streamlit Cloud secrets
gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not gemini_api_key or gemini_api_key == "YOUR_API_KEY_HERE":
    st.warning("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable.", icon="⚠️")
else:
    os.environ["GEMINI_API_KEY"] = gemini_api_key # Ensure it's set for qa_engine

# Set Streamlit page configuration
st.set_page_config(
    page_title="Renewable Energy Data Room Assistant",
    page_icon="🌱",
    layout="wide",
)

# --- Custom CSS Styling ---
# Includes styles for general layout, answer block, citation highlighting,
# improved citation expander/box styling, and example questions list.
st.markdown('''
<style>
    /* General Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px; /* Limit width for better readability */
        margin: auto; /* Center the container */
    }
    h1, h2, h3 {
        font-family: "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
        color: #1B5E20; /* Darker Green */
    }
    h1 { font-weight: 700; margin-bottom: 1.5rem; }
    h2 { border-bottom: 2px solid #A5D6A7; padding-bottom: 0.5rem; margin-top: 2rem; }
    h3 { margin-top: 1.5rem; color: #2E7D32; }

    /* Sidebar Styles */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        background-color: #f8f9fa; /* Lighter gray */
        border-right: 1px solid #e9ecef; /* Subtle border */
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        padding-left: 0.5rem;
        color: #000000; /* Changed to black (User's change) */
    }
    .file-item {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        background-color: white;
        border-radius: 8px;
        margin-bottom: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .file-icon { margin-right: 10px; font-size: 1.2rem; color: #4CAF50; }
    .file-name { flex: 1; font-size: 0.9rem; }
    /* --- Style for the Dustbin (Remove File) Button --- */
    /* Note: User changed this button to type="primary", so specific CSS might not be needed */
    /* Keeping the rule in case type is changed back */
    .file-action button {
        color: #d32f2f !important; /* Red color for the icon */
        background: none !important; /* No background */
        border: none !important;
        padding: 0 !important;
        font-size: 1.1rem;
        line-height: 1;
        margin-left: 5px;
        box-shadow: none !important; /* Remove shadow */
        transform: none !important; /* Remove transform */
    }
    .file-action button:hover {
        color: #b71c1c !important; /* Darker red on hover */
        background: none !important; /* Ensure no background on hover */
    }
    .document-list { margin-top: 1rem; max-height: 400px; overflow-y: auto; padding-right: 10px; }

    /* --- General Button Styles (Green/White) --- */
    /* User seems to be using type="primary" for most buttons now */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        border: none;
        background-color: #4CAF50; /* Default background color (Green) */
        color: white !important; /* Default text color (White) */
        transition: all 0.2s;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #388E3C; /* Darker green on hover */
        color: white !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }
    /* Style for specifically designated 'secondary' buttons (like Cancel) */
    .stButton button[kind="secondary"] {
        background-color: #e0e0e0;
        color: #333 !important;
    }
    .stButton button[kind="secondary"]:hover {
        background-color: #bdbdbd;
        color: #000 !important;
    }
    /* Style for the main 'Submit Question' button (if type="submit" is used) */
    /* User changed this to type="primary", so this rule might not apply unless changed back */
    .stButton button[type="submit"] {
        background-color: #2E7D32; /* Specific darker green for primary action */
    }
    .stButton button[type="submit"]:hover {
        background-color: #1B5E20;
    }

    /* Answer Block Styling */
    .answer-block {
        background-color: #E8F5E9; /* Light green background */
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #4CAF50; /* Green accent border */
        margin: 1.5rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.07);
    }
    .answer-block h3 {
        color: #1B5E20; /* Darker green for header */
        margin-top: 0;
        display: flex;
        align-items: center;
        font-size: 1.3rem; /* Slightly larger header */
        margin-bottom: 1rem;
    }
    .answer-block h3 span { /* Style icon */
        margin-right: 10px;
        font-size: 1.5rem; /* Make icon slightly larger */
        color: #4CAF50;
    }
    .answer-block .answer-content {
        font-size: 1.05rem;
        line-height: 1.7; /* Increase line height for readability */
        color: #333; /* Darker text color */
    }

    /* Citation Highlighting in Answer */
    .source-highlight {
        background-color: #C8E6C9; /* Slightly darker green highlight */
        padding: 0.1em 0.4em;
        border-radius: 4px;
        font-size: 0.9em; /* Slightly smaller font for highlight */
        color: #1B5E20; /* Dark green text */
        font-weight: 500;
        display: inline-block; /* Ensures background covers text properly */
        margin: 0 2px; /* Add slight margin */
    }

    /* Citation Expander and Box Styling */
    div.stExpander, div[data-testid="stExpander"] {
        border-radius: 8px;
        border: 1px solid #BDBDBD; /* Slightly darker border */
        box-shadow: none; /* Remove default shadow */
        transition: all 0.3s;
        margin-bottom: 0.75rem; /* Increased space between expanders */
        background-color: #fafafa; /* Light background for expander */
    }
    div.stExpander:hover, div[data-testid="stExpander"]:hover {
        border-color: #4CAF50; /* Highlight border on hover */
        background-color: #ffffff;
    }
    div[data-testid="stExpander"] summary { /* Style expander header */
        font-weight: 500;
        color: #388E3C;
    }
    .citation-box {
        border-left: 4px solid #66BB6A; /* Slightly lighter green border inside */
        padding: 15px 20px; /* Increased padding */
        margin: 10px 0;
        background-color: #ffffff; /* White background inside */
        border-radius: 0 5px 5px 0;
    }
    .citation-box blockquote { /* Style the quoted text */
        margin: 0.5em 0 1em 0; /* Adjust margins */
        padding: 0.5em 1em;
        border-left: 3px solid #AED581; /* Inner border for quote */
        background-color: #F1F8E9; /* Very light green for quote background */
        color: #424242; /* Darker text for quote */
        font-size: 0.95rem;
        line-height: 1.6;
        border-radius: 4px;
    }
    .citation-box blockquote p { margin-bottom: 0; }
    .citation-box p:last-child { margin-bottom: 0; }

    /* Message/Info Boxes */
    .info-box {
        background-color: #e8f5e9; padding: 1rem; border-radius: 8px; margin: 1rem 0; display: flex; align-items: center; border-left: 4px solid #4CAF50;
    }
    .warning-box {
        background-color: #fff3e0; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff9800; margin-bottom: 1rem;
    }
    .warning-box div { display: flex; align-items: center; }
    .warning-box span { font-size: 1.5rem; margin-right: 0.7rem; color: #ff9800; }
    .warning-box p { margin: 0; color: #e65100; }

    /* --- ADDED: Styling for Example Questions list --- */
    .example-questions-list ul {
        list-style-type: none; /* Remove default bullets */
        padding-left: 0;
        margin-top: 0.5rem;
    }
    .example-questions-list li {
        background-color: #f0f4f0; /* Light background for each question */
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        border-left: 3px solid #66BB6A;
        font-size: 0.95rem;
        color: #333;
    }
    .example-questions-list li::before { /* Optional: Add a small icon or marker */
        content: "›";
        margin-right: 8px;
        color: #388E3C;
        font-weight: bold;
    }

</style>
''', unsafe_allow_html=True)

# --- Session State Management ---
# Initialize session state variables if they don't exist.
create_session_state_if_not_exists() # Initializes 'document_store', 'processed_files'

# Add state variables for managing file uploader key, clear confirmation, and dynamic examples
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0 # Initial key value
if 'confirm_clear_pending' not in st.session_state:
    st.session_state.confirm_clear_pending = False
# --- ADDED: Session state for dynamic examples ---
if 'dynamic_examples' not in st.session_state:
    # Initialize with default static questions
    st.session_state.dynamic_examples = [
        "What is the total installed capacity mentioned?",
        "Summarize the environmental impact assessment.",
        "Describe the financing structure or PPA details.",
        "What are the land lease terms mentioned?",
        "List the main permitting requirements outlined."
    ]
if 'generating_examples' not in st.session_state:
    st.session_state.generating_examples = False # Flag to show loading state


# --- Helper Function for Highlighting ---
def highlight_citations_in_text(text):
    """Uses regex to find bracketed text and wrap it in a span for CSS styling."""
    pattern = r'(\[.*?\])'
    highlighted_text = re.sub(pattern, r'<span class="source-highlight">\1</span>', text)
    return highlighted_text

# --- ADDED: Function to Trigger Example Generation ---
def trigger_example_generation():
    """Sets the flag to start generating example questions based on current files."""
    # Check if there are processed files and we are not already generating
    if st.session_state.processed_files and not st.session_state.generating_examples:
        st.session_state.generating_examples = True
        # We need to rerun the script so the expander shows the spinner state
        st.rerun()

# --- UI Layout and Content ---

# Page Title and Description
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <div style="font-size: 2.5rem; margin-right: 0.8rem;">🌱</div>
    <div>
        <h1 style="margin: 0; padding: 0;">Renewable Energy Data Room Assistant</h1>
        <p style="margin: 0; color: #555; font-size: 1.1rem;">Your intelligent guide to renewable energy project information</p>
    </div>
</div>
<div style="padding: 1.2rem; background: linear-gradient(90deg, #e3f2fd, #f1f8e9); border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
    <p style="margin: 0; font-size: 1.1rem;">Upload your project documents (PDF, DOCX, TXT) and ask questions to get answers with specific citations.</p>
</div>
""", unsafe_allow_html=True)


# --- Sidebar for Document Management ---
with st.sidebar:
    st.header("📁 Document Management")
    st.subheader("Upload Documents")

    # File Uploader Widget
    uploaded_files = st.file_uploader(
        "Select PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}",
    )

    # Explicit Document Processing Button
    if uploaded_files:
        new_files_selected = any(f.name not in st.session_state.processed_files for f in uploaded_files)

        if new_files_selected:
            # User changed this button to type="primary"
            if st.button("Process Selected Files", help="Process any newly selected documents", type="primary"):
                files_to_process_this_time = [
                    f for f in uploaded_files
                    if f.name not in st.session_state.processed_files
                ]
                processed_successfully = False # Flag to track if any file was processed
                if files_to_process_this_time:
                    progress_bar = st.progress(0)
                    num_files = len(files_to_process_this_time)
                    with st.spinner("Processing documents..."): # Add spinner for processing
                        for i, uploaded_file in enumerate(files_to_process_this_time):
                            file_extension = get_file_extension(uploaded_file.name)
                            if not is_supported_file(file_extension):
                                st.error(f"Unsupported file type: {uploaded_file.name}")
                                continue

                            st.info(f"Processing {uploaded_file.name}...")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                temp_file_path = tmp_file.name
                            try:
                                extracted_text = extract_text_from_file(temp_file_path, file_extension)
                                if not extracted_text:
                                    st.warning(f"Could not extract text from {uploaded_file.name}. It might be empty or image-based.")
                                    continue

                                document_id = st.session_state.document_store.add_document(
                                    filename=uploaded_file.name,
                                    content=extracted_text
                                )

                                if document_id:
                                    st.session_state.processed_files[uploaded_file.name] = document_id
                                    st.success(f"✓ Processed: {uploaded_file.name}")
                                    processed_successfully = True # Mark success
                                else:
                                    st.error(f"✗ Failed to add document '{uploaded_file.name}' to store.")

                            except Exception as e:
                                st.error(f"✗ Error processing {uploaded_file.name}: {str(e)}")
                                if uploaded_file.name in st.session_state.processed_files:
                                    del st.session_state.processed_files[uploaded_file.name]
                            finally:
                                if os.path.exists(temp_file_path):
                                    os.remove(temp_file_path)
                            progress_bar.progress((i + 1) / num_files)
                    progress_bar.empty()

                    # --- MODIFIED: Trigger Dynamic Example Generation AFTER processing ---
                    if processed_successfully:
                        trigger_example_generation() # Call the function to start generation
                    else:
                        st.rerun() # Rerun just to update UI state if nothing was processed

                else:
                     st.info("No new files to process.")


    # Display Uploaded Documents and Deletion Options
    if st.session_state.processed_files:
        st.subheader("Active Documents")
        st.markdown('<div class="document-list">', unsafe_allow_html=True)
        sorted_filenames = sorted(list(st.session_state.processed_files.keys()))
        for filename in sorted_filenames:
            file_extension = get_file_extension(filename)
            file_icon = get_file_icon(file_extension)
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"""
                <div class="file-item">
                    <span class="file-icon">{file_icon}</span>
                    <span class="file-name" title="{filename}">{filename[:35] + '...' if len(filename) > 35 else filename}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                button_key = f"remove_{filename}_{st.session_state.file_uploader_key}"
                # User changed this button to type="primary"
                # The .file-action CSS rule might not apply unless type is changed back
                st.markdown('<div class="file-action">', unsafe_allow_html=True) # Keep class for potential future styling
                if st.button("🗑️", key=button_key, help=f"Remove {filename}", type="primary"):
                    document_id = st.session_state.processed_files.pop(filename)
                    st.session_state.document_store.delete_document(document_id)
                    st.toast(f"Removed '{filename}'", icon="🗑️")
                    st.session_state.confirm_clear_pending = False
                    # --- MODIFIED: Trigger example regeneration if files are removed ---
                    trigger_example_generation()
                    # st.rerun() # trigger_example_generation will cause rerun
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Clear All Documents Option
        st.markdown("---")
        # User changed this button to type="primary"
        if st.button("Clear All Documents", type="primary", help="Remove all processed documents"):
            st.session_state.confirm_clear_pending = True
            st.rerun()

        if st.session_state.confirm_clear_pending:
            st.warning("⚠️ Are you sure you want to remove ALL documents?")
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                # User changed this button to type="primary"
                if st.button("Confirm Clear", key="confirm_clear_button", type="primary"):
                    st.session_state.document_store.clear_all()
                    st.session_state.processed_files = {}
                    st.session_state.file_uploader_key += 1
                    st.session_state.confirm_clear_pending = False
                    # --- MODIFIED: Reset dynamic examples when clearing all ---
                    st.session_state.dynamic_examples = [ # Re-initialize with defaults
                        "What is the total installed capacity mentioned?",
                        "Summarize the environmental impact assessment.",
                        "Describe the financing structure or PPA details.",
                        "What are the land lease terms mentioned?",
                        "List the main permitting requirements outlined."
                    ]
                    st.session_state.generating_examples = False # Reset flag
                    st.toast("All documents cleared.", icon="✅")
                    st.rerun()
            with col_confirm2:
                 # Keep Cancel as secondary for less emphasis
                 if st.button("Cancel", key="cancel_clear_button", type="secondary"):
                     st.session_state.confirm_clear_pending = False
                     st.rerun()
    else:
        # Message when no documents are processed
        st.markdown("""
        <div style="padding: 20px; text-align: center; border-radius: 10px; background-color: #e9ecef; margin-top: 1rem; border: 1px dashed #ced4da;">
            <span style="font-size: 2rem;">📂</span>
            <p style="margin-top: 0.5rem; color: #6c757d;">No documents processed yet.<br>Upload files and click 'Process'.</p>
        </div>
        """, unsafe_allow_html=True)


# --- Main Content Area for Question Answering ---
main_container = st.container()

with main_container:
    # User removed border-bottom from this h2
    st.markdown('<h2 style="border-bottom: none;"><span style="margin-right: 10px;">💬</span> Ask a Question</h2>', unsafe_allow_html=True)

    if not st.session_state.processed_files:
        st.markdown('''
        <div class="warning-box">
             <div>
                 <span>⚠️</span>
                 <p>Please upload and process documents using the sidebar before asking questions.</p>
             </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        user_question = st.text_area(
            "Enter your question about the documents:",
            height=100,
            placeholder="Example: What is the total capacity of the solar project mentioned in [Project_Spec.pdf]?",
            key="user_question_input"
            )

        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
             # User changed this button to type="primary"
            submit_button = st.button("🔍 Get Answer", use_container_width=True, type="primary")

        if submit_button and user_question:
            with st.spinner("🧠 Analyzing documents and generating answer..."):
                try:
                    answer, citations = get_answer_with_citations(
                        question=user_question,
                        document_store=st.session_state.document_store
                    )
                    highlighted_answer = highlight_citations_in_text(answer)

                    st.markdown(f'''
                    <div class="answer-block">
                        <h3><span>💡</span> Answer</h3>
                        <div class="answer-content">{highlighted_answer}</div>
                    </div>
                    ''', unsafe_allow_html=True)

                    if citations:
                        st.markdown('''
                        <h3 style="display: flex; align-items: center; margin-top: 2rem; margin-bottom: 1rem;">
                            <span style="margin-right: 10px; font-size: 1.3rem;">📚</span> References
                        </h3>
                        ''', unsafe_allow_html=True)
                        grouped_citations = defaultdict(list)
                        for citation in citations:
                             if isinstance(citation, dict) and all(k in citation for k in ['filename', 'page_number', 'text']):
                                try:
                                    page_num = str(citation.get('page_number', 'N/A'))
                                    filename = citation.get('filename', 'Unknown Document')
                                    text = citation.get('text', '').strip()
                                    key = (filename, page_num)
                                    if text and text not in grouped_citations[key]:
                                        grouped_citations[key].append(text)
                                except Exception as group_e:
                                    print(f"Warning: Error grouping citation: {citation} - {group_e}")
                                    continue
                             else:
                                print(f"Warning: Skipping invalid or incomplete citation format: {citation}")
                                continue

                        if grouped_citations:
                            citation_count = 0
                            def sort_key(item):
                                filename, page_num_str = item[0]
                                try: return (filename, int(page_num_str))
                                except ValueError: return (filename, float('inf'))
                            sorted_keys = sorted(grouped_citations.keys(), key=lambda k: sort_key((k, None)))

                            for (filename, page_number) in sorted_keys:
                                texts = grouped_citations[(filename, page_number)]
                                citation_count += 1
                                expander_title = f"Reference {citation_count}: {filename} (Page {page_number})"
                                with st.expander(expander_title):
                                    st.markdown('<div class="citation-box">', unsafe_allow_html=True)
                                    for text in texts:
                                        st.markdown(f"> {text}", unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                        else:
                             st.markdown('''
                             <div class="info-box">
                                 <span style="font-size: 1.2rem; margin-right: 0.7rem;">ℹ️</span>
                                 <p style="margin: 0;">Citations were found but could not be displayed properly.</p>
                             </div>
                             ''', unsafe_allow_html=True)
                    else:
                        st.markdown('''
                        <div class="info-box">
                            <span style="font-size: 1.2rem; margin-right: 0.7rem;">ℹ️</span>
                            <p style="margin: 0;">No specific references were found for this answer in the provided documents.</p>
                        </div>
                        ''', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {str(e)}")

        elif submit_button and not user_question:
             st.warning("Please enter a question before submitting.", icon="⚠️")


    # --- Example Questions and About Sections ---
    st.markdown("---") # Separator
    col_exp1, col_exp2 = st.columns(2)

    # --- MODIFIED: Dynamic Example Questions Section ---
    with col_exp1:
        # Determine the title based on the generation state
        expander_title = "❓ Example Questions"
        if st.session_state.generating_examples:
             expander_title = "❓ Generating Example Questions..."

        # Use the generating flag to control the expander state and content
        with st.expander(expander_title, expanded=st.session_state.generating_examples):
            if st.session_state.generating_examples:
                # Show spinner and call the generation function *inside* the spinner context
                with st.spinner("Asking the AI for relevant examples..."):
                    current_filenames = list(st.session_state.processed_files.keys())
                    # Call the function from qa_engine to get new examples
                    st.session_state.dynamic_examples = generate_example_questions(current_filenames)
                    # Turn off the flag *after* generation is complete
                    st.session_state.generating_examples = False
                    # Rerun to update the expander title and display the new questions
                    st.rerun()
            else:
                # Display the examples (either newly generated or default/previous)
                if st.session_state.dynamic_examples:
                    # Use the CSS class for styling the list
                    st.markdown('<div class="example-questions-list"><ul>', unsafe_allow_html=True)
                    for q in st.session_state.dynamic_examples:
                        # Display each question as a list item
                        st.markdown(f"<li>{q}</li>", unsafe_allow_html=True)
                    st.markdown('</ul></div>', unsafe_allow_html=True)

                    # Add a button to refresh examples manually (use primary style as per user's changes)
                    if st.button("🔄 Refresh Examples", key="refresh_examples_button", help="Generate new examples based on current documents", type="primary"):
                         trigger_example_generation() # Call the trigger function on click
                else:
                    # Fallback message if examples list is somehow empty
                    st.write("No example questions available.")


    # --- About Section (Kept from user's version) ---
    with col_exp2:
        with st.expander("ℹ️ About this Assistant"):
            st.markdown("""
            This AI assistant helps analyze renewable energy project documents:
            1.  Extracts text from uploaded PDFs, DOCX, and TXT files.
            2.  Uses AI to understand document content.
            3.  Answers questions based *only* on the provided documents.
            4.  Provides citations pointing to the source document and page number.

            **Tips:**
            * Ask specific questions.
            * Ensure documents are text-searchable.

            **Limitations:**
            * Accuracy depends on document quality.
            * Cannot answer questions beyond the scope of the files.
            * Complex tables/diagrams might not be fully interpreted.
            """)

