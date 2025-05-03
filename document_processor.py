import os
# import PyPDF2 # No longer needed
import fitz  # PyMuPDF for faster PDF processing
import docx2txt
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def is_supported_file(file_extension):
    """Check if a file type is supported by the system."""
    return file_extension.lower() in ['pdf', 'docx', 'txt']

def extract_text_from_file(file_path, file_extension):
    """Extract text from various file formats."""
    file_extension = file_extension.lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf_pymupdf(file_path) # Use PyMuPDF
    elif file_extension == 'docx':
        return extract_text_from_docx(file_path)
    elif file_extension == 'txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

# Renamed PyPDF2 version (kept for reference, but not used)
# def extract_text_from_pdf_pypdf2(pdf_path):
#     """Extract text from a PDF file with page numbers using PyPDF2."""
#     text_with_pages = []
#     try:
#         with open(pdf_path, 'rb') as file:
#             pdf_reader = PyPDF2.PdfReader(file)
#             total_pages = len(pdf_reader.pages)
#             for page_num in range(total_pages):
#                 page = pdf_reader.pages[page_num]
#                 text = page.extract_text()
#                 if text and text.strip():  # Only add non-empty pages
#                     text_with_pages.append({
#                         'page_number': page_num + 1,  # 1-based page numbering
#                         'text': clean_text(text) # Clean text here
#                     })
#         return text_with_pages
#     except Exception as e:
#         raise Exception(f"Error extracting text from PDF (PyPDF2): {str(e)}")

def extract_text_from_pdf_pymupdf(pdf_path):
    """Extract text from a PDF file with page numbers using PyMuPDF (fitz)."""
    text_with_pages = []
    try:
        doc = fitz.open(pdf_path)  # Open the PDF
        for page_num, page in enumerate(doc):
            text = page.get_text("text") # Extract text
            if text and text.strip(): # Only add non-empty pages
                text_with_pages.append({
                    'page_number': page_num + 1,  # 1-based page numbering
                    'text': clean_text(text) # Clean text here
                })
        doc.close() # Close the document
        return text_with_pages
    except Exception as e:
        # Provide more context in the error message
        raise Exception(f"Error extracting text from PDF '{os.path.basename(pdf_path)}' using PyMuPDF: {str(e)}")


def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file (treating as one page)."""
    try:
        text = docx2txt.process(docx_path)
        if text and text.strip():
             # Clean text here
            return [{'page_number': 1, 'text': clean_text(text)}]
        else:
            return [] # Return empty list if no text extracted
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX '{os.path.basename(docx_path)}': {str(e)}")

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file (treating as one page)."""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file: # Added errors='ignore' for robustness
            text = file.read()
        if text and text.strip():
             # Clean text here
            return [{'page_number': 1, 'text': clean_text(text)}]
        else:
            return [] # Return empty list if no text extracted
    except Exception as e:
        raise Exception(f"Error extracting text from TXT '{os.path.basename(txt_path)}': {str(e)}")

def split_text_into_chunks(text_with_pages, chunk_size=1000, chunk_overlap=200):
    """
    Split text into smaller chunks while preserving page number information.

    Args:
        text_with_pages: List of dicts with page_number and text fields
        chunk_size: Target size of each text chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of dicts with text, page_number, and chunk_index fields
    """
    # Use default text splitter settings from Langchain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False, # Don't add start index metadata
    )

    chunks = []
    chunk_id_counter = 0 # Simple counter for unique chunk IDs within the document

    for page_info in text_with_pages:
        page_number = page_info['page_number']
        page_text = page_info['text']

        if not page_text or not page_text.strip():
            continue # Skip empty pages

        # Split the page text into chunks
        try:
            page_chunks = text_splitter.split_text(page_text)
        except Exception as e:
            print(f"Warning: Error splitting text on page {page_number}. Skipping page. Error: {e}")
            continue

        # Add each chunk with its metadata
        for i, chunk_text in enumerate(page_chunks):
            if chunk_text and chunk_text.strip(): # Ensure chunk is not empty
                chunks.append({
                    'text': chunk_text,
                    'page_number': page_number,
                    'chunk_index_in_page': i, # Index within the page
                    'doc_chunk_id': chunk_id_counter # Unique ID within the document being processed
                })
                chunk_id_counter += 1

    return chunks

def clean_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()
