import os
import PyPDF2
import docx2txt
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def is_supported_file(file_extension):
    """Check if a file type is supported by the system."""
    return file_extension.lower() in ['pdf', 'docx', 'txt']

def extract_text_from_file(file_path, file_extension):
    """Extract text from various file formats."""
    if file_extension.lower() == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension.lower() == 'docx':
        return extract_text_from_docx(file_path)
    elif file_extension.lower() == 'txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file with page numbers."""
    text_with_pages = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():  # Only add non-empty pages
                    text_with_pages.append({
                        'page_number': page_num + 1,  # 1-based page numbering
                        'text': text
                    })
        
        return text_with_pages
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file (without page numbers)."""
    try:
        text = docx2txt.process(docx_path)
        # For DOCX files, we don't have direct page number access, so we treat it as one page
        return [{'page_number': 1, 'text': text}]
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file (without page numbers)."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        # For TXT files, we treat it as one page
        return [{'page_number': 1, 'text': text}]
    except Exception as e:
        raise Exception(f"Error extracting text from TXT: {str(e)}")

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = []
    
    for page_info in text_with_pages:
        page_number = page_info['page_number']
        page_text = page_info['text']
        
        # Split the page text into chunks
        page_chunks = text_splitter.split_text(page_text)
        
        # Add each chunk with its metadata
        for i, chunk in enumerate(page_chunks):
            chunks.append({
                'text': chunk,
                'page_number': page_number,
                'chunk_index': i
            })
    
    return chunks

def clean_text(text):
    """Clean and normalize text."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text.strip()
