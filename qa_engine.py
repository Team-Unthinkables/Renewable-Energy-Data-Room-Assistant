import os
import json
import re
import google.generativeai as genai

# Set up the Gemini API with the provided key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCL7WcFojRcescuZfdGw4iK_syGD3YfG5E")
genai.configure(api_key=GEMINI_API_KEY)

# Configure the model
generation_config = {
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1500,
}

# Get the Gemini model
def get_gemini_model():
    try:
        # Use Gemini 1.5 Pro model for better performance with structured data
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config
        )
        return model
    except Exception as e:
        raise Exception(f"Error initializing Gemini model: {str(e)}")

def get_answer_with_citations(question, document_store, max_context_chunks=8):
    """
    Get an answer to a question with citations from the document store.
    
    Args:
        question: User's question
        document_store: Document store instance
        max_context_chunks: Maximum number of chunks to include in context
        
    Returns:
        Tuple of (answer_text, citations)
    """
    # Search for relevant document chunks
    search_results = document_store.similarity_search(question, k=max_context_chunks)
    
    if not search_results:
        return "I couldn't find any relevant information in the uploaded documents.", []
    
    # Format context from search results
    context = ""
    for i, (chunk_text, metadata, _) in enumerate(search_results):
        context += f"\nDocument {i+1}: {metadata['filename']} (Page {metadata['page_number']})\n"
        context += f"{chunk_text}\n"
    
    # Create a prompt for the LLM
    prompt = f"""
    You are an expert assistant specialized in renewable energy projects. Answer the user's question based ONLY on the information provided in the context below.
    
    If the answer cannot be found in the context, say "I couldn't find information about that in the uploaded documents."
    
    When you use information from the context, always cite the source document and page number.
    Format your citations like this: [Filename, Page X]
    
    Context:
    {context}
    
    User Question: {question}
    
    Provide a comprehensive answer with specific citations to document sources. After your answer, include a "CITATIONS" section that lists all the specific quotes you used with their exact document name and page number.
    
    Format your response as JSON with these fields:
    - answer: your detailed answer with inline citations
    - citations: an array of objects with fields:
      - filename: document filename
      - page_number: page number
      - text: the exact text you're citing
    """
    
    # Call the Gemini API
    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Try to parse the response to extract answer and citations
        try:
            # Check if there's a JSON-like structure
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            
            if start_index >= 0 and end_index > start_index:
                json_str = response_text[start_index:end_index+1]
                result = json.loads(json_str)
                answer = result.get("answer", "")
                citations = result.get("citations", [])
            else:
                # If no JSON structure, try to parse manually
                parts = response_text.split("CITATIONS")
                answer = parts[0].strip()
                citations = []
                
                if len(parts) > 1:
                    # Try to extract citations manually from text
                    citation_text = parts[1].strip()
                    extracted_citations = extract_citations_from_text(citation_text)
                    citations = []
                    
                    for citation in extracted_citations:
                        if 'filename' in citation and 'page_number' in citation:
                            # Look for the text in the search results
                            for chunk_text, metadata, _ in search_results:
                                if metadata['filename'] == citation['filename'] and metadata['page_number'] == citation['page_number']:
                                    citation['text'] = chunk_text[:150] + "..."  # Use first 150 chars as citation text
                                    citations.append(citation)
                                    break
                
            return answer, citations
        except Exception as parsing_error:
            # If parsing fails, return the raw text and extract citations
            print(f"Error parsing Gemini response: {parsing_error}")
            citations = extract_citations_from_text(response_text)
            return response_text, citations
            
    except Exception as e:
        raise Exception(f"Error generating answer with Gemini: {str(e)}")

def extract_citations_from_text(answer_text):
    """
    Extract citation references from the answer text.
    
    Args:
        answer_text: Text containing citation references like [Filename, Page X]
        
    Returns:
        List of citation references
    """
    # Pattern to match citations in the format [Filename, Page X]
    citation_pattern = r'\[(.*?), Page (\d+)\]'
    citations = re.findall(citation_pattern, answer_text)
    
    # Convert to structured citations
    structured_citations = []
    for filename, page_number in citations:
        structured_citations.append({
            'filename': filename.strip(),
            'page_number': int(page_number)
        })
    
    return structured_citations
