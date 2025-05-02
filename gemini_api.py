import os
import json
import google.generativeai as genai

# Set up the Gemini API with the provided key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
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

# Function to get answer with citations
def get_answer_with_gemini(question, context):
    """Use Gemini to get an answer with citations based on the provided context."""
    try:
        model = get_gemini_model()
        
        # Create the prompt
        prompt = f"""
        You are an expert assistant specialized in renewable energy projects. Answer the user's question based ONLY on the information provided in the context below.
        
        If the answer cannot be found in the context, say "I couldn't find information about that in the uploaded documents."
        
        When you use information from the context, always cite the source document and page number.
        Format your citations like this: [Filename, Page X]
        
        Context:
        {context}
        
        User Question: {question}
        
        Provide a comprehensive answer with specific citations to document sources. At the end of your answer, include a "CITATIONS" section that lists all the specific quotes you used with their exact document name and page number.
        
        Format your response as structured data with these exact components:
        1. First provide your detailed answer with inline citations
        2. Then provide a list of citations with these exact fields for each citation:
           - filename: document filename
           - page_number: page number as an integer
           - text: the exact text you're citing
        
        Make sure to use proper JSON structure in your response.
        """
        
        # Get response from Gemini
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
                    citation_lines = parts[1].strip().split("\n")
                    for line in citation_lines:
                        if "[" in line and "]" in line:
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                text = parts[1].strip()
                                citation_parts = parts[0].strip()
                                filename_page = citation_parts.strip("[]").split(", Page ")
                                if len(filename_page) == 2:
                                    filename = filename_page[0]
                                    try:
                                        page_number = int(filename_page[1])
                                        citations.append({
                                            "filename": filename,
                                            "page_number": page_number,
                                            "text": text
                                        })
                                    except ValueError:
                                        pass
                
            return answer, citations
        except Exception as parsing_error:
            # If JSON parsing fails, return the raw text
            print(f"Error parsing Gemini response: {parsing_error}")
            return response_text, []
            
    except Exception as e:
        raise Exception(f"Error generating answer with Gemini: {str(e)}")

# Embed text using Gemini embeddings
def get_embeddings(texts):
    """Get embeddings for a list of texts using Google's embedding model."""
    try:
        # Create an embedding model
        embedding_model = genai.get_model("models/embedding-001")
        
        # Get embeddings for the texts
        embeddings = []
        for text in texts:
            result = embedding_model.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_query"
            )
            embeddings.append(result.embedding)
            
        return embeddings
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")
