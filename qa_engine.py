# qa_engine.py
# This module contains the core Question Answering logic, including
# retrieval of relevant document chunks, generation of answers,
# and generation of example questions using the Gemini model.

import os
import json
import re
import google.generativeai as genai
from database import db # Assuming db instance is correctly initialized and imported
from collections import defaultdict # Ensure defaultdict is imported
import random # For potentially selecting sample chunks if needed later

# --- API Key Configuration ---
# API key is configured globally in app.py after loading from environment variables.
# No need to configure genai here again.

# --- Model Configuration (Generation) ---
# These parameters control the behavior of the Gemini model when generating text.
# Values are chosen to prioritize factual, grounded answers from the provided context.
generation_config = {
    "temperature": 0.4,  # Lower values (0.0-0.6) make output more deterministic and focused.
                         # Recommended for factual Q&A to reduce hallucination.
                         # 0.4 provides a good balance of factuality and natural phrasing.
    "top_p": 0.9,        # Lower values (e.g., 0.7-0.9) sample from a smaller, higher-probability token set.
                         # Helps focus the model on relevant terms from the context.
    "top_k": 40,         # Lower values (e.g., 20-50) restrict sampling to top tokens.
                         # Works with top_p to control diversity and focus.
    "max_output_tokens": 2000, # Maximum length of the generated answer.
                              # Set high enough to allow comprehensive answers from detailed documents.
                              # Adjust based on typical answer length requirements.
}

# --- Model Configuration (Example Question Generation) ---
# Slightly higher temperature might encourage more diverse questions.
example_generation_config = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 500, # Shorter output needed for examples
}


# Get the Gemini model (using a function to potentially manage different model instances later)
def get_gemini_model(config=generation_config):
    """Initializes and returns the configured Gemini model for text generation."""
    try:
        # Using gemini-1.5-pro as specified. Ensure this model name is correct and available.
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro", # Or another suitable model like gemini-1.5-flash
            generation_config=config # Use the provided config
        )
        return model
    except Exception as e:
        # More specific error handling for API key issues or model availability
        if "API key not valid" in str(e):
             # This specific error should ideally be caught during genai.configure in app.py
             raise ValueError("The provided Gemini API Key is invalid. Please check the key.") from e
        if "Could not find model" in str(e):
             raise ValueError(f"The model specified is not available.") from e
        raise Exception(f"Error initializing Gemini model: {str(e)}") # Catch other potential errors

def get_answer_with_citations(question, document_store, max_context_chunks=10):
    """
    Get an answer to a question with citations from the document store,
    formatted for better readability and with more relevant citation text.

    Args:
        question: User's question
        document_store: Document store instance (contains the FAISS index and chunk data)
        max_context_chunks: Maximum number of top relevant chunks to retrieve from the store.

    Returns:
        Tuple of (answer_text, citations)
    """
    # 1. Retrieve relevant chunks
    search_results = document_store.similarity_search(question, k=max_context_chunks)

    if not search_results:
        no_info_message = "I couldn't find any relevant information in the uploaded documents to answer that question."
        db.log_query(question, no_info_message) # Log the query even if no results
        return no_info_message, []

    # 2. Format context for the LLM prompt
    context = ""
    filenames_in_context = set()
    for i, (chunk_text, metadata, _) in enumerate(search_results):
        source_key = f"Source {i+1}"
        filename = metadata.get('filename', 'Unknown File') # Use .get for safety
        page_num = metadata.get('page_number', 'Unknown Page') # Use .get for safety
        filenames_in_context.add(filename)

        context += f"\n--- {source_key} ---\n"
        context += f"Filename: {filename}\n"
        context += f"Page: {page_num}\n"
        context += f"Content: {chunk_text}\n"
        context += "-----------------\n"

    # 3. Construct the prompt for the LLM
    primary_doc_mention = ""
    if len(filenames_in_context) == 1:
        primary_doc_mention = f"The document '{list(filenames_in_context)[0]}' "

    prompt = f"""
    You are an expert AI assistant specialized in analyzing renewable energy project documents.
    Your task is to answer the user's question based *only* on the provided context below.

    **Instructions for Answering:**
    1.  **Analyze Context:** Carefully read the context sections (Source 1, Source 2, etc.).
    2.  **Synthesize Answer:** Formulate a clear, coherent answer. Start with a brief introductory sentence if appropriate. {primary_doc_mention}
    3.  **Use Bullet Points:** If the answer involves listing multiple distinct points, findings, or aspects covered in the documents (e.g., types of clauses, requirements, steps), **use bullet points** for clarity. Each bullet point should summarize a specific finding.
    4.  **Cite Strategically:** Place inline citations `[Filename, Page X]` at the **end** of the sentence or bullet point that summarizes the information from that source. Avoid excessive repetition of the same citation within a single sentence or bullet point. If multiple sources support one point, list them like `[FileA, Page 1], [FileB, Page 5]`.
    5.  **Filename Mention:** If the answer primarily draws from a single document (as identified above), you've already mentioned it. Avoid repeating the full filename within the bullet points or sentences unless necessary for clarity (e.g., when contrasting information from different documents).
    6.  **Focus on Context:** Base the answer *strictly* on the provided context. If the information isn't present, state: "Based on the provided documents, I could not find specific information regarding [topic of the question]."

    **Instructions for JSON Output:**
    * After crafting the answer text, structure your *entire* response as a single JSON object.
    * This JSON object must have exactly two keys:
        * `"answer"`: A string containing your synthesized answer, potentially using bullet points and with inline citations formatted as described above.
        * `"citations"`: A JSON array. Each object in this array represents a *unique* source document and page number cited in your answer. Each object must have:
            * `"filename"`: The document filename (string).
            * `"page_number"`: The page number (integer).
            * `"text"`: **Crucially, this must be a *direct quote* (a specific sentence or key phrase) from the source document and page number listed, which *explicitly contains the information* you used to make the corresponding point in your answer text.** Ensure this quote is the *most relevant evidence* for the specific fact or statement cited in your answer. Do *not* just return the beginning of a text chunk or unrelated text.

    **Context:**
    {context}

    **User Question:** {question}

    **JSON Output:**
    """

    # 4. Call the Gemini API for generation
    try:
        model = get_gemini_model(config=generation_config) # Use standard config
        response = model.generate_content(prompt)
        response_text = response.text

        # 5. Attempt to parse the JSON response and extract citations
        try:
            cleaned_response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
            result = json.loads(cleaned_response_text)
            answer = result.get("answer", "Error: Could not parse answer from response.")
            citations = result.get("citations", [])

            # Validate and process citations
            valid_citations = []
            if isinstance(citations, list):
                processed_citation_keys = set()
                for cit in citations:
                    if isinstance(cit, dict) and \
                       'filename' in cit and \
                       'page_number' in cit and \
                       'text' in cit and \
                       isinstance(cit['text'], str) and \
                       cit['text'].strip():
                        try:
                            page_num = int(cit['page_number'])
                            filename = cit['filename']
                            text = cit['text'].strip()
                            key = (filename, page_num)

                            if key not in processed_citation_keys:
                                valid_citations.append({
                                    'filename': filename,
                                    'page_number': page_num,
                                    'text': text
                                })
                                processed_citation_keys.add(key)

                        except (ValueError, TypeError):
                             print(f"Warning: Invalid page number format in citation: {cit}")
                    else:
                        print(f"Warning: Skipping invalid or incomplete citation format: {cit}")
            else:
                 print(f"Warning: Citations field was not a list: {citations}")
                 valid_citations = []

            # 6. Log the query and return results
            db.log_query(question, answer, valid_citations)
            return answer, valid_citations

        except json.JSONDecodeError as json_err:
            print(f"Error parsing Gemini JSON response: {json_err}")
            print(f"Raw response was:\n{response_text}")
            answer = response_text
            citations = extract_citations_from_text(response_text)
            db.log_query(question, answer, citations)
            return answer, citations
        except Exception as parsing_error:
             print(f"Unexpected error parsing Gemini response: {parsing_error}")
             print(f"Raw response was:\n{response_text}")
             answer = response_text
             citations = extract_citations_from_text(response_text)
             db.log_query(question, answer, citations)
             return answer, citations

    except Exception as e:
        error_message = f"Error generating answer with Gemini: {str(e)}"
        db.log_query(question, error_message, [])
        raise Exception(error_message)


def extract_citations_from_text(answer_text):
    """
    Fallback: Extract citation references [Filename, Page X] from text
    if JSON parsing fails.
    """
    citation_pattern = r'\[([\w\s\-.\(\)]+),\s*Page\s*(\d+)\]'
    found_citations = re.findall(citation_pattern, answer_text)
    structured_citations = []
    processed_keys = set()
    for filename, page_number_str in found_citations:
        filename = filename.strip()
        try:
            page_number = int(page_number_str)
            key = (filename, page_number)
            if key not in processed_keys:
                structured_citations.append({
                    'filename': filename,
                    'page_number': page_number,
                    'text': 'N/A - Extracted from inline text'
                })
                processed_keys.add(key)
        except ValueError:
            print(f"Warning: Could not parse page number in fallback extraction: {page_number_str}")
    return structured_citations


# --- NEW FUNCTION: Generate Example Questions ---
def generate_example_questions(filenames: list[str], num_questions=5) -> list[str]:
    """
    Generates relevant example questions based on a list of document filenames.

    Args:
        filenames (list[str]): A list of filenames currently processed.
        num_questions (int): The desired number of example questions.

    Returns:
        list[str]: A list of generated example question strings.
                   Returns default static questions if generation fails or no filenames provided.
    """
    default_questions = [
        "What is the total installed capacity mentioned?",
        "Summarize the environmental impact assessment.",
        "Describe the financing structure or PPA details.",
        "What are the land lease terms mentioned?",
        "List the main permitting requirements outlined."
    ]

    if not filenames:
        print("No filenames provided, returning default example questions.")
        return default_questions

    # Format filenames for the prompt
    filenames_str = "\n".join([f"- {f}" for f in filenames])

    # Construct the prompt for example question generation
    prompt = f"""
    You are an AI assistant helping users explore renewable energy project documents.
    Based on the following list of filenames that have been uploaded, generate {num_questions} diverse and relevant example questions a user might ask.

    **Instructions:**
    1.  Generate exactly {num_questions} questions.
    2.  Make the questions specific to potential content in renewable energy documents (e.g., capacity, permits, PPA, land lease, environmental impact, technology used, timelines).
    3.  Where appropriate, incorporate one of the specific filenames provided into the question using brackets, like `[filename.pdf]`. Vary the filenames used.
    4.  Ensure the questions are distinct from each other.
    5.  Output *only* the questions, each on a new line. Do not include numbering, bullet points, or any introductory/concluding text.

    **Uploaded Filenames:**
    {filenames_str}

    **Example Questions (Output Format):**
    What is the project timeline mentioned in [Project_Schedule.pdf]?
    Summarize the key risks identified in the due diligence report.
    What technology is specified in [Technical_Specs.docx]?
    Describe the PPA tariff structure.
    List the permits required according to [Permitting_Overview.pdf].
    """

    try:
        # Use a separate config for potentially more creative questions
        model = get_gemini_model(config=example_generation_config)
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Split the response into lines and filter out empty lines
        generated_questions = [q.strip() for q in response_text.split('\n') if q.strip()]

        # Basic validation: Check if we got roughly the expected number of questions
        if 1 <= len(generated_questions) <= num_questions + 2: # Allow a bit of flexibility
            print(f"Successfully generated {len(generated_questions)} example questions.")
            return generated_questions[:num_questions] # Return up to the requested number
        else:
            print(f"Warning: Unexpected number of questions generated ({len(generated_questions)}). Raw response:\n{response_text}")
            return default_questions # Fallback to default

    except Exception as e:
        print(f"Error generating example questions with Gemini: {str(e)}")
        return default_questions # Fallback to default questions on error


# (save_feedback function remains the same)
def save_feedback(query_id, rating, comment=None):
    """Store user feedback on answers."""
    try:
        success = db.store_feedback(query_id, rating, comment)
        return success
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False
