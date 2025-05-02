import uuid
import os
import numpy as np
import faiss
from document_processor import split_text_into_chunks
import google.generativeai as genai

# Set up the Gemini API with the provided key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCL7WcFojRcescuZfdGw4iK_syGD3YfG5E")
genai.configure(api_key=GEMINI_API_KEY)

class DocumentStore:
    """Handles document storage, retrieval, and vector search."""
    
    def __init__(self):
        """Initialize the document store with a FAISS vector database."""
        self.documents = {}  # Store document metadata
        self.chunks = {}  # Store text chunks with metadata
        self.vector_store = None  # FAISS index
        self.chunk_ids_list = []  # List to keep track of chunk IDs in same order as vectors
        self.embedding_dimension = 768  # Dimension of the embedding vectors for text embedding
        
        # Initialize FAISS index
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize an empty FAISS index."""
        self.vector_store = faiss.IndexFlatL2(self.embedding_dimension)
        self.chunk_ids_list = []
    
    def _get_embeddings(self, texts):
        """Get embeddings for a list of texts using Gemini's embedding model."""
        try:
            # Create embeddings for each text
            embeddings = []
            
            # Process each text to get embeddings
            for text in texts:
                # Trim text if too long (Gemini has input limits)
                trimmed_text = text[:10000] if len(text) > 10000 else text
                
                # Get embedding from Gemini
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=trimmed_text,
                    task_type="retrieval_document"
                )
                
                # Extract the embedding vector
                embedding_vector = result['embedding']
                
                # Make sure it's the expected dimension
                if len(embedding_vector) != self.embedding_dimension:
                    # This shouldn't happen, but just in case
                    print(f"Warning: Unexpected embedding dimension: {len(embedding_vector)}")
                    # Pad or truncate to expected dimension
                    if len(embedding_vector) > self.embedding_dimension:
                        embedding_vector = embedding_vector[:self.embedding_dimension]
                    else:
                        embedding_vector = embedding_vector + [0] * (self.embedding_dimension - len(embedding_vector))
                
                embeddings.append(embedding_vector)
                
            # Convert to numpy array for FAISS
            return np.array(embeddings).astype('float32')
            
        except Exception as e:
            # If embeddings fail, use simple fallback
            print(f"Error getting embeddings: {str(e)}")
            return self._get_simple_embeddings(texts)
    
    def _get_simple_embeddings(self, texts):
        """Create simple fallback embeddings when API is unavailable."""
        # Create random embeddings of the right dimension
        # This is just a fallback when the API isn't working
        embeddings = np.random.rand(len(texts), self.embedding_dimension).astype('float32')
        return embeddings
        
    def add_document(self, filename, content):
        """
        Add a document to the store and update the vector database.
        
        Args:
            filename: Name of the document file
            content: List of dicts with page_number and text
            
        Returns:
            document_id: Unique ID for the document
        """
        # Generate a unique ID for the document
        document_id = str(uuid.uuid4())
        
        # Store document metadata
        self.documents[document_id] = {
            'filename': filename,
            'page_count': len(content)
        }
        
        # Split document into chunks
        document_chunks = split_text_into_chunks(content)
        
        # Store chunks with document metadata
        for chunk in document_chunks:
            chunk_id = str(uuid.uuid4())
            self.chunks[chunk_id] = {
                'document_id': document_id,
                'filename': filename,
                'page_number': chunk['page_number'],
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text']
            }
        
        # Update the vector store
        self._rebuild_vector_store()
        
        return document_id
    
    def delete_document(self, document_id):
        """
        Remove a document and its chunks from the store.
        
        Args:
            document_id: ID of the document to remove
        """
        if document_id in self.documents:
            # Remove document metadata
            del self.documents[document_id]
            
            # Remove associated chunks
            chunks_to_remove = []
            for chunk_id, chunk in self.chunks.items():
                if chunk['document_id'] == document_id:
                    chunks_to_remove.append(chunk_id)
            
            for chunk_id in chunks_to_remove:
                del self.chunks[chunk_id]
            
            # Rebuild the vector store if there are remaining documents
            if self.chunks:
                self._rebuild_vector_store()
            else:
                self.vector_store = None
    
    def clear_all(self):
        """Remove all documents and chunks from the store."""
        self.documents = {}
        self.chunks = {}
        self.vector_store = None
    
    def _rebuild_vector_store(self):
        """Rebuild the FAISS vector store with the current chunks."""
        if not self.chunks:
            self._init_vector_store()
            return
        
        # Prepare texts for embeddings
        texts = []
        self.chunk_ids_list = []
        
        for chunk_id, chunk in self.chunks.items():
            texts.append(chunk['text'])
            self.chunk_ids_list.append(chunk_id)
        
        # Get embeddings for all texts
        try:
            # Initialize a new FAISS index
            self.vector_store = faiss.IndexFlatL2(self.embedding_dimension)
            
            # Get embeddings for all texts
            if texts:
                embeddings = self._get_embeddings(texts)
                
                # Add vectors to the index
                if len(embeddings) > 0:
                    self.vector_store.add(embeddings)
                    
        except Exception as e:
            print(f"Error rebuilding vector store: {str(e)}")
            # Initialize empty index as fallback
            self._init_vector_store()
    
    def similarity_search(self, query, k=5):
        """
        Search for chunks similar to the query.
        
        Args:
            query: The search query text
            k: Number of results to return
            
        Returns:
            List of tuples with (chunk_text, chunk_metadata, score)
        """
        if not self.vector_store or not self.chunks or not self.chunk_ids_list:
            return []
        
        try:
            # Adjust k if we have fewer chunks than requested
            k = min(k, len(self.chunk_ids_list))
            if k == 0:
                return []
            
            # Get query embedding
            query_embedding = self._get_embeddings([query])
            
            # Search in FAISS
            distances, indices = self.vector_store.search(query_embedding, k)
            
            # Format results
            search_results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                distance = distances[0][i]
                
                # Skip invalid indices (shouldn't happen, but just in case)
                if idx >= len(self.chunk_ids_list) or idx < 0:
                    continue
                    
                chunk_id = self.chunk_ids_list[idx]
                
                # Skip if chunk was deleted (shouldn't happen if _rebuild_vector_store is called properly)
                if chunk_id not in self.chunks:
                    continue
                    
                chunk = self.chunks[chunk_id]
                
                # Convert distance to score (lower distance is better)
                # Normalize score between 0 and 1 (1 is best)
                score = 1.0 / (1.0 + distance)
                
                search_results.append((
                    chunk['text'],
                    {
                        'document_id': chunk['document_id'],
                        'filename': chunk['filename'],
                        'page_number': chunk['page_number']
                    },
                    float(score)
                ))
            
            return search_results
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    
    def get_document_text(self, document_id):
        """
        Get the full text of a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of page texts in order
        """
        if document_id not in self.documents:
            return None
        
        # Collect all chunks from this document
        document_chunks = []
        for chunk_id, chunk in self.chunks.items():
            if chunk['document_id'] == document_id:
                document_chunks.append(chunk)
        
        # Sort by page number and chunk index
        document_chunks.sort(key=lambda x: (x['page_number'], x['chunk_index']))
        
        # Group by page
        pages = {}
        for chunk in document_chunks:
            page_num = chunk['page_number']
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(chunk['text'])
        
        # Combine chunks for each page
        page_texts = []
        for page_num in sorted(pages.keys()):
            page_texts.append(''.join(pages[page_num]))
        
        return page_texts
