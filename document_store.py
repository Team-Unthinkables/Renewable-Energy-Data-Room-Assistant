# document_store.py
# This module handles document processing, embedding, indexing using FAISS,
# and managing the document data.

import uuid
import os
import numpy as np
import faiss
# Ensure split_text_into_chunks is imported correctly
from document_processor import split_text_into_chunks
# Import genai here as it's used for embeddings
import google.generativeai as genai
# Assuming db instance is correctly initialized in database.py and imported
from database import db
import time # For potential timing/debugging

# --- Gemini API Configuration ---
# API key is configured globally in app.py after loading from environment variables.
# No need to configure genai here again.

# --- Constants ---
EMBEDDING_MODEL = "models/embedding-001" # The specific embedding model to use.
                                        # This model provides 768-dimensional embeddings.
EMBEDDING_DIMENSION = 768               # Dimension of the embedding vectors.
# Gemini API limits might apply to batch size, adjust if needed.
# Processing in batches is more efficient than one text at a time.
MAX_EMBEDDING_BATCH_SIZE = 100 # Example limit, check Gemini docs for current limits.
                               # Adjust based on API quotas and network conditions.

class DocumentStore:
    """Handles document storage, embedding, indexing, and retrieval."""

    def __init__(self):
        """Initialize the document store."""
        # In-memory storage for quick access during the session.
        # Consider persistence (e.g., saving FAISS index and chunk data to disk/DB)
        # if you need data to persist beyond the application session.
        self.documents = {}  # Stores document metadata: {doc_id: {'filename': ..., 'page_count': ...}}
        self.chunks = {}     # Stores chunk details: {chunk_id: {'doc_id': ..., 'text': ..., 'metadata': ...}}
        self.chunk_id_to_index = {} # Maps chunk_id to its index in self.vector_store and self.ordered_chunk_ids
        self.ordered_chunk_ids = [] # Maintains order consistent with FAISS index

        # FAISS Index for efficient similarity search.
        self.vector_store = None
        self._init_vector_store() # Initialize an empty index

        print("DocumentStore initialized.")
        # Optionally load existing data if persisted (requires implementation)

    def _init_vector_store(self):
        """Initializes or resets the FAISS index."""
        print(f"Initializing FAISS IndexIDMap with dimension {EMBEDDING_DIMENSION}")
        # Using IndexFlatL2 for simplicity (L2 distance). IndexIDMap allows mapping
        # FAISS internal IDs to our custom chunk_ids, which is useful for deletion.
        self.vector_store = faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIMENSION))
        self.ordered_chunk_ids = [] # Reset ordered list
        self.chunk_id_to_index = {} # Reset mapping
        print("FAISS index initialized.")

    def _get_embeddings_batch(self, texts):
        """
        Get embeddings for a list of texts using Gemini's batch embedding capability.

        Args:
            texts (list[str]): A list of text strings to embed.

        Returns:
            list[list[float]]: A list of embedding vectors, or None if an error occurs.
        """
        if not texts:
            return []

        all_embeddings = []
        total_texts = len(texts)
        print(f"Requesting embeddings for {total_texts} text chunks in batches of {MAX_EMBEDDING_BATCH_SIZE}...")

        try:
            for i in range(0, total_texts, MAX_EMBEDDING_BATCH_SIZE):
                batch_texts = texts[i : i + MAX_EMBEDDING_BATCH_SIZE]
                print(f"  Processing batch {i // MAX_EMBEDDING_BATCH_SIZE + 1}/{(total_texts + MAX_EMBEDDING_BATCH_SIZE - 1) // MAX_EMBEDDING_BATCH_SIZE} ({len(batch_texts)} texts)")

                # Make the batch embedding request
                # Task type 'RETRIEVAL_DOCUMENT' is typically used for indexing content.
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch_texts, # Pass the list of texts directly
                    task_type="retrieval_document"
                )

                batch_embeddings = result.get('embedding')

                if not batch_embeddings or len(batch_embeddings) != len(batch_texts):
                    print(f"Error: Embedding count mismatch in batch {i}. Expected {len(batch_texts)}, got {len(batch_embeddings) if batch_embeddings else 0}.")
                    # Handle error: maybe retry, skip batch, or raise exception
                    # For simplicity, we'll skip this batch and continue, logging the error.
                    # In production, more robust error handling is needed.
                    continue # Skip adding potentially corrupted batch

                # Validate embedding dimension (optional but recommended)
                for emb in batch_embeddings:
                    if len(emb) != EMBEDDING_DIMENSION:
                        print(f"Warning: Unexpected embedding dimension {len(emb)} found. Expected {EMBEDDING_DIMENSION}. Skipping embedding.")
                        # Handle mismatch: skip this embedding, pad/truncate, or raise error
                        # Skipping for now
                        continue # Or handle appropriately

                all_embeddings.extend(batch_embeddings)
                print(f"  Batch {i // MAX_EMBEDDING_BATCH_SIZE + 1} processed.")
                # Optional: Add a small delay between batches if hitting rate limits
                # time.sleep(0.1)

            print(f"Successfully retrieved embeddings for {len(all_embeddings)} out of {total_texts} texts.")
            return all_embeddings

        except Exception as e:
            print(f"Error getting batch embeddings: {str(e)}")
            # Consider more specific error handling based on potential API errors
            return None # Indicate failure

    def add_document(self, filename, content):
        """
        Adds a document, processes it into chunks, generates embeddings,
        and updates the vector store incrementally.

        Args:
            filename (str): The name of the document file.
            content (list[dict]): List of dicts {'page_number': int, 'text': str} from document processor.

        Returns:
            str: The unique ID assigned to the document, or None if processing fails.
        """
        print(f"Adding document: {filename}")
        if not content:
            print(f"Warning: Document '{filename}' has no content to process. Skipping.")
            return None

        doc_id = str(uuid.uuid4())
        self.documents[doc_id] = {
            'filename': filename,
            'page_count': max(item['page_number'] for item in content) if content else 0,
            'added_time': time.time()
        }

        # --- 1. Split into Chunks ---
        try:
            # Assuming split_text_into_chunks is efficient enough and handles metadata
            # This function should return chunks with 'text', 'page_number', and potentially other metadata like 'chunk_index_in_page'.
            document_chunks_data = split_text_into_chunks(content) # Pass the raw content list
            if not document_chunks_data:
                print(f"Warning: No chunks generated for document '{filename}'. Skipping.")
                # Clean up document metadata if no chunks are added
                del self.documents[doc_id]
                return None
            print(f"  Split into {len(document_chunks_data)} chunks.")
        except Exception as e:
            print(f"Error splitting document '{filename}' into chunks: {e}")
            if doc_id in self.documents: del self.documents[doc_id] # Clean up
            return None

        # Prepare chunk texts and generate unique IDs for FAISS
        new_chunk_texts = [chunk['text'] for chunk in document_chunks_data]
        # Generate UUIDs for internal tracking and for mapping to FAISS IDs
        new_chunk_ids = [str(uuid.uuid4()) for _ in document_chunks_data]
        # Convert UUIDs to int64 for use as IDs in FAISS IndexIDMap
        # Using modulo 2^63 to ensure positive IDs within int64 range
        faiss_ids = np.array([int(chunk_id.replace('-', ''), 16) % (2**63) for chunk_id in new_chunk_ids]).astype('int64')


        # --- 2. Get Embeddings (Batch) ---
        start_time = time.time()
        embeddings = self._get_embeddings_batch(new_chunk_texts)
        end_time = time.time()
        print(f"  Generated {len(embeddings) if embeddings else 0} embeddings in {end_time - start_time:.2f} seconds.")

        if embeddings is None or len(embeddings) != len(new_chunk_texts):
            print(f"Error: Failed to generate embeddings or count mismatch for '{filename}'. Aborting add.")
            if doc_id in self.documents: del self.documents[doc_id] # Clean up
            return None

        # Convert embeddings to numpy array for FAISS
        embeddings_np = np.array(embeddings).astype('float32')


        # --- 3. Add to Vector Store (Incremental) ---
        if self.vector_store is None:
            print("Error: Vector store not initialized.")
            if doc_id in self.documents: del self.documents[doc_id] # Clean up
            return None

        try:
            print(f"  Adding {len(embeddings_np)} vectors to FAISS index...")
            # Use add_with_ids for IndexIDMap to associate embeddings with our generated IDs
            self.vector_store.add_with_ids(embeddings_np, faiss_ids)
            print(f"  FAISS index updated. Total vectors: {self.vector_store.ntotal}")
        except Exception as e:
            print(f"Error adding vectors to FAISS for '{filename}': {e}")
            # Potential rollback or cleanup needed here if FAISS add fails
            if doc_id in self.documents: del self.documents[doc_id] # Clean up
            return None

        # --- 4. Update Local Cache and Database ---
        # Only update caches and DB if embeddings and FAISS add were successful
        for i, chunk_data in enumerate(document_chunks_data):
            chunk_id = new_chunk_ids[i]
            faiss_id = faiss_ids[i] # The ID used in FAISS

            # Store chunk details locally, including original metadata from document_processor
            self.chunks[chunk_id] = {
                'doc_id': doc_id,
                'text': chunk_data['text'],
                'metadata': { # Store all relevant metadata
                    'filename': filename,
                    'page_number': chunk_data.get('page_number', 0), # Use .get with default for safety
                    'chunk_index_in_page': chunk_data.get('chunk_index_in_page', 0), # Use .get with default
                    'doc_chunk_id': chunk_data.get('doc_chunk_id', 0) # Use .get with default
                }
            }
            # Store mapping from our chunk_id to the FAISS ID (useful for deletion)
            self.chunk_id_to_index[chunk_id] = faiss_id # Store the FAISS ID

            # The ordered_chunk_ids list is less critical with IndexIDMap for retrieval,
            # but could be maintained if needed for specific operations.

        # Persist document metadata to Database (Optional but recommended for persistence)
        # Consider doing this asynchronously or batching DB writes if it becomes a bottleneck
        try:
            # Store original content/metadata in DB if needed for full document retrieval or other purposes
            # The current store_document in database.py primarily stores metadata.
            db.store_document(filename, content, doc_id) # Store original content/metadata
            # Storing individual chunks in the DB (db.store_chunks) is currently not implemented
            # in the provided database.py in a way that's used by DocumentStore for retrieval.
            # If persistence of chunks themselves is needed, this would require implementation
            # and potentially loading them back into self.chunks on startup.
            print(f"  Document metadata stored in DB for doc_id: {doc_id}")
        except Exception as e:
            print(f"Warning: Failed to store document/chunks in DB for '{filename}': {e}")
            # Decide on error handling: proceed without DB persistence or fail?

        print(f"Successfully added document '{filename}' (ID: {doc_id}).")
        return doc_id

    def delete_document(self, doc_id):
        """
        Removes a document and its associated chunks and vectors from the store and index.

        Args:
            doc_id (str): The ID of the document to remove.
        """
        if doc_id not in self.documents:
            print(f"Warning: Document ID {doc_id} not found for deletion.")
            return False

        filename = self.documents[doc_id]['filename']
        print(f"Deleting document: {filename} (ID: {doc_id})")

        # --- 1. Identify Chunks and their FAISS IDs belonging to this document ---
        chunk_ids_to_remove = [cid for cid, chunk in self.chunks.items() if chunk['doc_id'] == doc_id]
        # Get the corresponding FAISS IDs using the stored mapping
        faiss_ids_to_remove = np.array([self.chunk_id_to_index[cid] for cid in chunk_ids_to_remove if cid in self.chunk_id_to_index]).astype('int64')


        if not chunk_ids_to_remove:
            print("  No chunks found for this document ID in local cache.")
            # Still attempt to remove document metadata and from DB
            if doc_id in self.documents: del self.documents[doc_id]
            try:
                db.delete_document(doc_id) # Remove from DB
            except Exception as e:
                print(f"Warning: Failed to delete document {doc_id} from DB: {e}")
            return True

        # --- 2. Remove from FAISS Index ---
        # Only attempt to remove from FAISS if there are vectors to remove
        if self.vector_store and self.vector_store.ntotal > 0 and len(faiss_ids_to_remove) > 0:
            try:
                print(f"  Removing {len(faiss_ids_to_remove)} vectors from FAISS index...")
                # Use remove_ids with the array of IDs to remove
                remove_count = self.vector_store.remove_ids(faiss_ids_to_remove)
                print(f"  Successfully removed {remove_count} vectors. Index size: {self.vector_store.ntotal}")
                if remove_count != len(faiss_ids_to_remove):
                    print(f"Warning: FAISS remove count ({remove_count}) differs from expected ({len(faiss_ids_to_remove)}).")
            except Exception as e:
                print(f"Error removing vectors from FAISS for doc {doc_id}: {e}")
                # Log error and continue cleanup of local cache and DB

        # --- 3. Remove from Local Cache and Mappings ---
        print("  Removing chunks from local cache and mappings...")
        for chunk_id in chunk_ids_to_remove:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
            if chunk_id in self.chunk_id_to_index:
                del self.chunk_id_to_index[chunk_id]
            # If ordered_chunk_ids was strictly maintained and used, would need removal here too (inefficient for lists)
            # if chunk_id in self.ordered_chunk_ids: self.ordered_chunk_ids.remove(chunk_id)

        del self.documents[doc_id] # Remove document metadata
        print("  Local cache updated.")

        # --- 4. Remove from Database ---
        try:
            # Assuming db.delete_document also handles associated chunk records in the DB if they exist
            db.delete_document(doc_id)
            print(f"  Document {doc_id} removed from DB.")
        except Exception as e:
            print(f"Warning: Failed to delete document {doc_id} from DB: {e}")

        print(f"Successfully deleted document '{filename}' (ID: {doc_id}).")
        return True


    def clear_all(self):
        """Removes all documents, chunks, and resets the vector store."""
        print("Clearing all documents and resetting store...")
        self.documents = {}
        self.chunks = {}
        self.chunk_id_to_index = {}
        self.ordered_chunk_ids = []
        self._init_vector_store() # Re-initialize empty index, clearing FAISS data

        # Clear Database
        try:
            db.clear_all() # This should drop and re-create necessary collections
            print("Database cleared.")
        except Exception as e:
            print(f"Warning: Failed to clear database: {e}")

        print("Document store cleared.")

    # This method is now less necessary if add/delete are incremental,
    # but can be kept for manual rebuilds or recovery if needed.
    def _rebuild_vector_store(self):
        """Rebuilds the entire FAISS index from the current self.chunks. (Potentially Slow)"""
        print("Rebuilding vector store from all chunks...")
        self._init_vector_store() # Start with a fresh index

        if not self.chunks:
            print("No chunks to rebuild index from.")
            return

        all_chunk_ids = list(self.chunks.keys())
        all_texts = [self.chunks[cid]['text'] for cid in all_chunk_ids]
        # Recreate FAISS IDs from chunk_ids
        all_faiss_ids = np.array([int(cid.replace('-', ''), 16) % (2**63) for cid in all_chunk_ids]).astype('int64')

        print(f"Total chunks to process for rebuild: {len(all_texts)}")
        embeddings = self._get_embeddings_batch(all_texts)

        if embeddings is None or len(embeddings) != len(all_texts):
            print("Error: Failed to generate embeddings during rebuild. Index may be incomplete.")
            # Handle this error - maybe retry, or leave index empty
            return

        embeddings_np = np.array(embeddings).astype('float32')

        try:
            # Add all embeddings with their corresponding FAISS IDs
            self.vector_store.add_with_ids(embeddings_np, all_faiss_ids)
            # Update mappings after rebuild
            # self.ordered_chunk_ids = all_chunk_ids # If maintaining order is critical
            # self.chunk_id_to_index = {cid: all_faiss_ids[i] for i, cid in enumerate(all_chunk_ids)} # Rebuild mapping
            print(f"Vector store rebuilt successfully. Total vectors: {self.vector_store.ntotal}")
        except Exception as e:
            print(f"Error adding vectors during rebuild: {e}")
            # Index might be partially built or empty

    def similarity_search(self, query, k=5):
        """
        Searches for chunks similar to the query using FAISS.

        Args:
            query (str): The search query text.
            k (int): The number of top similar chunks to retrieve.

        Returns:
            list[tuple]: A list of tuples, each containing (chunk_text, chunk_metadata, score).
                         Returns empty list if no results or error.
        """
        if not self.vector_store or self.vector_store.ntotal == 0:
            print("Vector store is empty or not initialized. Cannot perform search.")
            return []

        if not query or not isinstance(query, str):
            print("Warning: Invalid query for similarity search.")
            return []

        # Ensure k is not greater than the number of items in the index
        k = min(k, self.vector_store.ntotal)
        if k <= 0:
            return []

        try:
            # --- 1. Get Query Embedding ---
            # Use 'retrieval_query' task type for search queries
            query_embedding_result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query" # Use query type
            )
            query_embedding = query_embedding_result.get('embedding')

            if not query_embedding or len(query_embedding) != EMBEDDING_DIMENSION:
                print(f"Error: Failed to generate valid query embedding for: '{query[:100]}...'")
                return []

            query_vector = np.array([query_embedding]).astype('float32')

            # --- 2. Search FAISS ---
            # `search` returns distances (lower is better) and indices/IDs (which are our faiss_ids)
            distances, faiss_ids_results = self.vector_store.search(query_vector, k)

            # --- 3. Format Results ---
            search_results = []
            # Check if search returned any results and if the IDs are valid
            if faiss_ids_results is not None and len(faiss_ids_results) > 0 and len(faiss_ids_results[0]) > 0:
                for i in range(len(faiss_ids_results[0])):
                    faiss_id = faiss_ids_results[0][i]
                    distance = distances[0][i]

                    # Find the original chunk_id string that corresponds to this faiss_id
                    # Use the stored mapping for efficient lookup
                    chunk_id = None
                    # This lookup assumes chunk_id_to_index is correctly populated and inverse mapping is needed.
                    # If faiss_ids_results directly gives us the IDs we need, the loop below is better.
                    # Let's use the loop approach as it matches the current code structure.
                    for cid_str, stored_faiss_id in self.chunk_id_to_index.items():
                         if stored_faiss_id == faiss_id:
                            chunk_id = cid_str
                            break

                    # If chunk_id is found and exists in local cache
                    if chunk_id and chunk_id in self.chunks:
                        chunk = self.chunks[chunk_id]
                        # Convert distance to a similarity score (e.g., 1 / (1 + distance))
                        # Ensure distance is non-negative
                        score = 1.0 / (1.0 + max(0.0, distance))

                        search_results.append((
                            chunk['text'],
                            chunk['metadata'], # Pass the whole metadata dict
                            float(score)
                        ))
                    else:
                        # This warning indicates a potential mismatch between FAISS index and local cache
                        print(f"Warning: Could not find chunk data in local cache for retrieved FAISS ID {faiss_id}.")

            return search_results

        except Exception as e:
            print(f"Error during similarity search for query '{query[:100]}...': {str(e)}")
            return []

    # get_document_text retrieves text from the local cache based on doc_id
    def get_document_text(self, doc_id):
        """
        Retrieves the text content associated with a document ID from the local cache,
        ordered by page and chunk index.

        Args:
            doc_id (str): The ID of the document.

        Returns:
            list[str]: A list of strings, where each string is the combined text of a page.
                       Returns None if the document ID is not found.
        """
        if doc_id not in self.documents:
            print(f"Document ID {doc_id} not found in store.")
            return None

        # Collect chunks belonging to the document from the local cache
        doc_chunks = []
        for chunk_id, chunk_data in self.chunks.items():
            if chunk_data['doc_id'] == doc_id:
                # Add chunk_id for potential sorting/debugging if needed
                doc_chunks.append({**chunk_data['metadata'], 'text': chunk_data['text']})

        if not doc_chunks:
            print(f"No chunks found locally for document ID {doc_id}.")
            # If persistence is needed, this part would need to load from DB.
            return []

        # Sort chunks by page number, then by their original index within the page/document
        doc_chunks.sort(key=lambda x: (x.get('page_number', 0), x.get('doc_chunk_id', 0)))

        # Reconstruct pages
        pages = {}
        for chunk in doc_chunks:
            page_num = chunk.get('page_number', 1) # Default to page 1 if missing
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(chunk['text'])

        # Combine chunks for each page and return ordered list of page texts
        page_texts = []
        # Sort page numbers before combining
        for page_num in sorted(pages.keys()):
            page_texts.append(' '.join(pages[page_num])) # Join chunks with space

        return page_texts
