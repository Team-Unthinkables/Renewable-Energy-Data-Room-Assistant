import os
import pymongo
import json
import uuid
from datetime import datetime

# MongoDB connection setup
class Database:
    def __init__(self):
        # Set up MongoDB connection - we'll use in-memory by default for Replit
        # In a production environment, you'd use MongoDB Atlas or another MongoDB service
        self.mongo_uri = os.environ.get("MONGODB_URI", None)
        self.client = None
        self.db = None
        self.using_fallback = False
        
        # Try to connect to MongoDB if URI is provided
        if self.mongo_uri:
            connected = self.connect()
            if not connected:
                self._setup_fallback_storage()
        else:
            # Default to in-memory storage for Replit environment
            print("No MongoDB URI found. Using in-memory storage.")
            self._setup_fallback_storage()
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            # Try to connect to MongoDB
            self.client = pymongo.MongoClient(self.mongo_uri)
            
            # Ping the database to check connection
            self.client.admin.command('ping')
            
            # Create or access the database
            self.db = self.client['renewable_energy_data_room']
            
            # Ensure indexes
            self._create_indexes()
            
            # Set flag to indicate we're not using fallback
            self.using_fallback = False
            
            print("Connected to MongoDB successfully!")
            return True
        except Exception as e:
            print(f"MongoDB connection error: {str(e)}")
            return False
    
    def _setup_fallback_storage(self):
        """Setup in-memory storage if MongoDB connection fails"""
        print("Using in-memory storage as fallback")
        self.fallback = {
            'documents': {},
            'user_queries': [],
            'feedback': []
        }
        self.using_fallback = True
    
    def _create_indexes(self):
        """Create necessary indexes"""
        # Documents collection
        self.db.documents.create_index("document_id", unique=True)
        self.db.documents.create_index("filename")
        
        # Chunks collection
        self.db.chunks.create_index("chunk_id", unique=True)
        self.db.chunks.create_index("document_id")
        
        # Queries collection
        self.db.queries.create_index("timestamp")
    
    def store_document(self, filename, content, document_id=None):
        """Store document metadata in MongoDB"""
        try:
            if self.using_fallback:
                # Fallback: store in memory
                doc_id = document_id or str(uuid.uuid4())
                self.fallback['documents'][doc_id] = {
                    'document_id': doc_id,
                    'filename': filename,
                    'page_count': len(content),
                    'upload_date': datetime.now().isoformat()
                }
                return doc_id
            
            # Generate a unique ID if not provided
            doc_id = document_id or str(uuid.uuid4())
            
            # Create document record
            document = {
                'document_id': doc_id,
                'filename': filename,
                'page_count': len(content),
                'upload_date': datetime.now()
            }
            
            # Insert or update document
            self.db.documents.update_one(
                {'document_id': doc_id}, 
                {'$set': document}, 
                upsert=True
            )
            
            return doc_id
        except Exception as e:
            print(f"Error storing document: {str(e)}")
            return None
    
    def store_chunks(self, document_id, chunks):
        """Store document chunks in MongoDB"""
        try:
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # Fallback: we don't need to store chunks separately in memory
                return True
            
            # Prepare bulk operation
            chunk_records = []
            
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                
                chunk_record = {
                    'chunk_id': chunk_id,
                    'document_id': document_id,
                    'text': chunk['text'],
                    'page_number': chunk['page_number'],
                    'chunk_index': chunk['chunk_index']
                }
                
                chunk_records.append(chunk_record)
            
            # Insert all chunks
            if chunk_records:
                self.db.chunks.insert_many(chunk_records)
            
            return True
        except Exception as e:
            print(f"Error storing chunks: {str(e)}")
            return False
    
    def delete_document(self, document_id):
        """Delete document and its chunks from MongoDB"""
        try:
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # Fallback: delete from memory
                if document_id in self.fallback['documents']:
                    del self.fallback['documents'][document_id]
                return True
            
            # Delete the document
            self.db.documents.delete_one({'document_id': document_id})
            
            # Delete associated chunks
            self.db.chunks.delete_many({'document_id': document_id})
            
            return True
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def get_document_chunks(self, document_id):
        """Get all chunks for a document"""
        try:
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # In fallback mode, we don't store chunks separately
                return []
            
            chunks = list(self.db.chunks.find(
                {'document_id': document_id},
                {'_id': 0}  # Exclude MongoDB ID
            ))
            
            return chunks
        except Exception as e:
            print(f"Error getting document chunks: {str(e)}")
            return []
    
    def log_query(self, user_question, answer, citations=None):
        """Log user queries and answers"""
        try:
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # Fallback: store in memory
                self.fallback['user_queries'].append({
                    'query_id': str(uuid.uuid4()),
                    'question': user_question,
                    'answer': answer,
                    'citations': citations or [],
                    'timestamp': datetime.now().isoformat()
                })
                return True
            
            # Create query record
            query = {
                'query_id': str(uuid.uuid4()),
                'question': user_question,
                'answer': answer,
                'citations': citations or [],
                'timestamp': datetime.now()
            }
            
            # Insert query
            self.db.queries.insert_one(query)
            
            return True
        except Exception as e:
            print(f"Error logging query: {str(e)}")
            return False
    
    def store_feedback(self, query_id, rating, comment=None):
        """Store user feedback on answers"""
        try:
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # Fallback: store in memory
                self.fallback['feedback'].append({
                    'feedback_id': str(uuid.uuid4()),
                    'query_id': query_id,
                    'rating': rating,
                    'comment': comment,
                    'timestamp': datetime.now().isoformat()
                })
                return True
            
            # Create feedback record
            feedback = {
                'feedback_id': str(uuid.uuid4()),
                'query_id': query_id,
                'rating': rating,
                'comment': comment,
                'timestamp': datetime.now()
            }
            
            # Insert feedback
            self.db.feedback.insert_one(feedback)
            
            return True
        except Exception as e:
            print(f"Error storing feedback: {str(e)}")
            return False
    
    def get_recent_queries(self, limit=10):
        """Get recent user queries"""
        try:
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # Fallback: get from memory
                return sorted(self.fallback['user_queries'], 
                              key=lambda x: x['timestamp'], 
                              reverse=True)[:limit]
            
            # Get recent queries
            queries = list(self.db.queries.find(
                {},
                {'_id': 0}  # Exclude MongoDB ID
            ).sort('timestamp', pymongo.DESCENDING).limit(limit))
            
            return queries
        except Exception as e:
            print(f"Error getting recent queries: {str(e)}")
            return []
    
    def get_documents(self):
        """Get all documents"""
        try:
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # Fallback: get from memory
                return list(self.fallback['documents'].values())
            
            # Get all documents
            documents = list(self.db.documents.find(
                {},
                {'_id': 0}  # Exclude MongoDB ID
            ))
            
            return documents
        except Exception as e:
            print(f"Error getting documents: {str(e)}")
            return []
    
    def clear_all(self):
        """Clear all data"""
        try:
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # Fallback: clear memory
                self.fallback = {
                    'documents': {},
                    'user_queries': [],
                    'feedback': []
                }
                return True
            
            # Drop collections
            self.db.documents.drop()
            self.db.chunks.drop()
            self.db.queries.drop()
            self.db.feedback.drop()
            
            # Re-create indexes
            self._create_indexes()
            
            return True
        except Exception as e:
            print(f"Error clearing data: {str(e)}")
            return False

# Database instance
db = Database()