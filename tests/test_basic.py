import unittest
import os
import sys

# Add the parent directory to path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestBasicFunctionality(unittest.TestCase):
    def test_environment(self):
        """Verify that the testing environment is set up correctly"""
        self.assertTrue(True)

    def test_imports(self):
        """Test that core modules can be imported"""
        try:
            import document_processor
            import document_store
            import qa_engine
            import gemini_api
            import utils
            import database
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_document_processor_functions(self):
        """Test that document_processor has required functions"""
        import document_processor
        self.assertTrue(hasattr(document_processor, 'is_supported_file'))
        self.assertTrue(hasattr(document_processor, 'extract_text_from_file'))
        self.assertTrue(hasattr(document_processor, 'clean_text'))

    def test_file_extension_detection(self):
        """Test file extension detection functionality"""
        import document_processor
        self.assertTrue(document_processor.is_supported_file('pdf'))
        self.assertTrue(document_processor.is_supported_file('docx'))
        self.assertTrue(document_processor.is_supported_file('txt'))
        self.assertFalse(document_processor.is_supported_file('jpg'))
        self.assertFalse(document_processor.is_supported_file('xlsx'))

    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        import document_processor
        dirty_text = "This has\n\nextra   spaces \t and\ntabs"
        clean_text = document_processor.clean_text(dirty_text)
        self.assertEqual(clean_text, "This has extra spaces and tabs")
        
    def test_text_splitting(self):
        """Test text splitting functionality"""
        import document_processor
        text = "Page 1\nThis is some text for testing.\n\nPage 2\nThis is more text for testing."
        chunks = document_processor.split_text_into_chunks(text, chunk_size=20, chunk_overlap=5)
        self.assertGreater(len(chunks), 1)
        # Check that page numbers are preserved
        self.assertTrue(any("Page 1" in chunk["text"] for chunk in chunks))

    def test_document_store_initialization(self):
        """Test document store initialization"""
        import document_store
        store = document_store.DocumentStore()
        self.assertIsNotNone(store.faiss_index)
        
    def test_add_document_to_store(self):
        """Test adding a document to the store"""
        import document_store
        import uuid
        store = document_store.DocumentStore()
        doc_id = str(uuid.uuid4())
        chunks = [{"text": "Test chunk", "page": 1, "chunk_id": 0}]
        store.add_document(doc_id, "test.pdf", chunks)
        self.assertIn(doc_id, store.documents)

    def test_example_question_generation(self):
        """Test example question generation"""
        from unittest.mock import patch
        import qa_engine
        
        # Mock the Gemini API response
        with patch('qa_engine.generate_example_questions_with_gemini') as mock_generate:
            mock_generate.return_value = ["What is renewable energy?", "How do solar panels work?"]
            chunks = [{"text": "Renewable energy comes from sources that naturally replenish.", "page": 1}]
            questions = qa_engine.generate_example_questions(chunks)
            self.assertEqual(len(questions), 2)
            self.assertIn("What is renewable energy?", questions)

    def test_gemini_api_configuration(self):
        """Test Gemini API configuration"""
        import os
        import gemini_api
        
        # Save original environment
        original_key = os.environ.get("GEMINI_API_KEY", "")
        
        try:
            # Test with mock key
            os.environ["GEMINI_API_KEY"] = "test_key"
            model = gemini_api.get_gemini_model()
            self.assertIsNotNone(model)
        finally:
            # Restore original environment
            if original_key:
                os.environ["GEMINI_API_KEY"] = original_key
            else:
                del os.environ["GEMINI_API_KEY"]

    def test_file_extension_extraction(self):
        """Test file extension extraction"""
        import utils
        self.assertEqual(utils.get_file_extension("document.pdf"), "pdf")
        self.assertEqual(utils.get_file_extension("file.name.with.dots.txt"), "txt")
        self.assertEqual(utils.get_file_extension("no_extension"), "")
        
    def test_text_truncation(self):
        """Test text truncation functionality"""
        import utils
        long_text = "This is a very long text that should be truncated properly when it exceeds the maximum length"
        truncated = utils.truncate_text(long_text, max_length=20)
        self.assertTrue(len(truncated) <= 23)  # 20 + "..."
        self.assertTrue(truncated.endswith("..."))

    def test_database_fallback(self):
        """Test database fallback mechanism"""
        import database
        
        # Create a new database instance with invalid URI to force fallback
        db = database.Database(uri="invalid_uri")
        self.assertTrue(hasattr(db, "documents"))
        self.assertTrue(hasattr(db, "chunks"))
        
        # Test basic storage operations
        doc_id = "test_doc"
        db.store_document(doc_id, "Test document content")
        chunks = [{"text": "Test chunk", "page": 1}]
        db.store_chunks(doc_id, chunks)
        
        # Retrieve and verify
        retrieved_chunks = db.get_document_chunks(doc_id)
        self.assertEqual(len(retrieved_chunks), 1)

if __name__ == '__main__':
    unittest.main()