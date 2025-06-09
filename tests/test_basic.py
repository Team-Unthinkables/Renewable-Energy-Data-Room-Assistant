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
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_document_processor_functions(self):
        """Test that document_processor has required functions"""
        import document_processor
        self.assertTrue(hasattr(document_processor, 'is_supported_file'))
        self.assertTrue(hasattr(document_processor, 'extract_text_from_file'))
        self.assertTrue(hasattr(document_processor, 'clean_text'))

if __name__ == '__main__':
    unittest.main()