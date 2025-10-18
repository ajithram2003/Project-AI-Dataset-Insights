"""
Integration tests for the AI Dataset Insights application.
"""
import unittest
import os
import tempfile
import pandas as pd
from io import BytesIO

# Add src to path to import our modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import create_app


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete application flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
    def test_file_upload_and_analysis(self):
        """Test complete file upload and analysis workflow."""
        # Create a test CSV file
        test_data = pd.DataFrame({
            'sales': [100, 200, 150, 300, 250],
            'profit': [20, 40, 30, 60, 50],
            'region': ['North', 'South', 'East', 'West', 'North']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_path = f.name
            
        try:
            # Test file upload
            with open(temp_path, 'rb') as f:
                response = self.client.post('/analyze', 
                                          data={'dataset': (f, 'test_data.csv')},
                                          content_type='multipart/form-data')
                
            # Should return result page (200) with analysis content
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Analysis Results', response.data)
            self.assertIn(b'Statistical Summary', response.data)
            
        finally:
            os.unlink(temp_path)
            
    def test_invalid_file_upload(self):
        """Test upload of invalid file type."""
        # Create a test text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('This is not a CSV file')
            temp_path = f.name
            
        try:
            with open(temp_path, 'rb') as f:
                response = self.client.post('/analyze',
                                          data={'dataset': (f, 'test_data.txt')},
                                          content_type='multipart/form-data')
                
            # Should redirect back to index with error
            self.assertEqual(response.status_code, 302)
            
        finally:
            os.unlink(temp_path)
            
    def test_empty_file_upload(self):
        """Test upload of empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')  # Empty file
            temp_path = f.name
            
        try:
            with open(temp_path, 'rb') as f:
                response = self.client.post('/analyze',
                                          data={'dataset': (f, 'empty.csv')},
                                          content_type='multipart/form-data')
                
            # Should redirect back to index with error
            self.assertEqual(response.status_code, 302)
            
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
