"""
Unit tests for the AI Dataset Insights application.
"""
import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path to import our modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import create_app, allowed_file, load_dataframe, compute_basic_stats, fmt_float


class TestApp(unittest.TestCase):
    """Test cases for the Flask application."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
    def test_index_route(self):
        """Test the index route returns 200."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
    def test_allowed_file(self):
        """Test file extension validation."""
        self.assertTrue(allowed_file('test.csv'))
        self.assertTrue(allowed_file('test.xlsx'))
        self.assertTrue(allowed_file('test.xls'))
        self.assertFalse(allowed_file('test.txt'))
        self.assertFalse(allowed_file('test'))
        
    def test_load_dataframe_csv(self):
        """Test loading CSV files."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('col1,col2\n1,2\n3,4\n')
            temp_path = f.name
            
        try:
            df = load_dataframe(temp_path)
            self.assertEqual(len(df), 2)
            self.assertEqual(list(df.columns), ['col1', 'col2'])
        finally:
            os.unlink(temp_path)
            
    def test_compute_basic_stats(self):
        """Test basic statistics computation."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10]
        })
        
        stats = compute_basic_stats(df)
        
        self.assertIn('col1', stats)
        self.assertIn('col2', stats)
        self.assertEqual(stats['col1']['mean'], 3.0)
        self.assertEqual(stats['col2']['mean'], 6.0)
        
    def test_fmt_float(self):
        """Test float formatting function."""
        self.assertEqual(fmt_float(3.14159), '3.1416')
        self.assertEqual(fmt_float(None), 'NA')
        self.assertEqual(fmt_float(np.nan), 'NA')
        self.assertEqual(fmt_float(np.inf), 'NA')


class TestDataAnalysis(unittest.TestCase):
    """Test cases for data analysis functionality."""
    
    def test_empty_dataframe_stats(self):
        """Test statistics computation on empty dataframe."""
        df = pd.DataFrame()
        stats = compute_basic_stats(df)
        self.assertEqual(len(stats), 0)
        
    def test_numeric_only_stats(self):
        """Test that only numeric columns are processed."""
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3],
            'text_col': ['a', 'b', 'c']
        })
        
        stats = compute_basic_stats(df)
        self.assertIn('numeric_col', stats)
        self.assertNotIn('text_col', stats)


if __name__ == '__main__':
    unittest.main()
