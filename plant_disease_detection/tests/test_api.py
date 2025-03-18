# tests/test_api.py
import unittest
from src.api import create_app
import io

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        
    def test_health_check(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
    def test_predict_no_file(self):
        response = self.client.post('/api/predict')
        self.assertEqual(response.status_code, 400)
