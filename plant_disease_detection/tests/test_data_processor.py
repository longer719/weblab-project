# tests/test_data_processor.py

import unittest
import numpy as np
from src.data_processing.data_processor import DataProcessor  # 更新导入路径
from configs.config import Config

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor(Config)
        
    def test_normalize_image(self):
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        normalized = self.processor.normalize_image(test_image)
        self.assertTrue(normalized.max() <= 1.0)
        self.assertTrue(normalized.min() >= 0.0)

if __name__ == '__main__':
    unittest.main()