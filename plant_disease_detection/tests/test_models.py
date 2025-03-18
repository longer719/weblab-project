# tests/test_models.py

import unittest
import torch
from src.models.plant_classifier import PlantClassifier  # 更新导入路径
from src.models.disease_detector import DiseaseDetector  # 更新导入路径
from src.models.model_config import ModelConfig  # 更新导入路径

class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.batch_size = 4
        
    def test_plant_classifier(self):
        model = PlantClassifier(self.config)
        x = torch.randn(self.batch_size, 3, 224, 224)
        output = model(x)
        self.assertEqual(
            output.shape, 
            (self.batch_size, self.config.CLASSIFIER_CONFIG['num_classes'])
        )
        
    def test_disease_detector(self):
        model = DiseaseDetector(self.config)
        model.eval()  # 设置为评估模式
        x = torch.randn(self.batch_size, 3, 224, 224)
        output = model(x)
    
        # 验证输出格式
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), self.batch_size)
        # 验证每个输出包含预测框和分数
        for pred in output:
            self.assertIn('boxes', pred)
            self.assertIn('scores', pred)
            self.assertIn('labels', pred)

if __name__ == '__main__':
    unittest.main()