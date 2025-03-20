# test/test_predictor.py

import unittest
import torch
import numpy as np
from src.predictor import TranslationPredictor
from src.models import TRANSAID_Embedding

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = TranslationPredictor(
            model_path='path/to/test/model.pth',
            device=-1
        )
        
    def test_predict_single(self):
        sequence = "ATGGCTAAGTAA"
        results = self.predictor.predict_single(sequence)
        self.assertIsNotNone(results)