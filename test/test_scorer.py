# test/test_scorer.py

import unittest
import numpy as np
from src.scorer import BayesianScorer

class TestScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = BayesianScorer()
        
    def test_kozak_score(self):
        sequence = "GCCGCCATGGCG"
        score = self.scorer.calculate_kozak_score(sequence, 6)
        self.assertGreater(score, 0)
        
    # 添加更多测试用例...