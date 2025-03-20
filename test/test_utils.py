# test/test_utils.py

import unittest
import torch
import numpy as np
from src.utils import sequence_to_tensor

class TestUtils(unittest.TestCase):
    def test_sequence_to_tensor(self):
        sequence = "ATGC"
        tensor = sequence_to_tensor(sequence)
        self.assertEqual(tensor.shape, (4, 5))
        
    # 添加更多测试用例...