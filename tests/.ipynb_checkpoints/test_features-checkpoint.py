# tests/test_features.py

import unittest
import torch
import numpy as np
from src.features import SequenceFeatureExtractor, StructureFeatureExtractor
from configs import ModelConfig

class TestSequenceFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.extractor = SequenceFeatureExtractor(self.config)
        self.test_seq = "ATGCCGTAA"
    
    def test_nucleotide_encoding(self):
        """测试核苷酸编码"""
        features = self.extractor.extract_features(self.test_seq)
        
        self.assertIn('nucleotide', features)
        self.assertEqual(len(features['nucleotide']), len(self.test_seq))
        self.assertTrue(torch.is_tensor(features['nucleotide']))
        self.assertEqual(features['nucleotide'].dtype, torch.long)
    
    def test_kmer_features(self):
        """测试k-mer特征"""
        features = self.extractor.extract_features(self.test_seq)
        
        self.assertIn('kmer', features)
        self.assertTrue(torch.is_tensor(features['kmer']))
        self.assertEqual(features['kmer'].dtype, torch.float32)
    
    def test_sequence_complexity(self):
        """测试序列复杂度特征"""
        features = self.extractor.extract_features(self.test_seq)
        
        self.assertIn('complexity', features)
        self.assertTrue(torch.is_tensor(features['complexity']))
        self.assertEqual(features['complexity'].dtype, torch.float32)
    
    def test_invalid_sequence(self):
        """测试无效序列处理"""
        invalid_seq = "ATGXXXTAA"
        features = self.extractor.extract_features(invalid_seq)
        
        self.assertIn('nucleotide', features)
        self.assertTrue(all(x < len(self.extractor.nuc_vocab) 
                          for x in features['nucleotide']))

class TestStructureFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.extractor = StructureFeatureExtractor(self.config)
        self.test_seq = "ATGCCGTAA"
    
    def test_structure_prediction(self):
        """测试结构预测"""
        features = self.extractor.extract_features(self.test_seq)
        
        self.assertIn('structure', features)
        self.assertEqual(len(features['structure']), len(self.test_seq))
        self.assertTrue(torch.is_tensor(features['structure']))
    
    def test_thermodynamic_features(self):
        """测试热力学特征"""
        features = self.extractor.extract_features(self.test_seq)
        
        self.assertIn('thermodynamic', features)
        self.assertTrue(torch.is_tensor(features['thermodynamic']))
        self.assertEqual(features['thermodynamic'].dtype, torch.float32)
    
    def test_local_structures(self):
        """测试局部结构特征"""
        features = self.extractor.extract_features(self.test_seq)
        
        self.assertIn('local_structures', features)
        self.assertTrue(torch.is_tensor(features['local_structures']))
        self.assertEqual(features['local_structures'].dtype, torch.float32)
    
    def test_cache_mechanism(self):
        """测试缓存机制"""
        # 第一次提取特征
        features1 = self.extractor.extract_features(self.test_seq)
        
        # 第二次提取相同序列的特征
        features2 = self.extractor.extract_features(self.test_seq)
        
        # 验证两次结果相同
        for key in features1:
            self.assertTrue(torch.equal(features1[key], features2[key]))

if __name__ == '__main__':
    unittest.main()