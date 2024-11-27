import unittest
import torch
from src.data import TranslationSiteDataset, create_data_loaders
import tempfile
import pickle
import os

class TestTranslationSiteDataset(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.sequences = [
            {
                'nucleotide': torch.randint(0, 8, (10,)),
                'structure': torch.randint(0, 5, (10,)),
                'kmer': torch.randn(4096),
                'complexity': torch.randn(2),
                'thermodynamic': torch.randn(4)
            }
            for _ in range(5)  # 5个测试序列
        ]
        self.labels = [torch.randint(0, 3, (10,)) for _ in range(5)]
        
        self.dataset = TranslationSiteDataset(self.sequences, self.labels)
    
    def test_dataset_length(self):
        """测试数据集长度"""
        self.assertEqual(len(self.dataset), 5)
    
    def test_dataset_getitem(self):
        """测试数据集获取项"""
        seq, label = self.dataset[0]
        
        self.assertIsInstance(seq, dict)
        self.assertIn('nucleotide', seq)
        self.assertIn('structure', seq)
        self.assertIsInstance(label, torch.Tensor)
    
    def test_dataset_stats(self):
        """测试数据集统计信息"""
        stats = self.dataset.stats
        
        self.assertIn('num_sequences', stats)
        self.assertIn('max_length', stats)
        self.assertIn('label_distribution', stats)
    
    def test_dataset_split(self):
        """测试数据集拆分"""
        train_data, val_data, test_data = self.dataset.split(
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        self.assertEqual(len(train_data) + len(val_data) + len(test_data), 
                        len(self.dataset))
    
    def test_save_load(self):
        """测试数据集保存和加载"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file_path = tmp.name
            
            # 保存数据集
            self.dataset.save(file_path)
            
            # 加载数据集
            loaded_dataset = TranslationSiteDataset.from_file(file_path)
            
            # 验证数据集大小
            self.assertEqual(len(loaded_dataset), len(self.dataset))
            
            # 验证数据内容
            orig_seq, orig_label = self.dataset[0]
            loaded_seq, loaded_label = loaded_dataset[0]
            
            for key in orig_seq:
                self.assertTrue(torch.equal(orig_seq[key], loaded_seq[key]))
            self.assertTrue(torch.equal(orig_label, loaded_label))
        
        # 清理
        os.remove(file_path)

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # 创建小型测试数据集
        self.sequences = [
            {
                'nucleotide': torch.randint(0, 8, (10,)),
                'structure': torch.randint(0, 5, (10,)),
                'kmer': torch.randn(4096),
                'complexity': torch.randn(2),
                'thermodynamic': torch.randn(4)
            }
            for _ in range(10)
        ]
        self.labels = [torch.randint(0, 3, (10,)) for _ in range(10)]
        self.dataset = TranslationSiteDataset(self.sequences, self.labels)
    
    def test_dataloader_creation(self):
        """测试数据加载器创建"""
        batch_size = 2
        dataloader = create_data_loaders(
            self.dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        self.assertEqual(len(dataloader), 5)  # 10/2 = 5 batches
    
    def test_batch_format(self):
        """测试批次格式"""
        dataloader = create_data_loaders(
            self.dataset,
            batch_size=2,
            shuffle=False
        )
        
        batch = next(iter(dataloader))
        
        self.assertIsInstance(batch, dict)
        self.assertIn('nucleotide', batch)
        self.assertIn('structure', batch)
        self.assertIn('labels', batch)
        
        # 检查批次维度
        self.assertEqual(batch['nucleotide'].size(0), 2)
        self.assertEqual(batch['labels'].size(0), 2)
    
    def test_padding(self):
        """测试序列填充"""
        # 创建不等长序列
        sequences = [
            {
                'nucleotide': torch.randint(0, 8, (5,)),
                'structure': torch.randint(0, 5, (5,))
            },
            {
                'nucleotide': torch.randint(0, 8, (10,)),
                'structure': torch.randint(0, 5, (10,))
            }
        ]
        labels = [torch.randint(0, 3, (5,)), torch.randint(0, 3, (10,))]
        dataset = TranslationSiteDataset(sequences, labels)
        
        dataloader = create_data_loaders(dataset, batch_size=2)
        batch = next(iter(dataloader))
        
        # 检查是否填充到最大长度
        self.assertEqual(batch['nucleotide'].size(1), 10)
        self.assertEqual(batch['labels'].size(1), 10)

if __name__ == '__main__':
    unittest.main()