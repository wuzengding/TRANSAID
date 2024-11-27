# tests/test_models.py
import unittest
import torch
from src.models import MultiModalTransformer, PositionalEncoding, ConvBlock
from configs import ModelConfig

class TestMultiModalTransformer(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.model = MultiModalTransformer(self.config)
        self.batch_size = 2
        self.seq_length = 10
        
        # 准备测试数据
        self.test_input = {
            'nucleotide': torch.randint(0, 8, (self.batch_size, self.seq_length)),
            'structure': torch.randint(0, 5, (self.batch_size, self.seq_length)),
            'kmer': torch.randn(self.batch_size, 4096),
            'complexity': torch.randn(self.batch_size, 2),
            'thermodynamic': torch.randn(self.batch_size, 4)
        }
    
    def test_forward_pass(self):
        """测试前向传播"""
        output = self.model(self.test_input)
        
        self.assertEqual(output.size(), (self.batch_size, self.seq_length, 3))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_attention_mask(self):
        """测试注意力掩码"""
        mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.bool)
        mask[:, 5:] = False
        
        output = self.model(self.test_input, mask=mask)
        
        self.assertEqual(output.size(), (self.batch_size, self.seq_length, 3))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_save_load(self):
        """测试模型保存和加载"""
        # 保存模型
        save_path = 'test_model.pt'
        self.model.save_checkpoint(save_path)
        
        # 加载模型
        loaded_model, _ = MultiModalTransformer.load_checkpoint(save_path)
        
        # 比较两个模型的输出
        with torch.no_grad():
            original_output = self.model(self.test_input)
            loaded_output = loaded_model(self.test_input)
            
            self.assertTrue(torch.allclose(original_output, loaded_output))
        
        # 清理
        import os
        os.remove(save_path)

class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.max_len = 100
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_len)
    
    def test_encoding(self):
        """测试位置编码"""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        output = self.pos_encoder(x)
        
        self.assertEqual(output.size(), x.size())
        self.assertTrue(torch.isfinite(output).all())

class TestConvBlock(unittest.TestCase):
    def setUp(self):
        self.in_channels = 64
        self.out_channels = 128
        self.conv_block = ConvBlock(self.in_channels, self.out_channels)
    
    def test_forward(self):
        """测试卷积块前向传播"""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, self.in_channels, seq_len)
        
        output = self.conv_block(x)
        
        self.assertEqual(output.size(), (batch_size, self.out_channels, seq_len))
        self.assertTrue(torch.isfinite(output).all())

if __name__ == '__main__':
    unittest.main()