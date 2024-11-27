# configs/model_config.py
from typing import List, Optional

class ModelConfig:
    """模型配置类"""
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_cnn: bool = True,
        max_seq_length: int = 5000,
        kmer_sizes: Optional[List[int]] = None,
        # 特征选择参数
        use_nucleotide: bool = True,  # 必选特征
        use_structure: bool = False,
        use_kmer: bool = False,
        use_thermodynamic: bool = False,
        use_conservation: bool = False,
        use_complexity: bool = False
    ):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_cnn = use_cnn
        self.max_seq_length = max_seq_length
        self.kmer_sizes = kmer_sizes or [3, 4, 5]
        
        # 模型架构配置
        self.use_hierarchical = True
        self.use_attention_pooling = True
        self.hidden_dim = embedding_dim * 4
        
        # 特征选择配置
        self.use_nucleotide = True  # 核苷酸特征始终为True
        self.use_structure = use_structure
        self.use_kmer = use_kmer
        self.use_thermodynamic = use_thermodynamic
        self.use_conservation = use_conservation
        self.use_complexity = use_complexity
        
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return self.__dict__