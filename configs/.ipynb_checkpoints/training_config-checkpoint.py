# configs/training_config.py
class TrainingConfig:
    """训练配置类"""
    def __init__(
        self,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        epochs: int = 50,
        warmup_steps: int = 4000,
        gradient_clip: float = 1.0,
        fp16: bool = True,
        num_workers: int = 8,
        accumulation_steps: int = 8,
        early_stopping_patience: int = 5,
        validation_interval: int = 1000,
        save_interval: int = 5000,
        attention_chunk_size: int = 512
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.gradient_clip = gradient_clip
        self.fp16 = fp16
        self.num_workers = num_workers
        self.accumulation_steps = accumulation_steps
        self.attention_chunk_size = attention_chunk_size
        
        # 训练控制
        self.early_stopping_patience = early_stopping_patience
        self.validation_interval = validation_interval
        self.save_interval = save_interval
        
        # 优化器配置
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        
        # 学习率调度
        self.lr_schedule = 'cosine'  # ['cosine', 'linear', 'constant']
        self.min_lr = 1e-7
        self.warmup_ratio = 0.1
        
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return self.__dict__