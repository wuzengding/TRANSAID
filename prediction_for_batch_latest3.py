import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
import seaborn as sns
from collections import defaultdict
from Bio import SeqIO
import pickle


# 自定义数据集
class mRNADataset(Dataset):
    def __init__(self, data_dir, max_len=5034):
        self.data_dir = data_dir
        self.max_len = max_len
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('_encoded.pt')]
        
        self.valid_files = []
        for file in tqdm(self.file_list, desc="Validating files"):
            try:
                x = torch.load(os.path.join(data_dir, file), weights_only=True)
                if x.shape[0] <= self.max_len:
                    self.valid_files.append(file)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        print(f"Found {len(self.valid_files)} valid files out of {len(self.file_list)} total files.")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_name = self.valid_files[idx]
        seq_id = '_'.join(file_name.split('_')[:-1])

        x = torch.load(os.path.join(self.data_dir, f'{seq_id}_encoded.pt'), weights_only=False)
        y = torch.load(os.path.join(self.data_dir, f'{seq_id}_labels.pt'), weights_only=False)

        # Ensure x and y have the same sequence length
        seq_len = min(x.size(0), y.size(0), self.max_len)
        #x = x[:seq_len]
        #y = y[:seq_len]

        # Calculate the true sequence length excluding padding
        if args.model_type in ['TRANSAID_Embedding', 'TRANSAID_Embedding_v2'] :
            true_seq_len = (x != 0).sum().item() 
        else:
            true_seq_len = (x.sum(dim=1) != 0).sum().item() 
        
        # Pad or truncate to max_len
        if x.size(0) < self.max_len:
            print("warning short reads:")
            x_padded = torch.zeros((self.max_len, 4), dtype=torch.float32)
            x_padded[:x.size(0), :] = x
            y_padded = torch.full((self.max_len, 3), -1, dtype=torch.long)
            y_padded[:y.size(0), :] = y
            return x_padded, y_padded, seq_id, seq_len
        else:
            return x, y, seq_id, true_seq_len

def load_sequences_from_fasta(fasta_file):
    """
    Load sequences from FASTA file
    
    Args:
        fasta_file (str): Path to FASTA file
        
    Returns:
        Dict[str, str]: Dictionary mapping transcript IDs to sequences
    """
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Extract transcript ID without version number
        transcript_id = record.id.split('.')[0]
        sequences[transcript_id] = str(record.seq)
    return sequences

class TRANSAID_Transformer(nn.Module):
    def __init__(self, 
                 num_embeddings=5,      # 0=padding, 1-4=ACGT/U
                 embedding_dim=128,      # embedding维度
                 num_layers=4,          # transformer层数
                 nhead=8,               # 注意力头数
                 dim_feedforward=512,    # 前馈网络维度
                 dropout=0.1,           # dropout率
                 output_channels=3):     # 输出类别数
        super(TRANSAID_Transformer, self).__init__()
        
        # Embedding层
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出映射
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_channels)
        )

    def forward(self, x):
        # 生成padding mask
        padding_mask = (x == 0)  # (batch_size, seq_len)
        
        # Embedding和位置编码
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # 输出层
        output = self.output_layer(x)
        
        return output
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5049):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels//8, 1)
        self.key = nn.Conv1d(channels, channels//8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, length = x.size()
        
        query = self.query(x).view(batch_size, -1, length)
        key = self.key(x).view(batch_size, -1, length)
        value = self.value(x).view(batch_size, -1, length)
        
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        
        return x + self.gamma * out
        
class TRANSAID_Embedding_v2(nn.Module):
    def __init__(self, embedding_dim=128, output_channels=3):  # 增加embedding维度
        super().__init__()
        
        # Embedding层
        self.embedding = nn.Embedding(5, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.emb_dropout = nn.Dropout(0.1)
        
        # 多尺度卷积分支
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(embedding_dim, 32, kernel_size=k, padding='same')
            for k in [3, 5, 7]  # 多个卷积核大小
        ])
        
        # 主干网络
        self.conv1 = nn.Conv1d(96, 64, kernel_size=3, padding='same')  # 96 = 32*3
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        
        # 多尺度残差块
        self.rb1 = nn.Sequential(*[
            ResidualBlock_v2(64, 26, dilation=2**i) 
            for i in range(4)  # 使用不同的膨胀率
        ])
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.rb2 = nn.Sequential(*[
            ResidualBlock_v2(128, 26, dilation=2**i)
            for i in range(4)
        ])
        
        # 注意力机制
        self.attention = SelfAttention(128)
        
        # 输出层
        self.conv3 = nn.Conv1d(128, 64, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        self.conv4 = nn.Conv1d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.emb_dropout(x)
        x = x.permute(0, 2, 1)
        
        # 多尺度特征提取
        conv_outputs = []
        for conv in self.conv_branches:
            conv_outputs.append(conv(x))
        x = torch.cat(conv_outputs, dim=1)
        
        # 主干网络处理
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.rb1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.rb2(x)
        
        # 注意力处理
        x = self.attention(x)
        
        # 输出层
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.conv4(x)
        
        return x.permute(0, 2, 1)

class TRANSAID_Embedding(nn.Module):
    def __init__(self, embedding_dim=128, output_channels=3):
        super(TRANSAID_Embedding, self).__init__()
        
        # Embedding层：5个可能的值(0-4)，0用作padding
        self.embedding = nn.Embedding(
            num_embeddings=5,  # 0=padding, 1=A, 2=C, 3=G, 4=T/U
            embedding_dim=embedding_dim,
            padding_idx=0  # 指定padding_idx确保padding token的embedding始终为0
        )
        
        # 其余架构保持不变
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        self.rb1 = nn.Sequential(*[ResidualBlock_v2(32, 26, 1) for _ in range(4)])
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        
        self.rb2 = nn.Sequential(*[ResidualBlock_v2(64, 26, 2) for _ in range(4)])
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        
        self.rb3 = nn.Sequential(*[ResidualBlock_v2(128, 36, 5) for _ in range(4)])
        self.conv4 = nn.Conv1d(128, 32, kernel_size=1, padding='same')
        
        self.conv5 = nn.Conv1d(32, output_channels, kernel_size=1, padding='same')

    def forward(self, x):
        # x形状: (batch_size, seq_len)，值域: 0-4
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.rb1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.rb2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.rb3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, output_channels)
        return x
        
class TRANSAID_v3(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(TRANSAID_v3, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        self.rb1 = nn.Sequential(*[ResidualBlock_v2(32, 26, 1) for _ in range(4)])
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        
        self.rb2 = nn.Sequential(*[ResidualBlock_v2(64, 26, 2) for _ in range(4)])
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        
        self.rb3 = nn.Sequential(*[ResidualBlock_v2(128, 36, 5) for _ in range(4)])
        self.conv4 = nn.Conv1d(128, 32, kernel_size=1, padding='same')
        
        self.conv5 = nn.Conv1d(32, output_channels, kernel_size=1, padding='same')
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change from (batch, seq_len, channels) to (batch, channels, seq_len)
        
        #x = self.conv1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.rb1(x)
        #x = self.conv2(x)
        x = self.relu(self.bn2(self.conv2(x)))
        
        x = self.rb2(x)
        #x = self.conv3(x)
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = self.rb3(x)
        x = self.conv4(x)
        
        x = self.conv5(x)
        
        x = x.permute(0, 2, 1)  # Change back to (batch, seq_len, channels)
        return x

class ResidualBlock_v2(nn.Module):
    def __init__(self, channels, width, dilation):
        super(ResidualBlock_v2, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(channels, channels, width, dilation=dilation, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, width, dilation=dilation, padding='same')

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual

class TRANSAID_v2(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(TRANSAID_v2, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=1, padding='same')
        
        self.rb1 = nn.Sequential(*[ResidualBlock_v2(32, 26, 1) for _ in range(4)])
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, padding='same')
        
        self.rb2 = nn.Sequential(*[ResidualBlock_v2(32, 26, 2) for _ in range(4)])
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, padding='same')
        
        self.rb3 = nn.Sequential(*[ResidualBlock_v2(32, 36, 5) for _ in range(4)])
        self.conv4 = nn.Conv1d(32, 32, kernel_size=1, padding='same')
        
        self.conv5 = nn.Conv1d(32, 3, kernel_size=1, padding='same')
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change from (batch, seq_len, channels) to (batch, channels, seq_len)
        
        x = self.conv1(x)
        
        x = self.rb1(x)
        x = self.conv2(x)
        
        x = self.rb2(x)
        x = self.conv3(x)
        
        x = self.rb3(x)
        x = self.conv4(x)
        
        x = self.conv5(x)
        #x = self.softmax(x)
        
        x = x.permute(0, 2, 1)  # Change back to (batch, seq_len, channels)
        return x

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

# TRANSAID模型
class TRANSAID(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(TRANSAID, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
        )
        
        self.conv2 = nn.Conv1d(256, output_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change from (batch, seq_len, channels) to (batch, channels, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # Change back to (batch, seq_len, channels)
        return x

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hn, cn) = self.rnn(x)
        out = self.fc(output)
        return out
        
def save_predictions(predictions, true_labels, seq_ids, seq_lengths, output_dir, prefix):
    """
    保存预测结果到pkl文件，分为匹配和不匹配两组
    """
    import pickle
    import numpy as np
    
    # 整理数据为字典格式
    results = []
    current_idx = 0
    
    for seq_id, seq_len in zip(seq_ids, seq_lengths):
        # 获取当前转录本的预测和真实标签
        pred = predictions[current_idx:current_idx + seq_len]
        true = true_labels[current_idx:current_idx + seq_len]
        
        # 判断是否完全匹配
        is_match = np.array_equal(pred, true)
        
        # 保存结果
        results.append({
            'transcript_id': seq_id,
            'length': seq_len,
            'predictions': pred,
            'true_labels': true,
            'is_match': is_match
        })
        
        current_idx += seq_len
    
    # 分离匹配和不匹配的结果
    matching = [r for r in results if r['is_match']]
    non_matching = [r for r in results if not r['is_match']]
    
    # 保存结果
    with open(f'{output_dir}/{prefix}_matching_predictions.pkl', 'wb') as f:
        pickle.dump(matching, f)
    with open(f'{output_dir}/{prefix}_non_matching_predictions.pkl', 'wb') as f:
        pickle.dump(non_matching, f)
    
    print(f"Saved {len(matching)} matching and {len(non_matching)} non-matching predictions")
    
    # 返回保存的文件路径，方便后续使用
    return {
        'matching': f'{output_dir}/{prefix}_matching_predictions.pkl',
        'non_matching': f'{output_dir}/{prefix}_non_matching_predictions.pkl'
    }

def plot_confusion_matrix(y_true, y_pred, outdir, prefix):
    #y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # 计算每个类别的总数
    total_tis_num = np.sum(y_true == 0)  # Start_Codon(TIS)
    total_tts_num = np.sum(y_true == 1)  # Stop_Codon(TTS)
    total_non_tis_tts_num = np.sum(y_true == 2)  # Non-TTS/TIS

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{prefix} Confusion Matrix in Base Level')
    
    # Custom labels
    labels = ['Start_Codon(TIS)', 'Stop_Codon(TTS)', 'Non-TTS/TIS']

    # Plot numbers on the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(i, j, format(cm[i, j], 'd'), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=30, ha='right')
    plt.yticks(tick_marks, labels)

    plt.gca().invert_yaxis()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.subplots_adjust(bottom=0.3)  # 调整图表底部间距以显示备注

    # 添加备注信息
    note_text = (
        f"Total Start_Codon(TIS) num: {total_tis_num}\n"
        f"Total Stop_Codon(TTS) num: {total_tts_num}\n"
        f"Total Non-TTS/TIS num: {total_non_tis_tts_num}"
    )
    plt.figtext(0.5, 0.02, note_text, wrap=True, horizontalalignment='center', fontsize=10)

    plt.savefig(os.path.join(outdir, f"{prefix}_confusion_matrix_in_Base_level.png"))
    plt.show()
    print(f"Confusion matrix saved to {outdir}/{prefix}_confusion_matrix_in_Base_level.png")



# 预测函数
def predict(model, loader, device, output_dir):
    model.eval()
    predictions = []
    true_labels = []
    seq_ids = []
    seq_lengths = []


    with torch.no_grad():
        for x, y, seq_id, seq_len in tqdm(loader, desc="Predicting"):
            x = x.to(device)
            y = y.to(device)
            #print("xshape", x.shape)
            #print("yshape", y.shape)

            outputs = model(x)
                        
           
            # Reshape outputs and labels
            
            #print("outputs", outputs.shape)
            outputs = outputs.reshape(-1, outputs.size(-1))  # (batch_size * seq_len, num_classes)
            outputs = outputs.argmax(dim=1) ## for debug
            #print("outputs", outputs.shape)
            
            y = y.reshape(-1, y.size(-1))  ## for debug
            y = y.argmax(dim=1)  ## for debug

            #y = y.argmax(dim=2).reshape(-1) ## for debug

            # Apply the mask
            # Create a mask for valid input regions or valid labels
            if args.model_type in ['TRANSAID_Embedding','TRANSAID_Embedding_v2','TRANSAID_Transforemer']:
                valid_mask = (x != 0).reshape(-1)
            else:
                valid_mask = (x.sum(dim=-1) != 0).reshape(-1)

            #print("valid_mask:", len(valid_mask))
            #print("outputs1", outputs.shape)
            #print("yshape1", y.shape)
            outputs = outputs[valid_mask]
            y = y[valid_mask]
            
            #print("outputs2", outputs.shape)
            #print("yshape2", y.shape)
            predictions.append(outputs.cpu().numpy())
            true_labels.append(y.cpu().numpy())
            seq_ids.append(np.array(list(seq_id)))
            seq_lengths.append(np.array(list(seq_len)))

            
            ##### block for debug  ####
            '''
            reads_len = len(valid_mask[valid_mask != False])
            print("valid_mask size:", valid_mask.size() )
            print("outputs size:", outputs.size() )
            print("y size:", y.size() )
            print("reads_length", reads_len )
            print(seq_id)
            for index, (a, b) in enumerate(zip(outputs.cpu().numpy(),y.cpu().numpy())):
                if a != b:
                    print(index, a, b) 
            input()
            '''
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)
    seq_lengths = np.concatenate(seq_lengths, axis=0)
    
    return predictions, true_labels, seq_ids, seq_lengths


def calculate_metrics(y_true, y_pred, output_dir, prefix):
    #y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_classes = y_pred
    
    # 分别计算每个类别的精确率、召回率和 F1-score
    precision_per_class = precision_score(y_true, y_pred_classes, average=None, labels=[0, 1, 2])
    recall_per_class = recall_score(y_true, y_pred_classes, average=None, labels=[0, 1, 2])
    f1_per_class = f1_score(y_true, y_pred_classes, average=None, labels=[0, 1, 2])

    # 计算宏平均（不考虑类别不平衡）
    precision_macro = precision_score(y_true, y_pred_classes, average='macro')
    recall_macro = recall_score(y_true, y_pred_classes, average='macro')
    f1_macro = f1_score(y_true, y_pred_classes, average='macro')

    # 输出每个类别的精确率、召回率和 F1-score
    labels = ['Start_Codon(TIS)', 'Stop_Codon(TTS)', 'Non-TTS/TIS']
    for i, label in enumerate(labels):
        print(f"{label} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1-score: {f1_per_class[i]:.4f}")

    print(f"Macro Average - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1-score: {f1_macro:.4f}")

    # 调用函数绘制矩阵图
    plot_metrics_matrix(precision_per_class, recall_per_class, f1_per_class, 
                        precision_macro, recall_macro, f1_macro, 
                        labels, output_dir, prefix)

    return precision_per_class, recall_per_class, f1_per_class
    
    
# 绘制矩阵图的函数
def plot_metrics_matrix(precision, recall, f1, precision_macro, recall_macro, f1_macro, labels, output_dir, prefix):
    # Adding Macro Average to the beginning of the lists
    labels = ['Macro Average'] + labels
    precision = np.insert(precision, 0, precision_macro)
    recall = np.insert(recall, 0, recall_macro)
    f1 = np.insert(f1, 0, f1_macro)

    # Create a matrix with precision, recall, and f1-score as rows and categories as columns
    metrics_matrix = np.array([precision, recall, f1])

    # Define labels for metrics
    metrics_labels = ['Precision', 'Recall', 'F1-score']

    plt.figure(figsize=(8, 6))
    plt.imshow(metrics_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.75, vmax=1)
    plt.title(f'{prefix} Performance Matrix in Base Level')
    plt.colorbar()

    # Set tick marks for x and y axes
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=30, ha='right')
    plt.yticks(np.arange(len(metrics_labels)), metrics_labels)

    # Plot values inside the matrix
    for i in range(metrics_matrix.shape[0]):
        for j in range(metrics_matrix.shape[1]):
            plt.text(j, i, format(metrics_matrix[i, j], '.4f'),
                     ha="center", va="center",
                     color="white" if metrics_matrix[i, j] > metrics_matrix.max() / 1.2 else "black")

    #plt.tight_layout(rect=[0, 0.1, 0.8, 0.8])  # Adjust layout to fit the formula below
    plt.subplots_adjust(bottom=0.3)  # 调整图表底部间距以显示备注

    # Adding formulas below the matrix
    formula_text = (
        "Precision = TP / (TP + FP)\n"
        "Recall = TP / (TP + FN)\n"
        "F1-score = 2 * (Precision * Recall) / (Precision + Recall)"
    )
    plt.figtext(0.5, 0.02, formula_text, wrap=True, horizontalalignment='center', fontsize=10)

    plt.savefig(f"{output_dir}/{prefix}_performance_matrix_in_Base_level.png")
    plt.show()
    print(f"Metrics matrix saved to {output_dir}/{prefix}_performance_matrix_in_Base_level.png")


# Classify Transcript Predictions
def classify_transcript_predictions(true_labels, predictions, seq_ids, seq_lengths):
    results = {
        'Right ORF with all base correct': 0,
        'Right ORF with base incorrect partially': 0,
        'Wrong ORF but with right TIS':0,
        'Wrong ORF but with right TTS': 0,
        'Other Errors': 0
    }
    total_transcripts = len(seq_ids)
    print("total_transcripts",total_transcripts)
    
    indices_dict = {}
    current_index = 0

    idx = 0
    for i, seq_id in enumerate(seq_ids):
        seq_len = seq_lengths[i]
        
        true_seq = true_labels[idx: idx+seq_len]
        pred_seq = predictions[idx: idx+seq_len]

        tis_true = (true_seq == 0)  # Start_Codon (TIS)
        tts_true = (true_seq == 1)  # Stop_Codon (TTS)
        nontistts_true = (true_seq == 2)  # no-TIS/TTS 
        tis_pred = (pred_seq == 0)
        tts_pred = (pred_seq == 1)
        nontistts_pred = (pred_seq == 2) 

        all_correct = np.array_equal(true_seq, pred_seq)
        tis_correct = np.array_equal(tis_true, tis_pred)
        tts_correct = np.array_equal(tts_true, tts_pred)
        nontistts_correct = np.array_equal(nontistts_true, nontistts_pred)
        

        # Check for partial correctness
        #partial_tis = np.any(tis_true & tis_pred)
        #partial_tts = np.any(tts_true & tts_pred)
        #partially_correct = (partial_tis and partial_tts) and nontistts_correct
        partial_tis = np.sum(tis_true & tis_pred)>=2
        partial_tts = np.sum(tts_true & tts_pred)>=2
        partial_nontistts = np.sum((nontistts_true == nontistts_pred) == 0) <=1
        partially_correct = (partial_tis and partial_tts) and partial_nontistts
        
        if all_correct:
            #print(partial_nontistts)
            results['Right ORF with all base correct'] += 1
        elif partially_correct:
            #print(partial_nontistts)
            results['Right ORF with base incorrect partially'] += 1
        elif tis_correct and not tts_correct:
            results['Wrong ORF but with right TIS'] += 1
        elif not tis_correct and tts_correct:
            results['Wrong ORF but with right TTS'] += 1
        else:
            results['Other Errors'] += 1
        
        idx += seq_len
    return results, total_transcripts

class PositionAnalyzer:
    def __init__(self):
        self.position_stats = defaultdict(list)
    
    def analyze_single_transcript(self, predictions: np.ndarray, seq_id: str, seq_len: int) -> Dict:
        """
        Analyze TIS/TTS predictions distribution in a single transcript
        """
        # Initialize empty value for positions if not found
        tis_relative_positions = np.array([])
        tts_relative_positions = np.array([])
        
        # Find TIS and TTS positions
        tis_positions = np.where(predictions == 0)[0]
        tts_positions = np.where(predictions == 1)[0]
        
        # Basic statistics dictionary
        stats = {
            'transcript_id': seq_id,
            'transcript_type': seq_id.split('_')[0],
            'length': seq_len,
            'tis_count': len(tis_positions),
            'tts_count': len(tts_positions),
            'tis_density': len(tis_positions)/seq_len if seq_len > 0 else 0,
            'tts_density': len(tts_positions)/seq_len if seq_len > 0 else 0,
            'tis_mean_position': 0,
            'tts_mean_position': 0,
            'tis_std_position': 0,
            'tts_std_position': 0,
            'tis_relative_positions': tis_relative_positions,
            'tts_relative_positions': tts_relative_positions,
        }
        
        # Calculate position statistics if sites exist
        if len(tis_positions) > 0:
            stats['tis_relative_positions'] = tis_positions / seq_len
            stats['tis_mean_position'] = np.mean(tis_positions) / seq_len
            stats['tis_std_position'] = np.std(tis_positions) / seq_len if len(tis_positions) > 1 else 0
        
        if len(tts_positions) > 0:
            stats['tts_relative_positions'] = tts_positions / seq_len
            stats['tts_mean_position'] = np.mean(tts_positions) / seq_len
            stats['tts_std_position'] = np.std(tts_positions) / seq_len if len(tts_positions) > 1 else 0
        
        # Analyze distances between sites if multiple sites exist
        if len(tis_positions) > 1:
            tis_distances = np.diff(tis_positions)
            stats['tis_min_distance'] = np.min(tis_distances)
            stats['tis_mean_distance'] = np.mean(tis_distances)
        else:
            stats['tis_min_distance'] = 0
            stats['tis_mean_distance'] = 0
        
        if len(tts_positions) > 1:
            tts_distances = np.diff(tts_positions)
            stats['tts_min_distance'] = np.min(tts_distances)
            stats['tts_mean_distance'] = np.mean(tts_distances)
        else:
            stats['tts_min_distance'] = 0
            stats['tts_mean_distance'] = 0
            
        # Add debug information
        print(f"Processing transcript {seq_id} with length {seq_len}")
        
        # Store all statistics
        for key, value in stats.items():
            if key not in self.position_stats:
                self.position_stats[key] = []
            self.position_stats[key].append(value)
            
        #print(f"Current stats length: {len(self.position_stats['transcript_type'])}")
        #print(f"Current tis_mean_position length: {len(self.position_stats['tis_mean_position'])}")
        
        return stats
    
    def plot_position_distributions(self, output_dir: str, prefix: str):
        """
        Generate visualization plots for position distributions
        """
        # 1. Density Plot of TIS/TTS relative positions
        plt.figure(figsize=(12, 6))
        for transcript_type in ['NM', 'NR', 'XM', 'XR']:
            # 获取特定类型的所有位置数据
            type_indices = [i for i, t in enumerate(self.position_stats['transcript_type']) 
                           if t == transcript_type]
        
            if not type_indices:  # 如果没有这种类型的转录本，跳过
                continue
            
            # TIS positions - 收集所有位置
            tis_positions = []
            for idx in type_indices:
                if idx < len(self.position_stats['tis_relative_positions']):
                    pos_array = self.position_stats['tis_relative_positions'][idx]
                    if isinstance(pos_array, np.ndarray):
                        tis_positions.extend(pos_array.tolist())
        
            # TTS positions - 收集所有位置
            tts_positions = []
            for idx in type_indices:
                if idx < len(self.position_stats['tts_relative_positions']):
                    pos_array = self.position_stats['tts_relative_positions'][idx]
                    if isinstance(pos_array, np.ndarray):
                        tts_positions.extend(pos_array.tolist())
        
            # 绘制密度图
            if tis_positions:
                sns.kdeplot(data=tis_positions, label=f'{transcript_type} TIS', linestyle='--')
            if tts_positions:
                sns.kdeplot(data=tts_positions, label=f'{transcript_type} TTS', linestyle='-')
    
        plt.xlabel('Relative Position in Transcript')
        plt.ylabel('Density')
        plt.title(f'{prefix} Distribution of TIS/TTS Positions')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{prefix}_position_distribution.png"))
        plt.close()
    
        # 2. Count Distribution
# Count Distribution - All in one figure
        fig = plt.figure(figsize=(20, 10))
        
        # 创建网格布局 2x2
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
        
        # TIS counts for NM and NR
        ax1 = fig.add_subplot(gs[0, 0])
        for transcript_type in ['NM', 'NR']:
            type_indices = [i for i, t in enumerate(self.position_stats['transcript_type']) 
                          if t == transcript_type]
            if not type_indices:
                continue
        
            counts = [self.position_stats['tis_count'][i] for i in type_indices]
            if counts:  # 确保有数据才画图
                sns.histplot(data=counts, label=transcript_type, ax=ax1, alpha=0.5)
    
        ax1.set_title('Distribution of TIS Counts (NM & NR)')
        ax1.set_xlabel('Number of TIS')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # TIS counts for XM and XR
        ax2 = fig.add_subplot(gs[0, 1])
        for transcript_type in ['XM', 'XR']:
            type_indices = [i for i, t in enumerate(self.position_stats['transcript_type']) 
                          if t == transcript_type]
            if not type_indices:
                continue
        
            counts = [self.position_stats['tis_count'][i] for i in type_indices]
            if counts:  # 确保有数据才画图
                sns.histplot(data=counts, label=transcript_type, ax=ax2, alpha=0.5)
    
        ax2.set_title('Distribution of TIS Counts (XM & XR)')
        ax2.set_xlabel('Number of TIS')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # TTS counts for NM and NR
        ax3 = fig.add_subplot(gs[1, 0])
        for transcript_type in ['NM', 'NR']:
            type_indices = [i for i, t in enumerate(self.position_stats['transcript_type']) 
                          if t == transcript_type]
            if not type_indices:
                continue
        
            counts = [self.position_stats['tts_count'][i] for i in type_indices]
            if counts:  # 确保有数据才画图
                sns.histplot(data=counts, label=transcript_type, ax=ax3, alpha=0.5)
    
        ax3.set_title('Distribution of TTS Counts (NM & NR)')
        ax3.set_xlabel('Number of TTS')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # TTS counts for XM and XR
        ax4 = fig.add_subplot(gs[1, 1])
        for transcript_type in ['XM', 'XR']:
            type_indices = [i for i, t in enumerate(self.position_stats['transcript_type']) 
                          if t == transcript_type]
            if not type_indices:
                continue
        
            counts = [self.position_stats['tts_count'][i] for i in type_indices]
            if counts:  # 确保有数据才画图
                sns.histplot(data=counts, label=transcript_type, ax=ax4, alpha=0.5)
    
        ax4.set_title('Distribution of TTS Counts (XM & XR)')
        ax4.set_xlabel('Number of TTS')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 添加总标题
        fig.suptitle(f'{prefix} Distribution of TIS/TTS Counts per Transcript Type', fontsize=16)
        plt.savefig(os.path.join(output_dir, f"{prefix}_count_distribution.png"))
        plt.close()

    def print_summary_statistics(self):
        """Print summary statistics for each transcript type"""
        #print(f"Total transcripts in stats: {len(self.position_stats['transcript_type'])}")
        #print(f"Total positions stats: {len(self.position_stats['tis_mean_position'])}")
        for transcript_type in ['NM', 'NR', 'XM', 'XR']:
            type_mask = np.array(self.position_stats['transcript_type']) == transcript_type
            if not any(type_mask):
                continue
                
            print(f"\nSummary Statistics for {transcript_type}:")
            print("----------------------------------------")
            print(f"Number of transcripts: {sum(type_mask)}")
            
            # TIS statistics
            tis_counts = np.array(self.position_stats['tis_count'])[type_mask]
            print("\nTIS Statistics:")
            print(f"Average TIS per transcript: {np.mean(tis_counts):.2f}")
            print(f"Median TIS per transcript: {np.median(tis_counts):.2f}")
            print(f"Std TIS per transcript: {np.std(tis_counts):.2f}")
            
            # TTS statistics
            tts_counts = np.array(self.position_stats['tts_count'])[type_mask]
            print("\nTTS Statistics:")
            print(f"Average TTS per transcript: {np.mean(tts_counts):.2f}")
            print(f"Median TTS per transcript: {np.median(tts_counts):.2f}")
            print(f"Std TTS per transcript: {np.std(tts_counts):.2f}")
            
            # Position statistics
            if sum(type_mask) > 0:
                tis_mean_positions = [p for p in np.array(self.position_stats['tis_mean_position'])[type_mask] if not isinstance(p, list)]
                tts_mean_positions = [p for p in np.array(self.position_stats['tts_mean_position'])[type_mask] if not isinstance(p, list)]
                
                if tis_mean_positions:
                    print("\nPosition Statistics:")
                    print(f"Average TIS relative position: {np.mean(tis_mean_positions):.3f}")
                if tts_mean_positions:
                    print(f"Average TTS relative position: {np.mean(tts_mean_positions):.3f}")

class SiteAnalyzer:
    def __init__(self):
        """Initialize the analyzer with storage for various statistics"""
        self.stats = defaultdict(list)
        self.codon_map = {
            'ATG': 'START',
            'TAA': 'STOP',
            'TAG': 'STOP',
            'TGA': 'STOP'
        }
    
    def analyze_transcript(self, 
                         sequence: str,
                         predictions: np.ndarray,
                         transcript_id: str) -> Dict:
        """
        Analyze a single transcript's TIS/TTS predictions
        
        Args:
            sequence: Original RNA sequence
            predictions: Model predictions (0=TIS, 1=TTS, 2=Non-TIS/TTS)
            transcript_id: Transcript identifier
        """
        # Get transcript type (NM/NR)
        trans_type = transcript_id.split('_')[0]
        
        # Find all predicted sites
        tis_positions = np.where(predictions == 0)[0]
        tts_positions = np.where(predictions == 1)[0]
        
        # Analyze continuity
        tis_stats = self._analyze_site_continuity(tis_positions, 'TIS')
        tts_stats = self._analyze_site_continuity(tts_positions, 'TTS')
        
        # Analyze potential codons
        tis_codon_stats = self._analyze_codons(sequence, tis_positions, 'TIS')
        tts_codon_stats = self._analyze_codons(sequence, tts_positions, 'TTS')
        
        # Combine statistics
        stats = {
            'transcript_id': transcript_id,
            'transcript_type': trans_type,
            'length': len(sequence),
            **tis_stats,
            **tts_stats,
            **tis_codon_stats,
            **tts_codon_stats
        }
        
        # Store all statistics
        for key, value in stats.items():
            self.stats[key].append(value)
        
        return stats
    
    def _analyze_site_continuity(self, positions: np.ndarray, site_type: str) -> Dict:
        """
        Analyze the continuity of predicted sites
        """
        if len(positions) < 1:
            return {
                f'{site_type}_count': 0,
                f'{site_type}_gaps': [],
                f'{site_type}_continuous_lengths': [],
                f'{site_type}_mean_gap': 0,
                f'{site_type}_gap_std': 0,
                f'{site_type}_mean_continuous_length': 0
            }
        
        # Calculate gaps between positions
        gaps = np.diff(positions)
        
        # Find continuous regions (gap = 1)
        continuous_regions = np.split(positions, np.where(gaps > 1)[0] + 1)
        continuous_lengths = [len(region) for region in continuous_regions]
        
        return {
            f'{site_type}_count': len(positions),
            f'{site_type}_gaps': gaps.tolist(),
            f'{site_type}_continuous_lengths': continuous_lengths,
            f'{site_type}_mean_gap': np.mean(gaps) if len(gaps) > 0 else 0,
            f'{site_type}_gap_std': np.std(gaps) if len(gaps) > 0 else 0,
            f'{site_type}_mean_continuous_length': np.mean(continuous_lengths)
        }
    
    def _analyze_codons(self, sequence: str, positions: np.ndarray, site_type: str) -> Dict:
        """
        Analyze codon sequences at predicted sites
        """
        valid_codons = []
        kozak_scores = []
        
        for pos in positions:
            if pos + 2 < len(sequence):
                codon = sequence[pos:pos+3]
                valid_codons.append(self.codon_map.get(codon, 'INVALID'))
                
                # For TIS sites, calculate Kozak score
                if site_type == 'TIS' and pos >= 3 and pos + 4 < len(sequence):
                    kozak = sequence[pos-3:pos+4]
                    kozak_scores.append(self._calculate_kozak_score(kozak))
        
        return {
            f'{site_type}_valid_codon_ratio': sum(1 for c in valid_codons if c != 'INVALID') / len(valid_codons) if valid_codons else 0,
            f'{site_type}_kozak_score': np.mean(kozak_scores) if kozak_scores else 0
        }
    
    def _calculate_kozak_score(self, kozak_seq: str) -> float:
        """
        Calculate Kozak sequence score
        Perfect Kozak: (GCC)GCCACC|ATG|G
        """
        if len(kozak_seq) != 7:
            return 0.0
            
        score = 0.0
        # Check -3 position (high importance)
        if kozak_seq[3] in 'AG':
            score += 0.4
        
        # Check +4 position (high importance)
        if kozak_seq[6] in 'G':
            score += 0.4
        
        # Check other positions
        for i, base in enumerate(kozak_seq[:3]):
            if base in 'GC':
                score += 0.067  # (0.2 distributed over 3 positions)
                
        return score
    
    def plot_continuity_statistics(self, output_dir: str, prefix: str):
        """
        Plot various continuity statistics
        """
        # 1. Gap Distribution Plot
        plt.figure(figsize=(12, 6))
        for trans_type in ['NM', 'NR']:
            type_indices = [i for i, t in enumerate(self.stats['transcript_type']) 
                           if t == trans_type]
        
            if not type_indices:  # 如果没有这种类型的转录本，跳过
                continue
            
            # Collect all gaps for TIS
            tis_gaps = []
            for idx in type_indices:
                if idx < len(self.stats['TIS_gaps']):
                    gaps = self.stats['TIS_gaps'][idx]
                    if isinstance(gaps, list):
                        tis_gaps.extend(gaps)
        
            # Collect all gaps for TTS
            tts_gaps = []
            for idx in type_indices:
                if idx < len(self.stats['TTS_gaps']):
                    gaps = self.stats['TTS_gaps'][idx]
                    if isinstance(gaps, list):
                        tts_gaps.extend(gaps)
        
            # Plot distributions if we have data
            if tis_gaps:
                sns.kdeplot(data=tis_gaps, 
                       label=f'{trans_type} TIS gaps', 
                       linestyle='--')
            if tts_gaps:
                sns.kdeplot(data=tts_gaps, 
                       label=f'{trans_type} TTS gaps', 
                       linestyle='-')
    
        plt.xlabel('Gap Size')
        plt.ylabel('Density')
        plt.xlim(-100,100)
        plt.title(f'{prefix} Distribution of Gaps Between Predicted Sites')
        plt.legend()
        plt.savefig(f'{output_dir}/{prefix}_gap_distribution.png')
        plt.close()
    
# 2. Continuous Length Distribution - TIS
        plt.figure(figsize=(12, 6))
        for trans_type in ['NM', 'NR']:
            type_indices = [i for i, t in enumerate(self.stats['transcript_type']) 
                           if t == trans_type]
            
            if not type_indices:
                continue
            
            # Collect TIS lengths
            tis_lengths = []
            for idx in type_indices:
                if idx < len(self.stats['TIS_continuous_lengths']):
                    lengths = self.stats['TIS_continuous_lengths'][idx]
                    if isinstance(lengths, list):
                        tis_lengths.extend(lengths)
            
            # Plot TIS distribution
            if tis_lengths:
                sns.histplot(data=tis_lengths, 
                           label=f'{trans_type} TIS', 
                           alpha=0.5,
                           binwidth=1,
                           stat='probability')
        
        plt.xlabel('Continuous Region Length')
        plt.ylabel('Probability')
        plt.title(f'{prefix} Distribution of TIS Continuous Prediction Lengths')
        plt.legend()
        plt.savefig(f'{output_dir}/{prefix}_TIS_continuous_length_distribution.png')
        plt.close()

        # 3. Continuous Length Distribution - TTS
        plt.figure(figsize=(12, 6))
        for trans_type in ['NM', 'NR']:
            type_indices = [i for i, t in enumerate(self.stats['transcript_type']) 
                           if t == trans_type]
            
            if not type_indices:
                continue
            
            # Collect TTS lengths
            tts_lengths = []
            for idx in type_indices:
                if idx < len(self.stats['TTS_continuous_lengths']):
                    lengths = self.stats['TTS_continuous_lengths'][idx]
                    if isinstance(lengths, list):
                        tts_lengths.extend(lengths)
            
            # Plot TTS distribution
            if tts_lengths:
                sns.histplot(data=tts_lengths, 
                           label=f'{trans_type} TTS', 
                           alpha=0.5,
                           binwidth=1,
                           stat='probability')
        
        plt.xlabel('Continuous Region Length')
        plt.ylabel('Probability')
        plt.title(f'{prefix} Distribution of TTS Continuous Prediction Lengths')
        plt.legend()
        plt.savefig(f'{output_dir}/{prefix}_TTS_continuous_length_distribution.png')
        plt.close()
    
    def print_summary_statistics(self):
        """Print summary statistics by transcript type"""
        for trans_type in ['NM', 'NR']:
            type_indices = [i for i, t in enumerate(self.stats['transcript_type']) 
                           if t == trans_type]
        
            if not type_indices:
                continue
            
            print(f"\nSummary Statistics for {trans_type}:")
            print("-" * 50)
        
            # TIS Statistics
            tis_counts = [self.stats['TIS_count'][i] for i in type_indices]
            print(f"\nTIS Statistics:")
            print(f"Average sites per transcript: {np.mean(tis_counts):.2f}")
        
            # Get all continuous lengths for TIS
            all_tis_lengths = []
            for idx in type_indices:
                if idx < len(self.stats['TIS_continuous_lengths']):
                    lengths = self.stats['TIS_continuous_lengths'][idx]
                    if isinstance(lengths, list):
                        all_tis_lengths.extend(lengths)
        
            if all_tis_lengths:
                print(f"Median continuous length: {np.median(all_tis_lengths):.2f}")
                print(f"Mean continuous length: {np.mean(all_tis_lengths):.2f}")
                print(f"Max continuous length: {np.max(all_tis_lengths):.2f}")
        
            # Codon statistics
            if self.stats.get('TIS_valid_codon_ratio'):
                valid_codon_ratios = [self.stats['TIS_valid_codon_ratio'][i] 
                                    for i in type_indices]
                print(f"Valid codon ratio: {np.mean(valid_codon_ratios):.2f}")
        
            if trans_type == 'NM' and self.stats.get('TIS_kozak_score'):
                kozak_scores = [self.stats['TIS_kozak_score'][i] 
                              for i in type_indices]
                print(f"Average Kozak score: {np.mean(kozak_scores):.2f}")
        
            # TTS Statistics
            tts_counts = [self.stats['TTS_count'][i] for i in type_indices]
            print(f"\nTTS Statistics:")
            print(f"Average sites per transcript: {np.mean(tts_counts):.2f}")
        
            # Get all continuous lengths for TTS
            all_tts_lengths = []
            for idx in type_indices:
                if idx < len(self.stats['TTS_continuous_lengths']):
                    lengths = self.stats['TTS_continuous_lengths'][idx]
                    if isinstance(lengths, list):
                        all_tts_lengths.extend(lengths)
        
            if all_tts_lengths:
                print(f"Median continuous length: {np.median(all_tts_lengths):.2f}")
                print(f"Mean continuous length: {np.mean(all_tts_lengths):.2f}")
                print(f"Max continuous length: {np.max(all_tts_lengths):.2f}")
        
            if self.stats.get('TTS_valid_codon_ratio'):
                valid_codon_ratios = [self.stats['TTS_valid_codon_ratio'][i] 
                                    for i in type_indices]
                print(f"Valid codon ratio: {np.mean(valid_codon_ratios):.2f}")

def analyze_transcripts(sequences: Dict[str, str], 
                       predictions: Dict[str, np.ndarray],
                       output_dir: str,
                       prefix: str) -> SiteAnalyzer:
    """
    Analyze multiple transcripts
    """
    analyzer = SiteAnalyzer()
    
    for transcript_id in sequences:
        if transcript_id in predictions:
            analyzer.analyze_transcript(
                sequences[transcript_id],
                predictions[transcript_id],
                transcript_id
            )
    
    analyzer.plot_continuity_statistics(output_dir, prefix)
    analyzer.print_summary_statistics()
    
    return analyzer

def analyze_predictions(predictions: np.ndarray, seq_ids: np.ndarray, 
                      seq_lengths: np.ndarray, output_dir: str, prefix: str) -> PositionAnalyzer:
    """
    Analyze predictions for all transcripts
    """
    analyzer = PositionAnalyzer()
    
    current_pos = 0
    for i, (seq_id, seq_len) in enumerate(zip(seq_ids, seq_lengths)):
        # Get predictions for current transcript
        transcript_predictions = predictions[current_pos:current_pos + seq_len]
        analyzer.analyze_single_transcript(transcript_predictions, seq_id, seq_len)
        current_pos += seq_len
    
    analyzer.plot_position_distributions(output_dir, prefix)
    analyzer.print_summary_statistics()
    
    return analyzer

# Plot Transcript Performance
def plot_transcript_performance(results, total, output_dir, prefix):
    labels = [f'group{i+1}' for i in range(len(results.keys()))]
    descriptions = list(results.keys())
    counts = list(results.values())
    ratios = [count / total for count in counts]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, ratios, color=['green', 'red', 'blue', 'orange', 'purple'])

    # Adjust text position for each bar
    for bar, count, ratio in zip(bars, counts, ratios):
        height = bar.get_height()
        if height < 0.9:  # Adjust the threshold for when to move text to the top
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{count} ({ratio:.2%})',
                     ha='center', va='bottom', fontsize=10, color='black')
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{count} ({ratio:.2%})',
                     ha='center', va='center', fontsize=10, color='black')
    
    plt.title(f'{prefix} Accuracy in Triplet Codon Level')
    plt.xlabel('Performance Category')
    plt.ylabel('Ratio')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bars, [f'{a}:{b}' for a,b in zip(labels, descriptions)] , loc='upper right')  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_triplet_performance.png"))
    plt.show()
    print(f"Transcript performance plot saved to {output_dir}/{prefix}_transcript_performance.png")


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    test_dataset = mRNADataset(args.data_dir, max_len=args.max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load model
    # Model selection based on model_type
    if args.model_type == 'SimpleLSTM':
        model = SimpleLSTM().to(device)
    elif args.model_type == 'LSTMWithAttention':
        model = LSTMWithAttention().to(device)
    elif args.model_type == 'SimpleRNN':
        model = SimpleRNN(input_size=4, hidden_size=32, output_size=3).to(device)
    elif args.model_type == 'TransformerModel':
        model = TransformerModel(input_dim=5, num_classes=3, max_len=max_len).to(device)
    elif args.model_type == 'TRANSAID_v1':
        model = TRANSAID().to(device)
    elif args.model_type == 'TRANSAID_v2':
        model = TRANSAID_v2().to(device)
    elif args.model_type == 'TRANSAID_v3':
        model = TRANSAID_v3().to(device)
    elif args.model_type == 'TRANSAID_Embedding':
        model = TRANSAID_Embedding(embedding_dim=64, output_channels=3).to(device)
    elif args.model_type == "TRANSAID_Embedding_v2":
        model = TRANSAID_Embedding_v2(embedding_dim=128,output_channels=3).to(device)
    elif args.model_type == 'TRANSAID_Transformer':
        model = TRANSAID_Transformer().to(device)
    else:
        raise ValueError("Unsupported model type.")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Run predictions
    predictions, true_labels, seq_ids, seq_lengths = predict(model, test_loader, device, args.output_dir)

    '''
    true_labels_str = np.array2string(true_labels, separator=',')
    predictions_str = np.array2string(predictions, separator=',')
    with open(os.path.join(args.data_dir, "results.txt"), "w") as f:
        np.savetxt(f, true_labels, fmt='%d', header='True Labels', comments='')
        f.write('\n')  # Add a newline for separation
        np.savetxt(f, predictions, fmt='%d', header='Predictions', comments='')
    
    print("true_labels",true_labels)
    print("predictions",predictions)
    '''
    result_files = save_predictions(predictions, true_labels, seq_ids, seq_lengths, 
                                  args.output_dir, args.prefix)
    '''
    # 添加位置分析
    print("\nAnalyzing TIS/TTS position distributions...")
    position_analyzer = analyze_predictions(predictions, seq_ids, seq_lengths, 
                                         args.output_dir, args.prefix)
    # Add site analysis
    print("\nAnalyzing TIS/TTS sites distribution...")
    
    # 准备预测数据
    transcript_predictions = {}
    current_pos = 0
    for seq_id, seq_len in zip(seq_ids, seq_lengths):
        transcript_predictions[seq_id] = predictions[current_pos:current_pos + seq_len]
        current_pos += seq_len
    
    # 加载FASTA序列
    sequences = load_sequences_from_fasta(args.fasta_file)
    
    # 运行站点分析
    site_analyzer = analyze_transcripts(
        sequences=sequences,
        predictions=transcript_predictions,
        output_dir=args.output_dir,
        prefix=f"{args.prefix}"
    )
    '''
    # Calculate metrics
    calculate_metrics(true_labels, predictions, args.output_dir, args.prefix)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, args.output_dir, args.prefix)

    # Classify performance
    results, total_transcripts = classify_transcript_predictions(true_labels, predictions, seq_ids, seq_lengths)

    # Plot transcript prediction performance
    plot_transcript_performance(results, total_transcripts, args.output_dir, args.prefix)

    # Calculate scores for ROC curve
    #y_true_binary = (y_true == 2).astype(int)  # Binary labels for the positive class (e.g., Start_Codon(TIS))
    #y_scores = y_pred[:,2]  # Predicted scores for the positive class
    #plot_roc_curve(y_true_binary, y_scores, args.output_dir, args.prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run predictions using TRANSAID-2k model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing encoded data')
    parser.add_argument('--fasta_file', type=str, required=True,
                      help='Path to the FASTA file containing reference sequences')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('-t','--model_type', type=str, required=True, help='Model type: SimpleLSTM, LSTMWithAttention, SimpleRNN,  TransformerModel, TRANSAID, TRANSAID_v2, TRANSAID_v3, TRANSAID_Embedding, TRANSAID_Embedding_v2, TRANSAID_Transforemer')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions and plots')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--max_len', type=int, default=5034, help='Maximum length of sequences to include')
    parser.add_argument('--prefix',type=str, default="validation", help='Prefix for output png')
    

    args = parser.parse_args()
    main(args)
