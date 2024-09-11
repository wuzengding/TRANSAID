import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# 设置打印选项以显示完整张量
torch.set_printoptions(threshold=float('inf'))

# Dataset class
class MRNADataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = [f.split('_encoded.pt')[0] for f in os.listdir(data_dir) if f.endswith('_encoded.pt')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        seq_id = self.file_names[idx]
        X = torch.load(os.path.join(self.data_dir, f'{seq_id}_encoded.pt'))
        y = torch.load(os.path.join(self.data_dir, f'{seq_id}_labels.pt'))
        length = X.size(0)  # Get the actual length of the sequence

        # 打印加载的数据形状
        #print(f"Loaded {seq_id} X: {X}")
        #print(f"Loaded {seq_id} y: {y}")
        #print(f"Loaded {seq_id}: X shape: {X.shape}, y shape: {y.shape}, length: {length}")

        return X, y, length

# Collate function
def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*batch)
    
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    
    return sequences, labels, torch.tensor(lengths)

# SimpleLSTM Model
class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(32 * 2, 3)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output)
        return out

# LSTM with Attention Model
class LSTMWithAttention(nn.Module):
    def __init__(self):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(32 * 2, 1)
        self.fc = nn.Linear(32 * 2, 3)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        '''
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        attn_weights = torch.softmax(self.attention(output), dim=1)
        context_vector = torch.sum(attn_weights * output, dim=1)

        #out = self.fc(context_vector)
        out = self.fc(output)
        '''
        packed_output, (ht, ct) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply attention
        attention_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)
        attended_output = output * attention_weights.unsqueeze(-1)  # Shape: (batch_size, seq_len, hidden_size*2)

        # Check for NaN or Inf in model output
        if torch.isnan(attended_output).any() or torch.isinf(attended_output).any():
            raise ValueError(f"NaN or Inf found in model output")

        out = self.fc(attended_output)  # Shape: (batch_size, seq_len, num_classes)
        
        return out

# SimpleRNN Model
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=4, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 3)  
        
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output)
        return out  #  return logits

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, d_model=32, dim_feedforward=128, max_len=10000):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        out = self.fc(x)
        return out

# TestPredictor class
class TestPredictor:
    def __init__(self, model_type, model_path, test_data_dir, max_len, gpu_id):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")
        self.test_data_dir = test_data_dir
        self.max_len = max_len

        # Model selection based on model_type
        if model_type == 'SimpleLSTM':
            self.model = SimpleLSTM()
        elif model_type == 'LSTMWithAttention':
            self.model = LSTMWithAttention()
        elif model_type == 'SimpleRNN':
            self.model = SimpleRNN()
        elif model_type == 'TransformerModel':
            self.model = TransformerModel(input_dim=5, num_classes=3, max_len=max_len)
        else:
            raise ValueError("Unsupported model type.")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # 打印模型结构
        print(self.model)
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self):
        dataset = MRNADataset(self.test_data_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        all_logits = []
        all_true_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch, lengths in dataloader:
                X_batch, y_batch, lengths = X_batch.to(self.device), y_batch.to(self.device), lengths.to(self.device)
                
                logits = self.model(X_batch, lengths)
                print("logits shape:",logits.shape)
                print("y_batch shape:", y_batch.shape)
                
                all_logits.append(logits.cpu().numpy())
                all_true_labels.append(y_batch.cpu().numpy())

                #print("all_logits",all_logits)
                #print("all_true_labels",all_true_labels)
        return np.concatenate(all_logits), np.concatenate(all_true_labels)

def plot_confusion_matrix(y_true, y_pred, outdir, prefix):
    print(y_true)
    print(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    plt.figure(figsize=(8, ))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{prefix} Confusion Matrix')
    
    # Custom labels
    labels = ['Non-TTS/TIS', 'Stop_Codon(TTS)', 'Start_Codon(TIS)']

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
    plt.subplots_adjust(bottom=0.2) 
    
    plt.savefig(os.path.join(outdir, f"{prefix}_confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix saved to {outdir}/{prefix}_confusion_matrix.png")

def plot_roc_curve(y_true, y_scores, outdir, prefix):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outdir, f"{prefix}_roc_curve.png"))
    plt.close()
    print(f"ROC curve saved to {outdir}/{prefix}_roc_curve.png")

def run_prediction_old(model_type, model_path, test_data_dir, max_len, outdir, prefix, gpu_id):
    predictor = TestPredictor(model_type, model_path, test_data_dir, max_len, gpu_id)
    logits, true_labels = predictor.predict()

    # Save predictions
    save_path = os.path.join(outdir, f"{prefix}_predictions.npy")
    np.save(save_path, logits)
    print(f"Predictions saved to {save_path}")

    # Flatten predictions and true labels for confusion matrix
    y_pred = np.argmax(logits, axis=-1).flatten()
    y_true = true_labels.flatten()
    
    print(f"Shape of y_pred: {y_pred.shape}")
    print(f"Shape of y_true: {y_true.shape}")
    
    # Filter out padding labels
    valid_idx = y_true != -1
    y_pred = y_pred[valid_idx]
    y_true = y_true[valid_idx]
    
    # 使用以下代码来检查并确保 logits 的形状正确
    print("Logits shape before filtering:", logits.shape)

    # 如果 logits 是 3D (N, L, C)，那么我们需要根据第一个维度进行展平
    logits = logits.reshape(-1, logits.shape[-1])  # 将其展平成 2D 数组
    logits = logits[valid_idx]  # 先过滤有效的预测
    print("Logits shape after filtering:", logits.shape)
    
    # 确保 y_scores 的形状正确
    y_scores = logits[:, 2]  # 选择第3类的概率或得分

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, outdir, prefix)

    # Plot ROC curve (assuming binary classification or one-vs-rest for multi-class)
    y_true_binary = (y_true == 2).astype(int)  # Binary labels for the positive class
    plot_roc_curve(y_true_binary, y_scores, outdir, prefix)
    
def run_prediction(model_type, model_path, test_data_dir, max_len, outdir, prefix, gpu_id):
    predictor = TestPredictor(model_type, model_path, test_data_dir, max_len, gpu_id)
    logits, true_labels = predictor.predict()
    
    #print('logits',logits)
    #print('true_labels', true_labels)

    # Save predictions
    save_path = os.path.join(outdir, f"{prefix}_predictions.npy")
    np.save(save_path, logits)
    print(f"Predictions saved to {save_path}")

    # Flatten predictions and true labels for confusion matrix
    y_pred = np.argmax(logits, axis=-1).flatten()
    y_true = true_labels.flatten()
    
    print(f"Shape of y_pred: {y_pred.shape}")
    print(f"Shape of y_true: {y_true.shape}")
    
    # Filter out padding labels
    valid_idx = y_true != -1
    y_pred = y_pred[valid_idx]
    y_true = y_true[valid_idx]
    
    # 使用以下代码来检查并确保 logits 的形状正确
    print("Logits shape before filtering:", logits.shape)

    # 如果 logits 是 3D (N, L, C)，那么我们需要根据第一个维度进行展平
    logits = logits.reshape(-1, logits.shape[-1])  # 将其展平成 2D 数组
    logits = logits[valid_idx]  # 先过滤有效的预测
    print("Logits shape after filtering:", logits.shape)
    
    # 确保 y_scores 的形状正确
    y_scores = logits[:, 2]  # 选择第3类的概率或得分

    # Post-process predictions: Ensure TIS and TTS cover 3 positions
    for i in range(1, len(y_pred)):
        if y_pred[i] == 2 and y_pred[i-1] != 2:
            y_pred[max(0, i-2):i+1] = 2  # Mark the previous 2 positions as 2 (TIS)
        if y_pred[i] == 1 and y_pred[i-1] != 1:
            y_pred[max(0, i-2):i+1] = 1  # Mark the previous 2 positions as 1 (TTS)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, outdir, prefix)
    
    # Plot ROC curve (assuming binary classification or one-vs-rest for multi-class)
    y_true_binary = (y_true == 2).astype(int)  # Binary labels for the positive class
    plot_roc_curve(y_true_binary, y_scores, outdir, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions using a pre-trained model.")
    parser.add_argument('-t','--model_type', type=str, required=True, help='Model type: SimpleLSTM, LSTMWithAttention, SimpleRNN, or TransformerModel.')
    parser.add_argument('-m','--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('-d','--test_data_dir', type=str, required=True, help='Directory containing the test dataset.')
    parser.add_argument('-o','--outdir', type=str, required=True, help='Directory to save the predictions.')
    parser.add_argument('-p','--prefix', type=str, required=True, help='Prefix for the saved predictions.')
    parser.add_argument('-g','--gpu', type=int, default=0, help='GPU device to use (0 for GPU 0, 1 for GPU 1, -1 for CPU).')
    parser.add_argument('-l','--max_len', type=int, default=10000, help='Maximum sequence length for positional encoding (used for TransformerModel).')

    args = parser.parse_args()
    run_prediction(args.model_type, args.model_path, args.test_data_dir, args.max_len, args.outdir, args.prefix, args.gpu)
