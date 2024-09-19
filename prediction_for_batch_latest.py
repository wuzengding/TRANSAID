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
        x = x[:seq_len]
        y = y[:seq_len]

        # Calculate the true sequence length excluding padding
        true_seq_len = (x.sum(dim=1) != 0).sum().item() 
        
        # Pad or truncate to max_len
        if x.size(0) < self.max_len:
            x_padded = torch.zeros((self.max_len, 4), dtype=torch.float32)
            x_padded[:x.size(0), :] = x
            y_padded = torch.full((self.max_len, 3), -1, dtype=torch.long)
            y_padded[:y.size(0), :] = y
            return x_padded, y_padded, seq_id, seq_len
        else:
            return x, y, seq_id, true_seq_len


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
            #print("seq_id", seq_id)
            #print("seq_len", seq_len)

            outputs = model(x)
                        
           
            # Reshape outputs and labels
            outputs = outputs.reshape(-1, outputs.size(-1))  # (batch_size * seq_len, num_classes)
            outputs = outputs.argmax(dim=1)
            
            y = y.reshape(-1, y.size(-1))
            y = y.argmax(dim=1)


            # Apply the mask
            # Create a mask for valid input regions or valid labels
            valid_mask = (x.sum(dim=-1) != 0)
            valid_mask = valid_mask.reshape(-1)

            outputs = outputs[valid_mask]
            y = y[valid_mask]
            
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
        partial_tis = np.any(tis_true & tis_pred)
        partial_tts = np.any(tts_true & tts_pred)
        partially_correct = (partial_tis and partial_tts) and nontistts_correct

        if all_correct:
            results['Right ORF with all base correct'] += 1
        elif partially_correct:
            results['Right ORF with base incorrect partially'] += 1
        elif tis_correct and not tts_correct:
            results['Wrong ORF but with right TIS'] += 1
        elif not tis_correct and tts_correct:
            results['Wrong ORF but with right TTS'] += 1
        else:
            results['Other Errors'] += 1
        
        idx += seq_len
    return results, total_transcripts

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
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('-t','--model_type', type=str, required=True, help='Model type: SimpleLSTM, LSTMWithAttention, SimpleRNN,  TransformerModel, TRANSAID, TRANSAID_v2,TRANSAID_v3 ')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions and plots')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--max_len', type=int, default=5034, help='Maximum length of sequences to include')
    parser.add_argument('--prefix',type=str, default="validation", help='Prefix for output png')
    

    args = parser.parse_args()
    main(args)
