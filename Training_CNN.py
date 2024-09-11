import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 自定义数据集
class mRNADataset(Dataset):
    def __init__(self, data_dir, max_len=5049):
        self.data_dir = data_dir
        self.max_len = max_len
        print("self.max_len",self.max_len)
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('_encoded.pt')]
        
        self.valid_files = []
        for file in tqdm(self.file_list, desc="Train files"):
            try:
                x = torch.load(os.path.join(data_dir, file))
                if x.shape[0] <= self.max_len:
                    self.valid_files.append(file)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        print(f"Found {len(self.valid_files)} Train files out of {len(self.file_list)} total files.")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_name = self.valid_files[idx]
        seq_id = '_'.join(file_name.split('_')[:-1])

        x = torch.load(os.path.join(self.data_dir, f'{seq_id}_encoded.pt'),weights_only=False)
        y = torch.load(os.path.join(self.data_dir, f'{seq_id}_labels.pt'),weights_only=False)

        return x[:self.max_len], y[:self.max_len]


class TranslationAI_v3(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(TranslationAI_v3, self).__init__()
        
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

class TranslationAI_v2(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(TranslationAI_v2, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=1, padding='same')
        
        self.rb1 = nn.Sequential(*[ResidualBlock_v2(32, 26, 1) for _ in range(4)])
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, padding='same')
        
        self.rb2 = nn.Sequential(*[ResidualBlock_v2(32, 26, 2) for _ in range(4)])
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, padding='same')
        
        self.rb3 = nn.Sequential(*[ResidualBlock_v2(32, 36, 5) for _ in range(4)])
        self.conv4 = nn.Conv1d(32, 32, kernel_size=1, padding='same')
        
        self.conv5 = nn.Conv1d(32, 3, kernel_size=1, padding='same')

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

# TranslationAI-2k模型
class TranslationAI(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(TranslationAI, self).__init__()
        
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


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)

        # Create a mask for valid input regions or valid labels
        valid_mask = (x.sum(dim=-1) != 0)  # Option 1: mask based on input encoding
        # valid_mask = (y.sum(dim=-1) != 0)  # Option 2: mask based on labels
        
        valid_mask = valid_mask.reshape(-1)  # Flatten mask to match reshaped tensors
        
        # Reshape outputs and labels
        outputs = outputs.reshape(-1, outputs.size(-1))  # (batch_size * seq_len, num_classes)
        y = y.argmax(dim=2).view(-1)  # (batch_size * seq_len)
        
        # Apply the mask
        outputs = outputs[valid_mask]
        y = y[valid_mask]
        
        # Check if mask filtering results in empty tensors, skip in such cases
        if y.numel() == 0:
            continue
            
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 评估函数
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            
            # Create a mask for valid input regions or valid labels
            valid_mask = (x.sum(dim=-1) != 0)  # Option 1: mask based on input encoding
            #print('length of masked_length:', len(valid_mask[0][valid_mask[0] != False]))
            
            # valid_mask = (y.sum(dim=-1) != 0)  # Option 2: mask based on labels
            
            valid_mask = valid_mask.reshape(-1)  # Flatten mask to match reshaped tensors
            
            # Reshape outputs and labels
            outputs = outputs.reshape(-1, outputs.size(-1))  # (batch_size * seq_len, num_classes)
            y = y.argmax(dim=2).reshape(-1)  # (batch_size * seq_len)
            
            # Apply the mask
            outputs = outputs[valid_mask]
            y = y[valid_mask]

            # Check if mask filtering results in empty tensors, skip in such cases
            if y.numel() == 0:
                continue
            
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(loader)

def main(args):
    set_seed(args.seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load training data
    full_train_dataset = mRNADataset(args.data_dir, max_len=args.max_len)
    
    # Split into train and dev sets
    train_size = int(0.9 * len(full_train_dataset))
    dev_size = len(full_train_dataset) - train_size
    train_dataset, dev_dataset = random_split(full_train_dataset, [train_size, dev_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    if args.model_type == 'TranslationAI_v1':
        model = TranslationAI().to(device)
    elif args.model_type == 'TranslationAI_v2':
        model = TranslationAI_v2().to(device)
    elif args.model_type == 'TranslationAI_v3':
        model =  TranslationAI_v3().to(device)
    else:
        raise ValueError("Unsupported model type.")

    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_dev_loss = float('inf')
    no_improve = 0
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        dev_loss = evaluate(model, dev_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")
        
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir,'_'.join([args.prefix, 'best_model.pth'])))
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TranslationAI-2k model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing encoded data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model')
    parser.add_argument('--model_type', type=str, required=True, help='Select the model to training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_len', type=int, default=5049, help='Maximum length of sequences to include')
    parser.add_argument('--prefix', type=str, default="Train", help='Prefix of model')
    
    args = parser.parse_args()
    main(args)