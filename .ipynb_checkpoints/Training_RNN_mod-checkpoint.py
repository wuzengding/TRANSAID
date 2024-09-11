import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MRNADataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = [f.split('_encoded.pt')[0] for f in os.listdir(data_dir) if f.endswith('_encoded.pt')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        seq_id = self.file_names[idx]
        X = torch.load(os.path.join(self.data_dir, f'{seq_id}_encoded.pt'),weights_only=False)
        y = torch.load(os.path.join(self.data_dir, f'{seq_id}_labels.pt'),weights_only=False)
        return X, y

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hn, cn) = self.rnn(x)
        out = self.fc(output)
        return out

def train_model(data_dir, model_save_path, num_epochs, batch_size, learning_rate, use_gpu, model_prefix, seed):
    set_seed(seed)
    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and use_gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    dataset = MRNADataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    # Get the shape of the first item in the dataset to determine input_size
    first_item = dataset[0]
    input_size = first_item[0].shape[1]  # Assuming shape is (seq_len, features)
    output_size = first_item[1].shape[1]  # Assuming shape is (seq_len, num_classes)
    print(f"Input size: {input_size}, Output size: {output_size}")
    
    # Assuming one-hot encoding (4 input features) and 3 output classes
    model = SimpleRNN(input_size=4, hidden_size=32, output_size=3).to(device)
    #criterion = nn.CrossEntropyLoss(ignore_index=-1)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            #print(f"X_batch shape: {X_batch.shape}")
            #print(f"y_batch shape: {y_batch.shape}")
            
            outputs = model(X_batch)
            
            #print(f"outputs shape before permute: {outputs.shape}")
            #outputs = outputs.permute(0, 2, 1)  # Change shape to (batch, num_classes, sequence_length)
            #print(f"outputs shape after permute: {outputs.shape}")
            
            #y_batch = y_batch.long()  # Ensure y_batch is of type Long
            y_batch = y_batch.float()  # Ensure y_batch is of type Long

            # 生成掩码：对于 X_batch 中全为 0 的位置，标记为 0；其他位置标记为 1
            mask = (X_batch.sum(dim=-1) != 0).float()

            # 计算逐元素损失
            loss = criterion(outputs, y_batch)

            # 只保留未被掩码掉的位置的损失
            loss = (loss * mask.unsqueeze(-1)).mean()
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')
        save_path = os.path.join(model_save_path, f"{model_prefix}_Epoch{epoch+1}_model.pth")
        torch.save(model.state_dict(), save_path)

    save_path = os.path.join(model_save_path, f"{model_prefix}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple RNN model on the given dataset and save the model.")
    parser.add_argument('-d','--data_dir', type=str, required=True, help='Directory containing the dataset.')
    parser.add_argument('-m','--model_save_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('-e','--num_epochs', type=int, default=10, help='Number of training epochs (default: 10).')
    parser.add_argument('-b','--batch_size', type=int, default=4, help='Batch size for training (default: 4).')
    parser.add_argument('-r','--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001).')
    parser.add_argument('-g','--use_gpu', type=int, default=0, help='GPU device to use (0 for GPU 0, 1 for GPU 1, -1 for CPU).')
    parser.add_argument('-p','--model_prefix', type=str, required=True, help='Prefix for output model')
    parser.add_argument('-s','--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    train_model(args.data_dir, args.model_save_path, args.num_epochs, args.batch_size, args.learning_rate, args.use_gpu, args.model_prefix, args.seed)