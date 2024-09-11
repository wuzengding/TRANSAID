import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import argparse

# Define custom dataset class for loading .pt files
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
        return X, y

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=4, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 3)  # 输出 3 个类别，而不是 1 个
    
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output)
        return out

# Training function
def train_model(data_dir, model_save_path, num_epochs, batch_size, learning_rate, use_gpu, model_prefix):
    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and use_gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    dataset = MRNADataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleRNN().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()  # Zero the parameter gradients
            
            lengths = torch.tensor([X_batch.size(1)] * X_batch.size(0)).to(device)  # All sequences have the same length
            outputs = model(X_batch, lengths)  # Forward pass
            
            # Check shapes before flattening
            print(f"Model output shape: {outputs.shape}")
            print(f"Target labels shape: {y_batch.shape}")
            
            outputs = outputs.permute(0, 2, 1)  # 调整维度以适应 CrossEntropyLoss
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the parameters
            
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

    args = parser.parse_args()

    train_model(args.data_dir, args.model_save_path, args.num_epochs, args.batch_size, args.learning_rate, args.use_gpu, args.model_prefix)
