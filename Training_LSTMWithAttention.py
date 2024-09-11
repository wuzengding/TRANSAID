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
        #X = torch.load(os.path.join(self.data_dir, f'{seq_id}_encoded.pt'), weights_only=True)
        #y = torch.load(os.path.join(self.data_dir, f'{seq_id}_labels.pt'), weights_only=True)
        X = torch.load(os.path.join(self.data_dir, f'{seq_id}_encoded.pt'))
        y = torch.load(os.path.join(self.data_dir, f'{seq_id}_labels.pt'))
        length = X.size(0)  # Get the actual length of the sequence

        # Check for NaN or Inf in inputs
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError(f"NaN or Inf found in input sequence {seq_id}_encoded.pt")
            
        return X, y, length

# Collate function to pad sequences and sort by length
def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*batch)
    
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # padding_value set to -1 for ignored positions
    
    return sequences, labels, torch.tensor(lengths)

# Define the LSTM model with Attention
class LSTMWithAttention(nn.Module):
    def __init__(self):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(32 * 2, 1)  # Attention mechanism
        self.fc = nn.Linear(32 * 2, 3)  # Output for each time step, predicting 3 classes

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
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

# Training function
def train_model(data_dir, model_save_path, num_epochs, batch_size, learning_rate, use_gpu, model_prefix):
    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and use_gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    dataset = MRNADataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = LSTMWithAttention().to(device)
    class_weights = torch.tensor([1.0, 10.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)  # Ignore padding label
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch, lengths in dataloader:
            X_batch, y_batch, lengths = X_batch.to(device), y_batch.to(device), lengths.to(device)
            optimizer.zero_grad()  # Zero the parameter gradients
            
            outputs = model(X_batch, lengths)  # Forward pass
            # outputs: (batch_size, seq_len, num_classes)
            
            # Check shapes before flattening
            #print(f"Model output shape: {outputs.shape}")
            #print(f"Target labels shape: {y_batch.shape}")
            
            # Compute loss
            loss = criterion(outputs.permute(0, 2, 1), y_batch)  # Permute to (batch_size, num_classes, seq_len)
            
            loss.backward()  # Backward pass

             # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
        save_path = os.path.join(model_save_path, f"{model_prefix}_Epoch{epoch+1}_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')

    save_path = os.path.join(model_save_path, f"{model_prefix}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RNN model on the given dataset and save the model.")
    parser.add_argument('-d','--data_dir', type=str, required=True, help='Directory containing the dataset.')
    parser.add_argument('-m','--model_save_path', type=str, required=True, help='Directory to save the trained model.')
    parser.add_argument('-e','--num_epochs', type=int, default=20, help='Number of training epochs (default: 20).')
    parser.add_argument('-b','--batch_size', type=int, default=4, help='Batch size for training (default: 4).')
    parser.add_argument('-r','--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001).')
    parser.add_argument('-g','--use_gpu', type=int, default=0, help='GPU device to use (0 for GPU 0, 1 for GPU 1, -1 for CPU).')
    parser.add_argument('-p','--model_prefix', type=str, required=True, help='Prefix for the saved model name.')

    args = parser.parse_args()

    train_model(args.data_dir, args.model_save_path, args.num_epochs, args.batch_size, args.learning_rate, args.use_gpu, args.model_prefix)
