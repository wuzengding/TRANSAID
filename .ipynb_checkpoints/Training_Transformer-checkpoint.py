import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
        X = torch.load(os.path.join(self.data_dir, f'{seq_id}_encoded.pt'), weights_only=True)
        y = torch.load(os.path.join(self.data_dir, f'{seq_id}_labels.pt'), weights_only=True)
        length = X.size(0)  # Get the actual length of the sequence
        
        return X, y, length

# Collate function to pad sequences and sort by length
def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*batch)
    
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # padding_value set to -1 for ignored positions
    
    return sequences, labels, torch.tensor(lengths)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, d_model=32, dim_feedforward=128, max_len=10000):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        
        # Create positional encoding with max_len as a parameter
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        seq_len = x.size(1)

        # Debug: Check the input range of x
        if torch.any(x < 0) or torch.any(x >= self.embedding.num_embeddings):
            print(f"Invalid input indices detected in the input tensor: {x}")
            print(f"Input tensor shape: {x.shape}")
            unique_values = torch.unique(x)
            print(f"Unique values in the input tensor: {unique_values}")
            raise ValueError("Input tensor contains indices out of the embedding layer's range.")

        # Apply embedding and add positional encoding
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        
        # Debug: Output shapes after embedding and positional encoding
        print(f"Shape after embedding and positional encoding: {x.shape}")
        
        x = self.transformer_encoder(x)
        out = self.fc(x)
        return out

# Training function
def train_model(data_dir, model_save_path, num_epochs, batch_size, learning_rate, use_gpu, model_prefix, max_len):
    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and use_gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    dataset = MRNADataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    input_dim = 5  # Number of unique tokens (0 for padding, 1-4 for A, C, G, T/U)
    num_classes = 3  # Number of output classes
    model = TransformerModel(input_dim=input_dim, num_classes=num_classes, max_len=max_len).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding label
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch, lengths in dataloader:
            print("Batch sequence lengths:",lengths)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()  # Zero the parameter gradients
            
            # Check for invalid input indices
            if torch.any(X_batch < 0) or torch.any(X_batch >= input_dim):
                print(f"Invalid input indices detected in batch: {X_batch}")
                print(f"Unique values in the batch: {torch.unique(X_batch)}")
                raise ValueError("Input tensor contains indices out of the embedding layer's range.")

            outputs = model(X_batch, lengths)  # Forward pass
            # outputs: (batch_size, seq_len, num_classes)

            # Debug: Check for NaN/Inf in outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN or Inf detected in model outputs")
                
            # Debug: Ensure that there are no NaN or inf values
            assert not torch.isnan(outputs).any(), "Model output contains NaN"
            assert not torch.isinf(outputs).any(), "Model output contains inf"

            # Check label range
            assert torch.all(y_batch >= 0) and torch.all(y_batch < num_classes), "Labels out of range"
            
            # Compute loss
            loss = criterion(outputs.permute(0, 2, 1), y_batch)  # Permute to (batch_size, num_classes, seq_len)
            
            loss.backward()  # Backward pass
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')

    save_path = os.path.join(model_save_path, f"{model_prefix}_transformer_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model on the given dataset and save the model.")
    parser.add_argument('-d','--data_dir', type=str, required=True, help='Directory containing the dataset.')
    parser.add_argument('-m','--model_save_path', type=str, required=True, help='Directory to save the trained model.')
    parser.add_argument('-e','--num_epochs', type=int, default=20, help='Number of training epochs (default: 20).')
    parser.add_argument('-b','--batch_size', type=int, default=4, help='Batch size for training (default: 4).')
    parser.add_argument('-r','--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001).')
    parser.add_argument('-g','--use_gpu', type=int, default=0, help='GPU device to use (0 for GPU 0, 1 for GPU 1, -1 for CPU).')
    parser.add_argument('-p','--model_prefix', type=str, required=True, help='Prefix for the saved model name.')
    parser.add_argument('-l','--max_len', type=int, default=10000, help='Maximum sequence length for positional encoding (default: 10000).')

    args = parser.parse_args()

    train_model(args.data_dir, args.model_save_path, args.num_epochs, args.batch_size, args.learning_rate, args.use_gpu, args.model_prefix, args.max_len)

