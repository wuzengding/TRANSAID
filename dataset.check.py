import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse

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

def main(args):
    # Create the dataset and dataloader
    dataset = MRNADataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    # Verify the data loading process
    for X_batch, y_batch in dataloader:
        if torch.any(X_batch < 0) or torch.any(X_batch >= args.embedding_size):
            print(f"Invalid input indices detected in batch: {X_batch}")
            print(f"Unique values in the batch: {torch.unique(X_batch)}")
            break  # Stop the training to debug the issue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRNA Dataset Loader")
    parser.add_argument('--data_dir', type=str, reguired=True, help='Directory containing the encoded data files')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for data loading')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the data')
    parser.add_argument('--embedding_size', type=int, default=, help='Size of the embedding layer')

    args = parser.parse_args()

    main(args)
