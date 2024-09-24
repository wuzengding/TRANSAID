import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from Bio.Seq import Seq
from Bio import SeqIO


# Function to encode base sequences (one-hot or integer encoding)
def encode_base_sequence(seq, max_len, encoding_type="one_hot"):
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'U': [0, 0, 0, 1]}
    base_encoding = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4}

    if encoding_type == 'one_hot':
        encoded_seq = np.zeros((max_len, 4))
        for i, nucleotide in enumerate(seq[:max_len]):
            encoded_seq[i] = encoding.get(nucleotide, [0, 0, 0, 0])
    else:
        encoded_seq = np.zeros(max_len, dtype=np.int64)
        for i, nucleotide in enumerate(seq[:max_len]):
            encoded_seq[i] = base_encoding.get(nucleotide, 0)

    return torch.tensor(encoded_seq, dtype=torch.float32 if encoding_type == 'one_hot' else torch.long)


# Custom dataset for mRNA base sequences without labels
class mRNADataset(Dataset):
    def __init__(self, fasta_file, max_len=5034, encoding_type="one_hot"):
        self.max_len = max_len
        self.encoding_type = encoding_type
        self.sequences = []
        self.seq_ids = []

        for record in SeqIO.parse(fasta_file, "fasta"):
            self.sequences.append(str(record.seq).upper())
            self.seq_ids.append(record.id)

        print(f"Loaded {len(self.sequences)} sequences from {fasta_file}.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        seq_id = self.seq_ids[idx]
        encoded_sequence = encode_base_sequence(sequence, self.max_len, self.encoding_type)
        true_seq_len = min(len(sequence), self.max_len)
        return encoded_sequence, seq_id, true_seq_len


# Define your TRANSAID models here (v1, v2, v3)
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

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.rb1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.rb2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.rb3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.permute(0, 2, 1)
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


# Function to predict and extract TIS and TTS
def predict(model, loader, device):
    model.eval()
    predictions = []
    seq_ids = []
    seq_lengths = []

    with torch.no_grad():
        for x, seq_id, seq_len in tqdm(loader, desc="Predicting"):
            x = x.to(device)
            outputs = model(x)
            outputs = outputs.reshape(-1, outputs.size(-1))
            outputs = outputs.argmax(dim=1)
            valid_mask = (x.sum(dim=-1) != 0).reshape(-1)
            outputs = outputs[valid_mask]
            predictions.append(outputs.cpu().numpy())
            seq_ids.append(np.array(list(seq_id)))
            seq_lengths.append(np.array(list(seq_len)))

    predictions = np.concatenate(predictions, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)
    seq_lengths = np.concatenate(seq_lengths, axis=0)

    return predictions, seq_ids, seq_lengths


# Translate mRNA to protein sequence based on predicted TIS and TTS
def translate_mrna_to_protein(seq, predictions, seq_id):
    tis_indices = np.where(predictions == 0)[0]
    tts_indices = np.where(predictions == 1)[0]

    if len(tis_indices) == 0 or len(tts_indices) == 0:
        print(f"Unable to find both TIS and TTS for {seq_id}")
        return None

    tis = tis_indices[0]
    tts = tts_indices[-1]

    if tts <= tis:
        print(f"TTS occurs before TIS for {seq_id}, skipping.")
        return None

    coding_region = seq[tis:tts]
    if len(coding_region) % 3 != 0:
        coding_region = coding_region[:-(len(coding_region) % 3)]

    coding_seq = Seq(coding_region)
    protein_sequence = coding_seq.translate(to_stop=True)
    return protein_sequence


# Main function
def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    test_dataset = mRNADataset(fasta_file=args.fasta_file, max_len=args.max_len, encoding_type=args.encoding_type)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = TRANSAID_v3().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    predictions, seq_ids, seq_lengths = predict(model, test_loader, device)

    # Output file for protein sequences
    protein_output_file = os.path.join(args.output_dir, f"{args.output_prefix}_predicted_proteins.faa")
    with open(protein_output_file, "w") as f_out:
        for seq, seq_id in zip(test_dataset.sequences, seq_ids):
            protein_sequence = translate_mrna_to_protein(seq, predictions, seq_id)
            if protein_sequence:
                f_out.write(f">{seq_id}_protein\n{protein_sequence}\n")

    print(f"Protein sequences saved to {protein_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run predictions and translate mRNA sequences using TRANSAID-2k model')
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the FASTA file containing mRNA sequences')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions and translations')
    parser.add_argument('--output_prefix', type=str, required=True, help='Prefix for the output files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--max_len', type=int, default=5034, help='Maximum length of sequences to include')
    parser.add_argument('--encoding_type', type=str, choices=['one_hot', 'base'], default='one_hot', help="Encoding type: 'one_hot' or 'base'")
    
    args = parser.parse_args()
    main(args)
