import argparse
import random
from Bio import SeqIO
import torch
import numpy as np
import os

class DataPreparator:
    def __init__(self, fasta_file, gbff_file, flank_size, train_ratio):
        self.fasta_file = fasta_file
        self.gbff_file = gbff_file
        self.flank_size = flank_size
        self.train_ratio = train_ratio
        self.transcript_info = self.parse_gbff()

    def parse_gbff(self):
        cds_annotations = {}
        for record in SeqIO.parse(self.gbff_file, "genbank"):
            for feature in record.features:
                if feature.type == "CDS":
                    cds_annotations[record.id.split(',')[0]] = \
                    (int(feature.location.start), int(feature.location.end))
        return cds_annotations

    @staticmethod
    def one_hot_encode(sequence):
        mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'U': [0,0,0,1]}
        return np.array([mapping.get(nuc, [0,0,0,0]) for nuc in sequence])

    def pad_sequence(self, seq):
        chunk_size = 6000
        padding = chunk_size - (len(seq) % chunk_size)
        return np.pad(seq, ((0, padding), (0, 0)), mode='constant')

    def add_flanking(self, seq):
        return np.pad(seq, ((self.flank_size, self.flank_size), (0, 0)), mode='constant')

    def encode_mrna(self, sequence):
        encoded = self.one_hot_encode(sequence)
        padded = self.pad_sequence(encoded)
        flanked = self.add_flanking(padded)
        return flanked

    def encode_labels(self, tis, tts, seq_length):
        labels = np.zeros((seq_length, 3))
        labels[:, 0] = 1  # Set "neither" as default
        labels[tis-1, 1] = 1
        labels[tis-1, 0] = 0
        labels[tts-1, 2] = 1
        labels[tts-1, 0] = 0
        return labels

    def prepare_data(self):
        data = []
        for record in SeqIO.parse(self.fasta_file, "fasta"):
            if record.id in self.transcript_info:
                tis, tts = self.transcript_info[record.id]
                sequence = str(record.seq)
                X = self.encode_mrna(sequence)
                y = self.encode_labels(tis, tts, len(sequence) + 2*self.flank_size)
                data.append((record.id, X, y))

        random.shuffle(data)
        split_idx = int(len(data) * self.train_ratio)
        return data[:split_idx], data[split_idx:]

    def save_data(self, train_data, val_data, output_dir):
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)

        for dataset, subdir in [(train_data, 'train'), (val_data, 'validation')]:
            for transcript_id, X, y in dataset:
                torch.save({
                    'X': torch.tensor(X),
                    'y': torch.tensor(y)
                }, os.path.join(output_dir, subdir, f"{transcript_id}.pt"))

        print(f"Data prepared and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for TranslationAI")
    parser.add_argument("--fasta", required=True, help="Path to the GRCh38_latest_rna.fna file")
    parser.add_argument("--gbff", required=True, help="Path to the GRCh38_latest_rna.gbff file")
    parser.add_argument("--flank_size", type=int, default=1000, help="Flanking sequence size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--output_dir", required=True, help="Output directory for prepared data")
    
    args = parser.parse_args()
    
    preparator = DataPreparator(args.fasta, args.gbff, args.flank_size, args.train_ratio)
    train_data, val_data = preparator.prepare_data()
    preparator.save_data(train_data, val_data, args.output_dir)