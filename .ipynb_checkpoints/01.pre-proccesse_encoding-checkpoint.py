import os
from Bio import SeqIO
import numpy as np
import torch
import matplotlib.pyplot as plt

# Paths to the input files
fasta_file = '/sata/wzd/insilico_translation/dataset/GRCh38_latest_rna.fna'
gbff_file = '/sata/wzd/insilico_translation/dataset/GRCh38_latest_rna.gbff'
output_dir = 'encoded_data_start3_code1_end2'
log_file = '/sata/wzd/insilico_translation/script/pre-processe_start_end.log'

logf = open(log_file,'w')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to parse FASTA file and retrieve sequences
def parse_fasta(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id.split('.')[0]] = str(record.seq)
    return sequences

# Function to parse GBFF file and retrieve CDS annotations
def parse_gbff(gbff_file):
    cds_annotations = {}
    for record in SeqIO.parse(gbff_file, "genbank"):
        for feature in record.features:
            if feature.type == "CDS":
                cds_annotations[record.id.split('.')[0]] = (int(feature.location.start),int(feature.location.end))
    return cds_annotations

# One-hot encoding for sequences
def one_hot_encode(seq, max_len):
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'U': [0, 0, 0, 1]}
    one_hot = np.zeros((max_len, 4))  # Initialize a matrix of zeros (max_len x 4)
    for i, nucleotide in enumerate(seq):
        if i < max_len:
            one_hot[i] = encoding.get(nucleotide, [0, 0, 0, 0])  # Default to [0, 0, 0, 0] if nucleotide is unknown
    
    return one_hot

# Function to create labels for CDS regions
def create_labels(seq, cds_regions, max_len):
    labels = np.zeros(max_len)
    start, end = cds_regions
    labels[start:start+3] = 3  # Label start codon
    labels[end-3:end] = 2  # Label stop codon
    labels[start+3: end-3] = 1 # Label coding region
    TIS,TTS = seq[start:start+3],seq[end-3:end] 
    if len(labels[0:end] == seq[0:end]):
        Flag = 1
    else:
        Flag = 0
    return labels,TIS,TTS, Flag

# Parse the files
sequences = parse_fasta(fasta_file)
cds_annotations = parse_gbff(gbff_file)

# Calculate length distribution of all sequences
lengths = [len(seq) for seq in sequences.values()]
upper_quartile = np.percentile(lengths, 75)

# Plot length distribution
plt.hist(lengths, bins=50)
plt.axvline(x=upper_quartile, color='r', linestyle='--')
plt.xlabel('Length of mRNA sequences')
plt.ylabel('Frequency')
plt.title('Distribution of mRNA Sequence Lengths')
plt.savefig('Distribution of mRNA Sequence Lengths.pdf')

# Set the max length based on the upper quartile
max_len = int(upper_quartile)
print(f'Set max_len to {max_len} based on upper quartile of length distribution.')
logf.write(f'Set max_len to {max_len} based on upper quartile of length distribution.'+'\n')

# Process sequences within the determined length
for seq_id in sequences:
    sequence = sequences[seq_id]
    if len(sequence) > max_len:
        logf.write(f'{seq_id} skipped as longer than the cutoff'+'\n')
        continue  # Skip sequences longer than the cutoff
    else:
        if seq_id in cds_annotations:
            cds_regions = cds_annotations[seq_id]

            # Encode the sequence and create labels
            encoded_sequence = one_hot_encode(sequence, max_len)
            labels,TIS,TTS,Flag = create_labels(sequence, cds_regions, max_len)

            # Convert to PyTorch tensors
            X = torch.tensor(encoded_sequence, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32)

            # Save the tensors to disk
            torch.save(X, os.path.join(output_dir, f'{seq_id}_encoded.pt'))
            torch.save(y, os.path.join(output_dir, f'{seq_id}_labels.pt'))

            logf.write(f'{seq_id} Processed and saved, start:{cds_regions[0]} {TIS},end:{cds_regions[1]} {TTS}  {Flag}'+'\n')
        else:
            logf.write(f'{seq_id} skipped as no genebank annotation'+'\n')

print("All sequences processed and saved"+'\n')
logf.write("All sequences processed and saved.")
logf.close()
