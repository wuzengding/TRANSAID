import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from Bio import SeqIO
import random
import RNA
from tqdm import tqdm

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def structure_encode(seq, max_len):
    """
    编码RNA二级结构信息。
    
    Args:
        seq (str): RNA序列
        max_len (int): 最大序列长度
    
    Returns:
        np.ndarray: 结构编码数组，0=padding, 1=unpaired, 2=paired
    """
    # 初始化结构编码数组
    struct_encoded = np.zeros(max_len, dtype=np.int64)
    
    # 使用ViennaRNA预测结构
    fc = RNA.fold_compound(seq)
    struct, mfe = fc.mfe()
    
    # 编码结构信息
    for i, char in enumerate(struct):
        if i >= max_len:
            break
        if char == '.':
            struct_encoded[i] = 1  # unpaired
        elif char in '()':
            struct_encoded[i] = 2  # paired
    
    return struct_encoded

# 解析FASTA文件并获取序列
def parse_fasta(fasta_file, transcript_types=None):
    """
    Parse FASTA file and get sequences for specified transcript types.
    
    Args:
        fasta_file (str): Path to FASTA file
        transcript_types (list): List of transcript types (e.g., ["NM", "XM"])
    """
    sequences = {}
    type_counts = {}
    
    if transcript_types is None:
        transcript_types = ["NM"]  # Default to NM if not specified
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        record_type = record.id.split("_")[0]
        if record_type in transcript_types:
            sequences[record.id.split('.')[0]] = str(record.seq)
            type_counts[record_type] = type_counts.get(record_type, 0) + 1
    
    # Print statistics
    for t_type, count in type_counts.items():
        print(f'Number of {t_type} transcripts: {count}')
        
    return sequences

# 解析GBFF文件并获取CDS注释
def parse_gbff(gbff_file):
    cds_annotations = {}
    for record in SeqIO.parse(gbff_file, "genbank"):
        for feature in record.features:
            if feature.type == "CDS":
                cds_annotations[record.id.split('.')[0]] = (int(feature.location.start), int(feature.location.end))
    return cds_annotations

# 计算UTR区域
def get_utr_regions(cds_start, cds_end, seq_length):
    utr_regions = {}
    if cds_start > 1:
        utr_regions['5'] = (0, cds_start - 1)  # Adjusted for zero-based indexing
    if cds_end < seq_length:
        utr_regions['3'] = (cds_end, seq_length - 1)
    return utr_regions

# Shuffle UTR regions based on the determined UTR regions from CDS
def shuffle_utr(seq, cds_start, cds_end, utr_type='both'):
    seq_list = list(seq)
    seq_length = len(seq)
    utr_regions = get_utr_regions(cds_start, cds_end, seq_length)

    if utr_type in ['5', 'both'] and '5' in utr_regions:
        five_utr_start, five_utr_end = utr_regions['5']
        shuffled_five_utr = random.sample(seq_list[five_utr_start:five_utr_end + 1], five_utr_end - five_utr_start + 1)
        seq_list[five_utr_start:five_utr_end + 1] = shuffled_five_utr

    if utr_type in ['3', 'both'] and '3' in utr_regions:
        three_utr_start, three_utr_end = utr_regions['3']
        shuffled_three_utr = random.sample(seq_list[three_utr_start:three_utr_end + 1], three_utr_end - three_utr_start + 1)
        seq_list[three_utr_start:three_utr_end + 1] = shuffled_three_utr

    return ''.join(seq_list)


# Function to delete bases in the CDS region
def delete_bases_in_cds(seq, cds_start, cds_end, delete_bases):
    """
    Deletes a given number of bases randomly from the CDS region.
    Adjusts the CDS end position accordingly.
    """
    #print(f"delete_bases number is : {delete_bases}")
    seq_list = list(seq)
    cds_length = cds_end - cds_start + 1

    # Ensure we don't delete more bases than available in the CDS
    if delete_bases >= cds_length:
        raise ValueError(f"Cannot delete {delete_bases} bases; CDS length is only {cds_length}.")

    # Randomly select positions within the CDS to delete bases
    deletion_start = random.randint(cds_start+ 4, cds_end  -4 - delete_bases)

    del seq_list[deletion_start:deletion_start + delete_bases]

    # Adjust the CDS end position
    adjusted_cds_end = cds_end - delete_bases

    return ''.join(seq_list), (cds_start, adjusted_cds_end)

# Function to insert a random base sequence into the CDS region
def insert_bases_in_cds(seq, cds_start, cds_end, insert_length):
    """
    Inserts a random sequence of a given length into the CDS region.
    Adjusts the CDS end position accordingly.
    """
    seq_list = list(seq)
    
    # Generate a random sequence of the given length
    bases = ['A', 'C', 'G', 'T']
    insert_seq = ''.join(random.choices(bases, k=insert_length))
    
    # Randomly select a position within the CDS to insert the sequence
    insertion_position = random.randint(cds_start + 4, cds_end - 4)
    
    # Insert the sequence
    seq_list = seq_list[:insertion_position] + list(insert_seq) + seq_list[insertion_position:]
    
    # Adjust the CDS end position
    adjusted_cds_end = cds_end + insert_length
    
    return ''.join(seq_list), (cds_start, adjusted_cds_end)

# one-hot编码
def one_hot_encode(seq, max_len):
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'U': [0, 0, 0, 1]}
    one_hot = np.zeros((max_len, 4))  # Initialize a matrix of zeros (max_len x 4)
    for i, nucleotide in enumerate(seq):
        one_hot[i] = encoding.get(nucleotide, [0, 0, 0, 0])  # Default to [0, 0, 0, 0] if nucleotide is unknown
    return one_hot


# base 编码
def base_encode(seq, max_len):
    encoding = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4}  # Map each base to an integer
    encoded_seq = np.zeros(max_len, dtype=np.int64)
    for i, nucleotide in enumerate(seq):
        encoded_seq[i] = encoding.get(nucleotide, 0)  # 0 for unknown nucleotides
    return encoded_seq

# 创建标签
def create_labels(seq, cds_regions, max_len, label_type):
    if label_type == 'type3':
        labels = np.full((max_len, 3), [0, 0, 0], dtype=np.int32)  # Initialize with padding label [0,0,0]
    else:
        labels = np.full(max_len, -1, dtype=np.int32)
    
    seq_len = len(seq)
    if label_type == 'type3':
        labels[0:seq_len] = [0, 0, 1]  # Label non-TIS/TTS region
    else:
        labels[0:seq_len] = 0  # Label 5' & 3' UTR region
    
    start, end = cds_regions
    if label_type == 'type1':
        labels[start:start+3] = 2  # Label start codon
        labels[end-3:end] = 1  # Label stop codon
    elif label_type == 'type2':
        labels[start:start+3] = 3  # Label start codon
        labels[end-3:end] = 2  # Label stop codon
        labels[start+3:end-3] = 1  # Label coding region
    elif label_type == 'type3':
        labels[start:start+3] = [1, 0, 0]  # Label TIS region
        labels[end-3:end] = [0, 1, 0]  # Label TTS region
        labels[start+3:end-3] = [0, 0, 1]  # Label non-TIS/TTS coding region

    TIS, TTS = seq[start:start+3], seq[end-3:end]
    Flag = int(len(labels[0:seq_len]) == len(seq[0:seq_len]))
    return labels, TIS, TTS, Flag

# 随机划分数据集
def split_data(sequences, labels, train_ratio, seed):
    set_seed(seed)
    all_keys = list(sequences.keys())
    train_size = int(len(all_keys) * train_ratio)
    train_keys = all_keys[:train_size]
    val_keys = all_keys[train_size:]

    train_data = {key: sequences[key] for key in train_keys}
    val_data = {key: sequences[key] for key in val_keys}

    train_labels = {key: labels[key] for key in train_keys}
    val_labels = {key: labels[key] for key in val_keys}

    return train_data, val_data, train_labels, val_labels


# 主函数
def main(args):
    set_seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"validation"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"train_check"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"validation_check"), exist_ok=True)

    # 解析FASTA和GBFF文件
    sequences = parse_fasta(args.fasta_file, args.transcript_types)
    cds_annotations = parse_gbff(args.gbff_file)

    # 计算长度分布
    lengths = [len(seq) for seq in sequences.values()]
    if args.max_len <= 1.0:
        max_len = int(np.percentile(lengths, args.max_len * 100))
    else:
        max_len = int(args.max_len)
    
    print(f'Set max_len to {max_len}.')

    '''
    # 绘制长度分布
    plt.hist(lengths, bins=50)
    plt.axvline(x=np.percentile(lengths, 75), color='r', linestyle='--')
    plt.xlabel('Length of mRNA sequences')
    plt.ylabel('Frequency')
    plt.xlim(0, 40000)
    plt.title('Distribution of mRNA Sequence Lengths')
    plt.savefig(os.path.join(args.output_dir,   'Distribution_of_mRNA_Sequence_Lengths.pdf'))
    '''

    # 准备训练和验证数据
    print("Processing sequences...")
    encoded_data = {}
    struct_data = {}  # 新增结构数据字典
    labels_data = {}
    
    # 获取要处理的序列总数，用于进度显示
    sequences_to_process = [(seq_id, sequence) for seq_id, sequence in sequences.items() 
                       if len(sequence) <= max_len and seq_id in cds_annotations]
    # 使用tqdm创建进度条
    for seq_id, sequence in tqdm(sequences_to_process, desc="Encoding sequences"):
        if seq_id in cds_annotations:
            cds_regions = cds_annotations[seq_id]

            # Shuffle UTRs if specified
            if args.utr_shuffle != 'none':
                sequence = shuffle_utr(sequence, cds_regions[0], cds_regions[1], args.utr_shuffle)
            
            # Apply deletion/insertion mutations if specified
            if args.delete_bases > 0:
                sequence, cds_regions = delete_bases_in_cds(sequence, cds_regions[0], cds_regions[1], args.delete_bases)
            if args.insert_bases > 0:
                sequence, cds_regions = insert_bases_in_cds(sequence, cds_regions[0], cds_regions[1], args.insert_bases)

            # 序列编码
            if args.encoding_type == 'one_hot':
                encoded_sequence = one_hot_encode(sequence, max_len)
                dtype = torch.float32
            elif args.encoding_type == 'base':
                encoded_sequence = base_encode(sequence, max_len)
                dtype = torch.long
            
            # 结构编码（如果启用）
            if args.encode_structure:
                struct_encoded = structure_encode(sequence, max_len)
                struct_data[seq_id] = torch.tensor(struct_encoded, dtype=torch.long)
            
            # 创建标签
            labels, TIS, TTS, Flag = create_labels(sequence, cds_regions, max_len, args.label_type)
            encoded_data[seq_id] = torch.tensor(encoded_sequence, dtype=dtype)
            labels_data[seq_id] = torch.tensor(labels, dtype=torch.long)
            
        else:
            print(f'{seq_id} not in annotation files')

    # 划分数据集
    train_data, val_data, train_labels, val_labels = split_data(
        encoded_data, labels_data, args.train_ratio, args.seed)
    
    if args.encode_structure:
        train_struct, val_struct, _, _ = split_data(
            struct_data, labels_data, args.train_ratio, args.seed)

    # 保存数据
    for seq_id, tensor in train_data.items():
        torch.save(tensor, os.path.join(args.output_dir, "train", f'{seq_id}_encoded.pt'))
        if args.encode_structure:
            torch.save(train_struct[seq_id], 
                      os.path.join(args.output_dir, "train", f'{seq_id}_structure.pt'))
    
    for seq_id, tensor in train_labels.items():
        torch.save(tensor, os.path.join(args.output_dir, "train", f'{seq_id}_labels.pt'))
 
    for seq_id, tensor in val_data.items():
        torch.save(tensor, os.path.join(args.output_dir, "validation", f'{seq_id}_encoded.pt'))
        if args.encode_structure:
            torch.save(val_struct[seq_id], 
                       os.path.join(args.output_dir, "validation", f'{seq_id}_structure.pt'))
    
    for seq_id, tensor in val_labels.items():
        torch.save(tensor, os.path.join(args.output_dir, "validation", f'{seq_id}_labels.pt'))

    # 保存检查样本
    if args.encode_structure:
        train_struct = train_struct
        val_struct = val_struct
    else:
        train_struct = None
        val_struct = None
    
    save_check_samples_with_structure(train_data, train_struct, train_labels, 
                                        args.output_dir, "train_check")
    save_check_samples_with_structure(val_data, val_struct, val_labels, 
                                        args.output_dir, "validation_check")

    print("All sequences processed and saved.")

# 保存检查样本
def save_check_samples_with_structure(data, struct_data, labels, output_dir, 
                                        folder_name, sample_size=100):
    check_dir = os.path.join(output_dir, folder_name)
    os.makedirs(check_dir, exist_ok=True)
        
    sample_keys = list(data.keys())[:sample_size]
    for seq_id in sample_keys:
        torch.save(data[seq_id], os.path.join(check_dir, f'{seq_id}_encoded.pt'))
        if args.encode_structure and struct_data != None:
            torch.save(struct_data[seq_id], 
                      os.path.join(check_dir, f'{seq_id}_structure.pt'))
        torch.save(labels[seq_id], os.path.join(check_dir, f'{seq_id}_labels.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and encode mRNA sequences.')
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the FASTA file')
    parser.add_argument('--gbff_file', type=str, required=True, help='Path to the GBFF file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save encoded data')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    parser.add_argument('--encode_structure', action='store_true',help='Whether to encode RNA secondary structure')
    parser.add_argument('--structure_method', type=str,choices=['vienna'], default='vienna',
                       help='Method for structure prediction')
    parser.add_argument('--encoding_type', type=str, choices=['one_hot', 'base'], required=True, help='Encoding type: "one_hot" or "base"')
    parser.add_argument('--label_type', type=str, choices=['type1', 'type2', 'type3'], required=True, help='Label type to create: "type1", "type2", or "type3"')
    parser.add_argument('--transcript_types', type=str, nargs='+',
                       default=['NM'],
                       help='List of transcript types to include (e.g., NM XM NR XR)')
    parser.add_argument('--max_len', type=float, required=True, help='Absolute length value or a decimal (0-1) indicating percentile')
    parser.add_argument('--train_ratio', type=float, required=True, help='Ratio of training data (e.g., 0.83 for a 5:1 train:validation split)')
    parser.add_argument('--gpu', type=int, required=True, choices=[0, 1], help='GPU device to use: 0 or 1')
    parser.add_argument('--utr_shuffle', type=str, choices=['5', '3', 'both', 'none'], default='none', help="Shuffle UTR sequences: '5' for 5' UTR, '3' for 3' UTR, 'both' for both, or 'none'")
    parser.add_argument('--delete_bases', type=int, default=0, help="Number of bases to delete in the CDS region")
    parser.add_argument('--insert_bases', type=int, default=0, help="Number of random bases to insert in the CDS region")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)