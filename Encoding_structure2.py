import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from Bio import SeqIO
import random
import RNA
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def structure_encode(seq, max_len):
    """编码RNA二级结构信息"""
    struct_encoded = np.zeros(max_len, dtype=np.int64)
    fc = RNA.fold_compound(seq)
    struct, mfe = fc.mfe()
    
    for i, char in enumerate(struct):
        if i >= max_len:
            break
        if char == '.':
            struct_encoded[i] = 1  # unpaired
        elif char in '()':
            struct_encoded[i] = 2  # paired
    
    return struct_encoded

def parse_fasta(fasta_file, transcript_types=None):
    """解析FASTA文件并获取序列"""
    sequences = {}
    type_counts = {}
    
    if transcript_types is None:
        transcript_types = ["NM"]
    print("transcript_types", transcript_types)
    for record in SeqIO.parse(fasta_file, "fasta"):
        record_type = record.id.split("_")[0]
        #print("record_type", record_type)
        if record_type in transcript_types:
            #print("record_type", record_type)
            sequences[record.id.split('.')[0]] = str(record.seq)
            type_counts[record_type] = type_counts.get(record_type, 0) + 1
    
    for t_type, count in type_counts.items():
        print(f'Number of {t_type} transcripts: {count}')
        
    return sequences

def parse_gbff(gbff_file, transcript_types=None):
    """解析GBFF文件并获取CDS注释"""
    if transcript_types is None:
        transcript_types = ["NM"]
        
    cds_annotations = {}
    for record in SeqIO.parse(gbff_file, "genbank"):
        trans_type = record.id.split("_")[0]
        trans_id = record.id.split(".")[0]
        if trans_type in ["NM","XM"] and trans_type in transcript_types:
            for feature in record.features:
                if feature.type == "CDS":
                    cds_annotations[trans_id] = (int(feature.location.start), 
                                             int(feature.location.end))
        elif trans_type in ["NR","XR"] and trans_type in transcript_types:
            # 对于NR转录本，按10%-80%-10%划分
            seq_length = len(record.seq)
            utr5_end = int(seq_length * 0.1)
            utr3_start = int(seq_length * 0.9)
            cds_annotations[trans_id] = (utr5_end, utr3_start)
    return cds_annotations

def find_poly_a(sequence, min_length=4):
    """Find polyA sequence at the beginning"""
    count = 0
    for base in sequence:
        if base.upper() == 'A':
            count += 1
        else:
            break
    return sequence[:count] if count >= min_length else ""

def shuffle_sequence_with_poly_a(sequence, is_five_prime=False):
    """Shuffle sequence while preserving polyA if present at 5' end"""
    if not is_five_prime:
        return ''.join(random.sample(sequence, len(sequence)))
        
    poly_a = find_poly_a(sequence)
    if poly_a:
        remaining_seq = sequence[len(poly_a):]
        shuffled_seq = ''.join(random.sample(remaining_seq, len(remaining_seq)))
        return poly_a + shuffled_seq
    else:
        return ''.join(random.sample(sequence, len(sequence)))

def insert_random_bases(sequence, num_bases):
    """Insert random bases at the end of sequence"""
    bases = ['A', 'C', 'G', 'T']
    insert_seq = ''.join(random.choices(bases, k=num_bases))
    return sequence + insert_seq

def delete_bases(sequence, num_bases):
    """Delete bases from the end of sequence"""
    if len(sequence) <= num_bases:
        return sequence
    return sequence[:-num_bases]

def get_utr_regions(cds_start, cds_end, seq_length):
    """计算UTR区域"""
    utr_regions = {}
    if cds_start > 1:
        utr_regions['5'] = (0, cds_start - 1)
    if cds_end < seq_length:
        utr_regions['3'] = (cds_end, seq_length - 1)
    return utr_regions

def modify_sequence_regions(sequence, cds_start, cds_end, 
                          utr_shuffle=None, cds_shuffle=False,
                          utr_insert_bases=None, utr_delete_bases=None,
                          cds_insert_bases=0, cds_delete_bases=0):
    """
    Modified sequence regions with various operations
    
    Args:
        sequence: Original sequence
        cds_start, cds_end: CDS region boundaries
        utr_shuffle: None, '5', '3', or 'both'
        cds_shuffle: Whether to shuffle CDS region
        utr_insert_bases: Format like '3-1', '5-3', 'both-1'
        utr_delete_bases: Similar to utr_insert_bases
        cds_insert_bases: Number of bases to insert in CDS
        cds_delete_bases: Number of bases to delete from CDS
    """
    seq_list = list(sequence)
    seq_length = len(sequence)
    utr_regions = get_utr_regions(cds_start, cds_end, seq_length)
    
    # Process UTR shuffle
    if utr_shuffle:
        if utr_shuffle in ['5', 'both'] and '5' in utr_regions:
            five_utr_start, five_utr_end = utr_regions['5']
            utr_seq = ''.join(seq_list[five_utr_start:five_utr_end + 1])
            shuffled_seq = shuffle_sequence_with_poly_a(utr_seq, is_five_prime=True)
            seq_list[five_utr_start:five_utr_end + 1] = list(shuffled_seq)

        if utr_shuffle in ['3', 'both'] and '3' in utr_regions:
            three_utr_start, three_utr_end = utr_regions['3']
            utr_seq = ''.join(seq_list[three_utr_start:three_utr_end + 1])
            shuffled_seq = shuffle_sequence_with_poly_a(utr_seq, is_five_prime=False)
            seq_list[three_utr_start:three_utr_end + 1] = list(shuffled_seq)
    
    # Process CDS shuffle
    if cds_shuffle:
        cds_seq = ''.join(seq_list[cds_start:cds_end])
        shuffled_cds = ''.join(random.sample(cds_seq, len(cds_seq)))
        seq_list[cds_start:cds_end] = list(shuffled_cds)
    
    # Process UTR insertions
    if utr_insert_bases:
        region, bases = utr_insert_bases.split('-')
        bases = int(bases)
        if region in ['5', 'both'] and '5' in utr_regions:
            five_utr_start, _ = utr_regions['5']
            insert_seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=bases))
            seq_list.insert(five_utr_start, insert_seq)
        if region in ['3', 'both'] and '3' in utr_regions:
            _, three_utr_end = utr_regions['3']
            insert_seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=bases))
            seq_list.insert(three_utr_end, insert_seq)
    
    # Process UTR deletions
    if utr_delete_bases:
        region, bases = utr_delete_bases.split('-')
        bases = int(bases)
        if region in ['5', 'both'] and '5' in utr_regions:
            five_utr_start, _ = utr_regions['5']
            del seq_list[five_utr_start:five_utr_start + bases]
        if region in ['3', 'both'] and '3' in utr_regions:
            _, three_utr_end = utr_regions['3']
            del seq_list[three_utr_end - bases:three_utr_end]
    
    # Process CDS modifications
    if cds_insert_bases > 0:
        insert_pos = random.randint(cds_start + 4, cds_end - 4)
        insert_seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=cds_insert_bases))
        seq_list[insert_pos:insert_pos] = list(insert_seq)
    
    if cds_delete_bases > 0:
        delete_start = random.randint(cds_start + 4, cds_end - 4 - cds_delete_bases)
        del seq_list[delete_start:delete_start + cds_delete_bases]
    
    return ''.join(seq_list)

def one_hot_encode(seq, max_len):
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'U': [0, 0, 0, 1]}
    one_hot = np.zeros((max_len, 4))
    for i, nucleotide in enumerate(seq):
        one_hot[i] = encoding.get(nucleotide, [0, 0, 0, 0])
    return one_hot

def base_encode(seq, max_len):
    encoding = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4}
    encoded_seq = np.zeros(max_len, dtype=np.int64)
    for i, nucleotide in enumerate(seq):
        encoded_seq[i] = encoding.get(nucleotide, 0)
    return encoded_seq

def create_labels(seq, cds_regions, max_len, label_type):
    if label_type == 'type3':
        labels = np.full((max_len, 3), [0, 0, 0], dtype=np.int32)
    else:
        labels = np.full(max_len, -1, dtype=np.int32)
    
    seq_len = len(seq)
    if label_type == 'type3':
        labels[0:seq_len] = [0, 0, 1]
    else:
        labels[0:seq_len] = 0
    
    start, end = cds_regions
    if start != None and end != None:
        if label_type == 'type1':
            labels[start:start+3] = 2
            labels[end-3:end] = 1
        elif label_type == 'type2':
            labels[start:start+3] = 3
            labels[end-3:end] = 2
            labels[start+3:end-3] = 1
        elif label_type == 'type3':
            labels[start:start+3] = [1, 0, 0]
            labels[end-3:end] = [0, 1, 0]
            labels[start+3:end-3] = [0, 0, 1]

    return labels

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

def main(args):
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"validation"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"train_check"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"validation_check"), exist_ok=True)
    print(args.transcript_types)
    sequences = parse_fasta(args.fasta_file, args.transcript_types)
    cds_annotations = parse_gbff(args.gbff_file, args.transcript_types)

    lengths = [len(seq) for seq in sequences.values()]
    if args.max_len <= 1.0:
        max_len = int(np.percentile(lengths, args.max_len * 100))
    else:
        max_len = int(args.max_len)
    
    print(f'Set max_len to {max_len}.')

    print("Processing sequences...")
    encoded_data = {}
    struct_data = {}
    labels_data = {}
    
    sequences_to_process = [(seq_id, sequence) for seq_id, sequence in sequences.items() 
                       if len(sequence) <= max_len and seq_id in cds_annotations]

    for seq_id, sequence in tqdm(sequences_to_process, desc="Encoding sequences"):
        if seq_id in cds_annotations:
            cds_regions = cds_annotations[seq_id]
            is_nr = seq_id.startswith('NR_') or seq_id.startswith('XR_')

            # Apply sequence modifications
            if args.utr_shuffle != 'none' or args.cds_shuffle or args.utr_insert_bases or \
               args.utr_delete_bases or args.cds_insert_bases or args.cds_delete_bases:
                sequence = modify_sequence_regions(
                    sequence=sequence,
                    cds_start=cds_regions[0],
                    cds_end=cds_regions[1],
                    utr_shuffle=args.utr_shuffle,
                    cds_shuffle=args.cds_shuffle,
                    utr_insert_bases=args.utr_insert_bases,
                    utr_delete_bases=args.utr_delete_bases,
                    cds_insert_bases=args.cds_insert_bases,
                    cds_delete_bases=args.cds_delete_bases
                )

            if args.encoding_type == 'one_hot':
                encoded_sequence = one_hot_encode(sequence, max_len)
                dtype = torch.float32
            elif args.encoding_type == 'base':
                encoded_sequence = base_encode(sequence, max_len)
                dtype = torch.long
            
            if args.encode_structure:
                struct_encoded = structure_encode(sequence, max_len)
                struct_data[seq_id] = torch.tensor(struct_encoded, dtype=torch.long)
            
            labels = create_labels(sequence, cds_regions, max_len, args.label_type)
            encoded_data[seq_id] = torch.tensor(encoded_sequence, dtype=dtype)
            labels_data[seq_id] = torch.tensor(labels, dtype=torch.long)
        else:
            print(f'{seq_id} not in annotation files')

    train_data, val_data, train_labels, val_labels = split_data(
        encoded_data, labels_data, args.train_ratio, args.seed)
    
    if args.encode_structure:
        train_struct, val_struct, _, _ = split_data(
            struct_data, labels_data, args.train_ratio, args.seed)

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
    parser.add_argument('--encoding_type', type=str, choices=['one_hot', 'base'], required=True, 
                       help='Encoding type: "one_hot" or "base"')
    parser.add_argument('--label_type', type=str, choices=['type1', 'type2', 'type3'], required=True, 
                       help='Label type to create: "type1", "type2", or "type3"')
    parser.add_argument('--transcript_types',  type=lambda s: s.split(','),
                       default=['NM'],
                       help='List of transcript types to include (e.g., NM XM NR XR)')
    parser.add_argument('--max_len', type=float, required=True, 
                       help='Absolute length value or a decimal (0-1) indicating percentile')
    parser.add_argument('--train_ratio', type=float, required=True, 
                       help='Ratio of training data (e.g., 0.83 for a 5:1 train:validation split)')
    parser.add_argument('--gpu', type=int, required=True, choices=[0, 1], 
                       help='GPU device to use: 0 or 1')

    # 新增参数
    parser.add_argument('--utr_shuffle', type=str, 
                       choices=['5', '3', 'both', 'none'], 
                       default='none',
                       help="Shuffle UTR sequences: '5' for 5' UTR, '3' for 3' UTR, 'both' for both, or 'none'")
    parser.add_argument('--cds_shuffle', action='store_true',
                       help="Shuffle CDS region")
    parser.add_argument('--utr_insert_bases', type=str,
                       help="Insert random bases in UTR (format: '3-1', '5-3', 'both-1')")
    parser.add_argument('--utr_delete_bases', type=str,
                       help="Delete bases from UTR (format: '3-1', '5-3', 'both-1')")
    parser.add_argument('--cds_insert_bases', type=int, default=0,
                       help="Number of random bases to insert in CDS region")
    parser.add_argument('--cds_delete_bases', type=int, default=0,
                       help="Number of bases to delete from CDS region")
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')

    args = parser.parse_args()
    
    # 验证参数
    if args.utr_insert_bases and args.utr_delete_bases:
        parser.error("--utr_insert_bases and --utr_delete_bases cannot be used together")
    
    # 验证utr_insert_bases和utr_delete_bases的格式
    for param in [args.utr_insert_bases, args.utr_delete_bases]:
        if param:
            try:
                position, num = param.split('-')
                if position not in ['3', '5', 'both']:
                    parser.error(f"Invalid position in {param}, must be '3', '5', or 'both'")
                if not num.isdigit():
                    parser.error(f"Invalid number in {param}, must be a positive integer")
            except ValueError:
                parser.error(f"Invalid format for {param}, must be 'position-number' (e.g., '3-1')")
    
    # 验证cds相关参数
    if args.cds_insert_bases < 0:
        parser.error("--cds_insert_bases must be non-negative")
    if args.cds_delete_bases < 0:
        parser.error("--cds_delete_bases must be non-negative")
    if args.cds_insert_bases > 0 and args.cds_delete_bases > 0:
        parser.error("Cannot specify both --cds_insert_bases and --cds_delete_bases")

    main(args)