import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from collections import defaultdict
import os
import random
from Bio import SeqIO
import csv

class TranscriptAnalyzer:
    def __init__(self, predictions_file, fasta_file, prefix):
        """初始化分析器"""
        # 读取预测结果
        with open(predictions_file, 'rb') as f:
            raw_data = pickle.load(f)
            self.data = {item['transcript_id']: {
                'predictions': item['predictions'],
                'true_labels': item['true_labels'],
                'length': item['length'],
                'is_match': item['is_match']
            } for item in raw_data}
            
        # 读取序列数据
        self.sequences = {}
        for record in SeqIO.parse(fasta_file, "fasta"):
            transcript_id = record.id.split('.')[0]
            self.sequences[transcript_id] = str(record.seq)
        self.formatted_sequence_stats = defaultdict(int)
        self.prefix = prefix
        
    def normalize_transcript(self, predictions, length):
        """将转录本标准化到0-1区间"""
        norm_positions = np.linspace(0, 1, length)
        return norm_positions, predictions
        
    def find_continuous_regions(self, predictions, label_type):
        """找到连续的预测区域"""
        regions = []
        current_start = None
        current_length = 0
        
        for i, pred in enumerate(predictions):
            if pred == label_type:
                if current_start is None:
                    current_start = i
                current_length += 1
            else:
                if current_length > 0:
                    regions.append({
                        'start': current_start,
                        'length': current_length,
                        'rel_start': current_start / len(predictions)
                    })
                current_start = None
                current_length = 0
                
        if current_length > 0:
            regions.append({
                'start': current_start,
                'length': current_length,
                'rel_start': current_start / len(predictions)
            })
            
        return regions

    def analyze_transcript(self, transcript_id):
        """分析单个转录本"""
        transcript_data = self.data[transcript_id]
        predictions = transcript_data['predictions']
        length = transcript_data['length']
        
        # 找到TIS和TTS区域
        tis_regions = self.find_continuous_regions(predictions, 0)
        tts_regions = self.find_continuous_regions(predictions, 1)
        
        return {
            'tis_regions': tis_regions,
            'tts_regions': tts_regions,
            'normalized_predictions': self.normalize_transcript(predictions, length)
        }

    def plot_transcript_heatmap(self, output_dir, group_size=400):
        """
        绘制按长度分组的转录本预测热图,每组固定group_size个转录本
        """
        # 获取所有转录本长度并排序
        transcript_lengths = [(tid, d['length']) for tid, d in self.data.items()]
        transcript_lengths.sort(key=lambda x: x[1])
        
        # 将转录本分组
        num_groups = (len(transcript_lengths) + group_size - 1) // group_size
        groups = [transcript_lengths[i*group_size : (i+1)*group_size] 
                 for i in range(num_groups)]
        
        # 创建颜色映射 (TIS=红色, TTS=蓝色, non-TIS/TTS=浅灰色, 背景=白色)
        colors = ['red', 'blue', '#EEEEEE', 'white']
        cmap = ListedColormap(colors)
        
        # 为每个组绘制热图
        for idx, group in enumerate(groups):
            group_min = group[0][1]
            group_max = group[-1][1]
            
            # 创建数据矩阵
            data_matrix = np.full((len(group), group_max), 3)  # 背景为白色
            
            # 填充预测数据
            for row, (tid, length) in enumerate(group):
                predictions = self.data[tid]['predictions']
                data_matrix[row, :length] = predictions
            
            # 绘制热图
            plt.figure(figsize=(15, 10))
            
            # 绘制热图并保存返回的图像对象
            im = plt.imshow(data_matrix, aspect='auto', cmap=cmap, interpolation='none')
            
            # 设置标题和标签
            plt.title(f'{self.prefix} Length {group_min}-{group_max}bp (n={len(group)})')
            plt.ylabel('Transcripts')
            plt.xlabel('Position (bp)')

            # 修改后的 colorbar 创建方式
            cbar = plt.colorbar(im, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['TIS', 'TTS', 'non-TIS/TTS'])  # 设置标签
            plt.yticks([])
            
            # 保存图片
            output_file = os.path.join(output_dir, 
                         f"{self.prefix}_heatmap_length{group_min}-{group_max}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved heatmap to {output_file}")

    def plot_density_distribution(self, output_dir):
        """绘制TIS/TTS位置分布密度图"""
        plt.figure(figsize=(12, 6))
        
        # 收集TIS和TTS位置
        tis_positions = []
        tts_positions = []
        
        for transcript_id in self.data:
            analysis = self.analyze_transcript(transcript_id)
            tis_positions.extend([r['rel_start'] for r in analysis['tis_regions']])
            tts_positions.extend([r['rel_start'] for r in analysis['tts_regions']])
        
        # 绘制密度图
        sns.kdeplot(data=tis_positions, color='red', label='TIS')
        sns.kdeplot(data=tts_positions, color='blue', label='TTS')
        
        plt.xlabel('Relative Position in Transcript')
        plt.ylabel('Density')
        plt.title(f'{self.prefix} TIS/TTS Position Distribution')
        plt.legend()
        
        output_file = os.path.join(output_dir, f"{self.prefix}_position_density.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved density plot to {output_file}")

    def plot_continuity_stats(self, output_dir):
        """绘制连续区域长度分布"""
        tis_lengths = []
        tts_lengths = []
        
        for transcript_id in self.data:
            analysis = self.analyze_transcript(transcript_id)
            tis_lengths.extend([r['length'] for r in analysis['tis_regions']])
            tts_lengths.extend([r['length'] for r in analysis['tts_regions']])
        
        plt.figure(figsize=(12, 6))
        plt.hist([tis_lengths, tts_lengths], 
                label=['TIS', 'TTS'], 
                bins=30, 
                alpha=0.7)
        plt.xlabel('Continuous Region Length (bp)')
        plt.ylabel('Frequency')
        plt.title(f'{self.prefix} Distribution of Continuous TIS/TTS Regions')
        plt.legend()
        
        output_file = os.path.join(output_dir, f"{self.prefix}_continuity_distribution.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved continuity distribution to {output_file}")

    def generate_statistics(self):
        """生成统计摘要"""
        stats = {
            'total_transcripts': len(self.data),
            'matching_transcripts': sum(1 for d in self.data.values() if d['is_match']),
            'tis_stats': defaultdict(float),
            'tts_stats': defaultdict(float),
            'transcript_types': defaultdict(int)
        }
        
        # 统计每种转录本类型的数量
        for tid in self.data:
            trans_type = tid.split('_')[0]
            stats['transcript_types'][trans_type] += 1
        
        # TIS和TTS统计
        for transcript_id in self.data:
            analysis = self.analyze_transcript(transcript_id)
            
            # TIS统计
            tis_regions = analysis['tis_regions']
            stats['tis_stats']['total_regions'] += len(tis_regions)
            if tis_regions:
                stats['tis_stats']['avg_length'] += sum(r['length'] for r in tis_regions)
                stats['tis_stats']['max_length'] = max(
                    stats['tis_stats']['max_length'],
                    max(r['length'] for r in tis_regions)
                )
            
            # TTS统计
            tts_regions = analysis['tts_regions']
            stats['tts_stats']['total_regions'] += len(tts_regions)
            if tts_regions:
                stats['tts_stats']['avg_length'] += sum(r['length'] for r in tts_regions)
                stats['tts_stats']['max_length'] = max(
                    stats['tts_stats']['max_length'],
                    max(r['length'] for r in tts_regions)
                )
        
        # 计算平均值
        if stats['tis_stats']['total_regions'] > 0:
            stats['tis_stats']['avg_length'] /= stats['tis_stats']['total_regions']
        if stats['tts_stats']['total_regions'] > 0:
            stats['tts_stats']['avg_length'] /= stats['tts_stats']['total_regions']
            
        return stats

    def generate_formatted_sequences_report_(self, output_dir):
        """
        生成格式化序列的HTML报告
        """
        html_content = """
        <html>
        <head>
            <style>
                table { border-collapse: collapse; width: 100%; }
                th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .sequence { font-family: monospace; }
            </style>
        </head>
        <body>
            <h2>Formatted Sequence Report</h2>
            <table>
                <tr>
                    <th>Transcript ID</th>
                    <th>Formatted Sequence</th>
                </tr>
        """
        
        # 添加每个转录本的格式化序列
        for transcript_id in sorted(self.data.keys()):
            html_content += self.format_sequence_with_predictions(transcript_id)
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # 保存HTML报告
        output_file = os.path.join(output_dir, f"{self.prefix}_formatted_sequences.html")
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        print(f"Saved formatted sequences report to {output_file}")

    def format_sequence_with_predictions(self, transcript_id):
        """
        根据预测结果格式化序列显示
        返回 transcript_id, formatted_seq, formatted_seq_with_pos 的元组
        """
        if transcript_id not in self.data or transcript_id not in self.sequences:
            return "", "", ""
            
        predictions = self.data[transcript_id]['predictions']
        sequence = self.sequences[transcript_id]
        formatted_parts = []
        formatted_parts_with_pos = []
        i = 0
        
        while i < len(predictions):
            # 找到当前位置的预测值
            current_pred = predictions[i]
            
            if current_pred in [0, 1]:  # TIS或TTS
                # 寻找连续的0或1区域
                start = i
                while i < len(predictions) and predictions[i] == current_pred:
                    i += 1
                    
                # 格式化该区域
                color = 'red' if current_pred == 0 else 'blue'
                bases = sequence[start:i]
                
                # 不带位置的格式化
                formatted_parts.append(f'<span style="color:{color}">{bases.upper()}</span>')
                
                # 带位置的格式化
                formatted_parts_with_pos.append(
                    f'<span style="color:{color}">{bases.upper()}<sub>{start+1}</sub></span>'
                )
                
            else:  # 预测值为2的区域
                # 寻找连续的2区域
                start = i
                while i < len(predictions) and predictions[i] == 2:
                    i += 1
                
                # 判断前后的预测值
                prev_pred = predictions[start-1] if start > 0 else None
                next_pred = predictions[i] if i < len(predictions) else None
                gap_length = i - start
                
                # 根据规则处理间隔区域
                if gap_length < 4:
                    bases = sequence[start:i]
                    formatted_str = f'<span style="color:gray">{bases.lower()}</span>'
                    # 两个版本使用相同的格式
                    formatted_parts.append(formatted_str)
                    formatted_parts_with_pos.append(formatted_str)
                else:
                    if (prev_pred == 0 and next_pred == 1) or (prev_pred == 1 and next_pred == 0):
                        symbol = '★'
                    else:
                        symbol = '—'
                    formatted_str = f'<span style="color:gray">{symbol}</span>'
                    # 两个版本使用相同的格式
                    formatted_parts.append(formatted_str)
                    formatted_parts_with_pos.append(formatted_str)
        
        # 返回所有需要的格式化结果
        return {
            'transcript_id': transcript_id,
            'formatted_seq': ''.join(formatted_parts),
            'formatted_seq_with_pos': ''.join(formatted_parts_with_pos)
        }
    
    def collect_pattern_statistics(self):
        """
        收集所有格式化序列的统计信息
        """
        pattern_stats = defaultdict(int)
        total_count = 0
        
        for transcript_id in self.data.keys():
            result = self.format_sequence_with_predictions(transcript_id)
            pattern = result['formatted_seq']
            pattern_stats[pattern] += 1
            total_count += 1
        
        # 转换为排序后的列表
        sorted_stats = [(pattern, count, count/total_count*100) 
                       for pattern, count in pattern_stats.items()]
        sorted_stats.sort(key=lambda x: x[1], reverse=True)
        
        return sorted_stats, total_count

    def generate_formatted_sequences_report(self, output_dir):
        """
        生成格式化序列报告和统计报告
        """
        # 1. 生成主要的序列报告
        html_content = """
        <html>
        <head>
            <style>
                table { border-collapse: collapse; width: 100%; }
                th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .sequence { font-family: monospace; }
            </style>
        </head>
        <body>
            <h2>Formatted Sequence Report</h2>
            <table>
                <tr>
                    <th>Transcript ID</th>
                    <th>Formatted Sequence</th>
                    <th>Formatted Sequence with Position</th>
                </tr>
        """
        
        # 添加每个转录本的格式化序列
        for transcript_id in sorted(self.data.keys()):
            result = self.format_sequence_with_predictions(transcript_id)
            html_content += f"""
            <tr>
                <td>{result['transcript_id']}</td>
                <td class="sequence">{result['formatted_seq']}</td>
                <td class="sequence">{result['formatted_seq_with_pos']}</td>
            </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # 保存主要报告
        output_file = os.path.join(output_dir, f"{self.prefix}_formatted_sequences.html")
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        # 2. 生成统计报告
        pattern_stats, total_count = self.collect_pattern_statistics()
        
        stats_html = f"""
        <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .sequence {{ font-family: monospace; }}
            </style>
        </head>
        <body>
            <h2>Formatted Sequence Pattern Statistics</h2>
            <p>Total number of transcripts analyzed: {total_count}</p>
            <table>
                <tr>
                    <th>Formatted Pattern</th>
                    <th>Count</th>
                    <th>Frequency</th>
                </tr>
        """
        
        for pattern, count, percentage in pattern_stats:
            stats_html += f"""
            <tr>
                <td class="sequence">{pattern}</td>
                <td>{count}</td>
                <td>{percentage:.4f}%</td>
            </tr>
            """
        
        stats_html += """
            </table>
        </body>
        </html>
        """
        
        # 保存统计报告
        stats_file = os.path.join(output_dir, f"{self.prefix}_formatted_sequences_stats.html")
        with open(stats_file, 'w') as f:
            f.write(stats_html)
        
        print(f"Saved formatted sequences report to {output_file}")
        print(f"Saved pattern statistics to {stats_file}")

def main(args):
    """主函数"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化分析器
    analyzer = TranscriptAnalyzer(args.predictions_file, args.fasta_file, args.prefix)
    
    # 生成可视化
    analyzer.plot_transcript_heatmap(args.output_dir)
    analyzer.plot_density_distribution(args.output_dir)
    analyzer.plot_continuity_stats(args.output_dir)
    analyzer.generate_formatted_sequences_report(args.output_dir)
    #analyzer.generate_formatted_sequence_stats(args.output_dir)
    
    # 生成统计报告
    stats = analyzer.generate_statistics()
    
    # 保存统计结果
    report_file = os.path.join(args.output_dir, f"{args.prefix}_statistics_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"Prediction Analysis Report - {args.prefix}\n")
        f.write("=====================================\n\n")
        
        f.write("General Statistics:\n")
        f.write("-----------------\n")
        f.write(f"Total transcripts analyzed: {stats['total_transcripts']}\n")
        f.write(f"Perfectly matching transcripts: {stats['matching_transcripts']} "
                f"({stats['matching_transcripts']/stats['total_transcripts']*100:.2f}%)\n\n")
        
        f.write("Transcript Type Distribution:\n")
        f.write("--------------------------\n")
        for t_type, count in stats['transcript_types'].items():
            f.write(f"{t_type}: {count} ({count/stats['total_transcripts']*100:.2f}%)\n")
        f.write("\n")
        
        f.write("TIS Statistics:\n")
        f.write("--------------\n")
        f.write(f"Total regions: {stats['tis_stats']['total_regions']}\n")
        f.write(f"Average length: {stats['tis_stats']['avg_length']:.2f} bp\n")
        f.write(f"Maximum length: {stats['tis_stats']['max_length']} bp\n\n")
        
        f.write("TTS Statistics:\n")
        f.write("--------------\n")
        f.write(f"Total regions: {stats['tts_stats']['total_regions']}\n")
        f.write(f"Average length: {stats['tts_stats']['avg_length']:.2f} bp\n")
        f.write(f"Maximum length: {stats['tts_stats']['max_length']} bp\n")
        
    print(f"Saved statistics report to {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze prediction results')
    parser.add_argument('--predictions_file', type=str, required=True,
                      help='Path to the predictions pkl file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save analysis results')
    parser.add_argument('--prefix', type=str, required=True,
                       help='Prefix for output files')
    parser.add_argument('--fasta_file', type=str, required=True,
                       help='Path to FASTA file with transcript sequences')
    
    args = parser.parse_args()
    main(args)