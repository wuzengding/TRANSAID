## 20241220
#python  /home/jovyan/work/insilico_translation/script/TISTTS_Pair_Analyzer2.py --predictions /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TRANSAID_Embedding_batch4_NM_matching_predictions.pkl --fasta /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna  --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TTSTIS_selection  --min_score 0.0 

## 20241220
#python  /home/jovyan/work/insilico_translation/script/TISTTS_Pair_Analyzer.py --predictions /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TRANSAID_Embedding_batch4_NM_matching_predictions.pkl --fasta /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna  --gbff  /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff --output-dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TTSTIS_selection  --min-score 0.0 --species human --plot

## 20241223
python  /home/jovyan/work/insilico_translation/script/TISTTS_Pair_Analyzer2.py --predictions /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TRANSAID_Embedding_batch4_NM_matching_predictions.pkl --fasta /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna  --output-dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TTSTIS_selection  --min-prob 0.2 --min-cds-len 50  --prefix "TRANSAID_Embedding_batch4_NM_matching"

## 20241223
python  /home/jovyan/work/insilico_translation/script/TISTTS_Pair_Analyzer2.py --predictions /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TRANSAID_Embedding_batch4_NM_non_matching_predictions.pkl --fasta /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna  --output-dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TTSTIS_selection  --min-prob 0.2 --min-cds-len 50  --prefix "TRANSAID_Embedding_batch4_NM_non_matching"

## 20241223
python  /home/jovyan/work/insilico_translation/script/TISTTS_Pair_Analyzer2.py --predictions /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TRANSAID_Embedding_batch4_NR_matching_predictions.pkl --fasta /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna  --output-dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TTSTIS_selection  --min-prob 0.2 --min-cds-len 50  --prefix "TRANSAID_Embedding_batch4_NR_matching"

## 20241223
python  /home/jovyan/work/insilico_translation/script/TISTTS_Pair_Analyzer2.py --predictions /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TRANSAID_Embedding_batch4_NR_non_matching_predictions.pkl --fasta /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna  --output-dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR/TTSTIS_selection  --min-prob 0.2 --min-cds-len 50  --prefix "TRANSAID_Embedding_batch4_NR_non_matching"