python  /home/jovyan/work/insilico_translation/script/01.Encoding.py --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna \
                      --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff \
                      --output_dir /home/jovyan/work/insilico_translation/encode_base_type1_maxlen95_trainratio80 \
                      --log_file  /home/jovyan/work/insilico_translation/encode.log \
                      --encoding_type base \
                      --label_type type1 \
                      --max_len 0.95 \
                      --train_ratio 0.8 \
                      --gpu 0 \
                      --seed 42
