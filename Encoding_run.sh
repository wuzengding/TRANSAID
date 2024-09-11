####  generate training dataset 20240819
# python  /home/jovyan/work/insilico_translation/script/01.Encoding.py --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna \
#                      --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff \
#                      --output_dir /home/jovyan/work/insilico_translation/encode_base_type1_maxlen95_trainratio80 \
#                      --log_file  /home/jovyan/work/insilico_translation/encode.log \
#                      --encoding_type base \
#                      --label_type type1 \
#                      --max_len 0.95 \
#                      --train_ratio 0.8 \
#                      --gpu 0 \
#                      --seed 42


####  generate training dataset 20240819
#python  /home/jovyan/work/insilico_translation/script/Encoding.py --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna \
#                      --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff \
#                      --output_dir /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80 \
#                      --log_file  /home/jovyan/work/insilico_translation/encode.log \
#                      --encoding_type base \
#                      --label_type type1 \
#                      --max_len 0.75 \
#                      --train_ratio 0.8 \
#                      --gpu 0 \
#                      --seed 42

####  generate training dataset 20240824
#python  /home/jovyan/work/insilico_translation/script/Encoding.py --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna \
#                      --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff \
#                      --output_dir /home/jovyan/work/insilico_translation/encode_onehot_type1_maxlen75_trainratio80 \
#                      --log_file  /home/jovyan/work/insilico_translation/encode20240824.log \
#                      --encoding_type one_hot \
#                      --label_type type1 \
#                      --max_len 0.75 \
#                      --train_ratio 0.8 \
#                      --gpu 0 \
#                      --seed 42

####  generate training dataset 20240826
#python  /home/jovyan/work/insilico_translation/script/Encoding.py --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna \
#                      --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff \
#                      --output_dir /home/jovyan/work/insilico_translation/encoded_onehot_type3_maxlen75_trainratio80 \
#                      --log_file  /home/jovyan/work/insilico_translation/encode20240826.log \
#                      --encoding_type one_hot \
#                      --label_type type3 \
#                      --max_len 0.75 \
#                      --train_ratio 0.8 \
#                      --gpu 0 \
#                      --seed 42
                      
####  generate training dataset 20240826
#python  /home/jovyan/work/insilico_translation/script/Encoding.py \
#                      --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna \
#                      --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff \
#                      --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM \
#                      --log_file  /home/jovyan/work/insilico_translation/encode20240830.log \
#                      --encoding_type one_hot \
#                      --label_type type3 \
#                      --max_len 0.75 \
#                      --train_ratio 0.8 \
#                      --gpu 0 \
#                      --seed 42


####  generate training dataset 20240826
python  /home/jovyan/work/insilico_translation/script/Encoding.py \
                      --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna \
                      --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff \
                      --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM \
                      --log_file  /home/jovyan/work/insilico_translation/encode20240830.log \
                      --encoding_type one_hot \
                      --label_type type3 \
                      --max_len 0.9995 \
                      --train_ratio 0.8 \
                      --gpu 0 \
                      --seed 42
