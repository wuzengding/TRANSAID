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
#    --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna \
#    --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff \
#    --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM \
#    --log_file  /home/jovyan/work/insilico_translation/encode20240830.log \
#    --encoding_type one_hot \
#    --label_type type3 \
#    --max_len 0.75 \
#    --train_ratio 0.8 \
#    --gpu 0 \
#    --seed 42


####  generate training dataset 20240826
#python  /home/jovyan/work/insilico_translation/script/Encoding.py \
#    --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna \
#    --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff \
#    --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM \
#    --log_file  /home/jovyan/work/insilico_translation/encode20240830.log \
#    --encoding_type one_hot \
#    --label_type type3 \
#    --max_len 0.9995 \
#    --train_ratio 0.8 \
#    --gpu 0 \
#    --seed 42

####  generate training dataset 20240911
#python  /home/jovyan/work/insilico_translation/script/Encoding.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM_3UTR__shuffle   --log_file  /home/jovyan/work/insilico_translation/encode20240830.log   --encoding_type one_hot     --label_type type3 --max_len 0.9995  --train_ratio 0.8  --gpu 0 --utr_shuffle 3  --delete_bases 0 --seed 42

####  generate training dataset 20240912
#python  /home/jovyan/work/insilico_translation/script/Encoding.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM_3DinCDS   --log_file  /home/jovyan/work/insilico_translation/encode20240830.log   --encoding_type one_hot   --label_type type3 --max_len 0.9995  --train_ratio 0.8  --gpu 0 --utr_shuffle 'none'  --delete_bases 3 --seed 42

####  generate training dataset 20240912
#python  /home/jovyan/work/insilico_translation/script/Encoding.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM_3INSinCDS   --log_file  /home/jovyan/work/insilico_translation/encode20240830.log   --encoding_type one_hot   --label_type type3 --max_len 0.9995  --train_ratio 0.8  --gpu 0 --utr_shuffle 'none'  --insert_bases 3 --seed 42

####  generate training dataset 20240912 for mouse  
#/home/jovyan/work/insilico_translation/script/Encoding.py    --fasta_file /home/jovyan/work/insilico_translation/dataset/GCF_000001635.27_GRCm39_rna.fna  --gbff_file /home/jovyan/work/insilico_translation/dataset/GCF_000001635.27_GRCm39_rna.gbff  --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen27308_ratio60_NM_mouse  --log_file  /home/jovyan/work/insilico_translation/encode20240830.log  --encoding_type one_hot  --label_type type3  --max_len 27308   --train_ratio 0.6  --gpu 0 --utr_shuffle none  --seed 42

####  generate training dataset 20240912 for D.rerio
#python  /home/jovyan/work/insilico_translation/script/Encoding.py  --fasta_file /home/jovyan/work/insilico_translation/dataset/GCF_000002035.6_GRCz11_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GCF_000002035.6_GRCz11_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen27308_ratio60_NM_Drerio   --log_file  /home/jovyan/work/insilico_translation/encode20240830.log  --encoding_type one_hot    --label_type type3   --max_len 27308  --train_ratio 0.6  --gpu 0  --utr_shuffle none  --seed 42

####  generate training dataset 20240912 for Saccharomycodes ludwigii
python  /home/jovyan/work/insilico_translation/script/Encoding.py  --fasta_file /home/jovyan/work/insilico_translation/dataset/fungi.14.rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/fungi.14.rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen27308_ratio90_NM_Sludwigii  --log_file  /home/jovyan/work/insilico_translation/encode20240830.log  --encoding_type one_hot    --label_type type3   --max_len 27308  --train_ratio 0.9  --gpu 0  --utr_shuffle none  --seed 42