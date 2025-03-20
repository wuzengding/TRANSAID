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
#python  /home/jovyan/work/insilico_translation/script/Encoding.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM_3UTR_shuffle   --log_file  /home/jovyan/work/insilico_translation/encode20240830.log   --encoding_type one_hot     --label_type type3 --max_len 0.9995  --train_ratio 0.8  --gpu 0 --utr_shuffle 3  --delete_bases 0 --seed 42

####  generate training dataset 20240912
#python  /home/jovyan/work/insilico_translation/script/Encoding.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM_3DinCDS   --log_file  /home/jovyan/work/insilico_translation/encode20240830.log   --encoding_type one_hot   --label_type type3 --max_len 0.9995  --train_ratio 0.8  --gpu 0 --utr_shuffle 'none'  --delete_bases 3 --seed 42

####  generate training dataset 20240912
#python  /home/jovyan/work/insilico_translation/script/Encoding.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM_3INSinCDS   --log_file  /home/jovyan/work/insilico_translation/encode20240830.log   --encoding_type one_hot   --label_type type3 --max_len 0.9995  --train_ratio 0.8  --gpu 0 --utr_shuffle 'none'  --insert_bases 3 --seed 42

####  generate training dataset 20240912 for mouse  
#/home/jovyan/work/insilico_translation/script/Encoding.py    --fasta_file /home/jovyan/work/insilico_translation/dataset/GCF_000001635.27_GRCm39_rna.fna  --gbff_file /home/jovyan/work/insilico_translation/dataset/GCF_000001635.27_GRCm39_rna.gbff  --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen27308_ratio60_NM_mouse  --log_file  /home/jovyan/work/insilico_translation/encode20240830.log  --encoding_type one_hot  --label_type type3  --max_len 27308   --train_ratio 0.6  --gpu 0 --utr_shuffle none  --seed 42

####  generate training dataset 20240912 for D.rerio
#python  /home/jovyan/work/insilico_translation/script/Encoding.py  --fasta_file /home/jovyan/work/insilico_translation/dataset/GCF_000002035.6_GRCz11_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GCF_000002035.6_GRCz11_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen27308_ratio60_NM_Drerio   --log_file  /home/jovyan/work/insilico_translation/encode20240830.log  --encoding_type one_hot    --label_type type3   --max_len 27308  --train_ratio 0.6  --gpu 0  --utr_shuffle none  --seed 42

####  generate training dataset 20240912 for Saccharomycodes ludwigii
#python  /home/jovyan/work/insilico_translation/script/Encoding.py  --fasta_file /home/jovyan/work/insilico_translation/dataset/fungi.14.rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/fungi.14.rna.gbff   --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen27308_ratio90_NM_Sludwigii  --log_file  /home/jovyan/work/insilico_translation/encode20240830.log  --encoding_type one_hot    --label_type type3   --max_len 27308  --train_ratio 0.9  --gpu 0  --utr_shuffle none  --seed 42

####  generate training dataset 20241202 with embedding code for human
#python  /home/jovyan/work/insilico_translation/script/Encoding2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen75_ratio80_NM  --log_file  /home/jovyan/work/insilico_translation/encode20241202.log   --encoding_type base   --label_type type3 --max_len 0.75  --train_ratio 0.8  --gpu 1 --utr_shuffle 'none'  --seed 42

#### generate training dataset 20241203 with embedding code and structure for human
#python  /home/jovyan/work/insilico_translation/script/Encoding_structure.py  --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_struct_type3_maxlen75_ratio80_NM  --log_file  /home/jovyan/work/insilico_translation/encode20241202.log   --encoding_type base   --label_type type3  --encode_structure --structure_method vienna --max_len 0.75  --train_ratio 0.8  --gpu 1 --utr_shuffle 'none'  --seed 42

####  generate training dataset 20241204 for mouse  
#python /home/jovyan/work/insilico_translation/script/Encoding2.py    --fasta_file /home/jovyan/work/insilico_translation/dataset/GCF_000001635.27_GRCm39_rna.fna  --gbff_file /home/jovyan/work/insilico_translation/dataset/GCF_000001635.27_GRCm39_rna.gbff  --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen5049_ratio60_NM_mouse  --log_file  /home/jovyan/work/insilico_translation/encode20240830.log  --encoding_type base  --label_type type3  --max_len 5049   --train_ratio 0.6  --gpu 0 --utr_shuffle none  --seed 42

#### generate training dataset 20241205 with embedding code for human  NR
#python  /home/jovyan/work/insilico_translation/script/Encoding_structure.py  --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen5049_ratio80_NM_NR  --log_file  /home/jovyan/work/insilico_translation/encode20241202.log   --encoding_type base   --label_type type3  --max_len 5049  --train_ratio 0.8  --gpu 1 --utr_shuffle 'none'  --seed 42  --transcript_types NM,NR

#### generate training dataset 20241210 with embedding code for human  NR
#python  /home/jovyan/work/insilico_translation/script/Encoding_structure.py  --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR  --log_file  /home/jovyan/work/insilico_translation/encode20241202.log   --encoding_type base   --label_type type3  --max_len 0.9995  --train_ratio 0.8  --gpu 0 --utr_shuffle 'none'  --seed 42  --transcript_types NM,NR

#### generate training dataset 20241210 with embedding code for human  NR
#python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py  --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR2  --log_file  /home/jovyan/work/insilico_translation/encode20241202.log   --encoding_type base   --label_type type3  --max_len 0.9995  --train_ratio 0.8  --gpu 0 --utr_shuffle 'none'  --seed 42  --transcript_types NM,NR

#### generate training dataset 20241210 with embedding code for human  NR
#python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py  --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NM_NR4  --log_file  /home/jovyan/work/insilico_translation/encode20241202.log   --encoding_type base   --label_type type3  --max_len 0.9995  --train_ratio 0.8  --gpu 0 --utr_shuffle 'none'  --seed 42  --transcript_types NM,NR


if false
then
####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_shuffle   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_shuffle/encode20240830.log   --encoding_type base  --label_type type3 --max_len 0.9995  --train_ratio 0.8  --utr_shuffle 3  --transcript_types NM,NR  --seed 42 --gpu 0

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_shuffle   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_shuffle/encode20240830.log   --encoding_type base  --label_type type3 --max_len 0.9995  --train_ratio 0.8  --utr_shuffle 5  --transcript_types NM,NR  --seed 42 --gpu 0

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_bothUTR_shuffle   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_bothUTR_shuffle/encode20240830.log   --encoding_type base  --label_type type3 --max_len 0.9995  --train_ratio 0.8  --utr_shuffle both --transcript_types NM,NR  --seed 42 --gpu 0

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_shuffle   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_shuffle/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8  --cds_shuffle  --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_1INS   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_1INS/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_insert_bases '3-1' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_2INS   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_2INS/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_insert_bases '3-2' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_3INS   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_3INS/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_insert_bases '3-3' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_1INS   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_1INS/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_insert_bases '5-1' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_2INS   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_2INS/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_insert_bases '5-2' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_3INS   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_3INS/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_insert_bases '5-3' --transcript_types NM,NR  --seed 42  --gpu 0 


####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_1DEL   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_1DEL/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_delete_bases '3-1' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_2DEL   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_2DEL/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_delete_bases '3-2' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_3DEL   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_3UTR_3DEL/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_delete_bases '3-3' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_1DEL   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_1DEL/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_delete_bases '5-1' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_2DEL   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_2DEL/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_delete_bases '5-2' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_3DEL   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_5UTR_3DEL/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --utr_delete_bases '5-3' --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_1INS   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_1INS/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --cds_insert_bases 1 --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_2INS   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_2INS/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --cds_insert_bases 2 --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_3INS   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_3INS/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --cds_insert_bases 3 --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_1DEL   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_1DEL/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --cds_delete_bases 1 --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_2DEL   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_2DEL/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --cds_delete_bases 2 --transcript_types NM,NR  --seed 42  --gpu 0 

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_3DEL   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_3DEL/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8   --cds_delete_bases 3 --transcript_types NM,NR  --seed 42  --gpu 0 
fi

####  generate training dataset 20250114
python  /home/jovyan/work/insilico_translation/script/Encoding_structure2.py   --fasta_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.fna   --gbff_file /home/jovyan/work/insilico_translation/dataset/GRCh38_latest_rna.gbff   --output_dir /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_shuffle_tmp   --log_file  /home/jovyan/work/insilico_translation/embedding_type3_maxlen9995_ratio80_NMNR_CDS_shuffle_tmp/encode20240830.log   --encoding_type base   --label_type type3 --max_len 0.9995  --train_ratio 0.8  --cds_shuffle  --transcript_types NM,NR  --seed 42  --gpu 0 