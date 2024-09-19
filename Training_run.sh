## training with dataset max_len=9700 && 0.95
#python /home/jovyan/work/insilico_translation/script/Training_Transformer.py -d /home/jovyan/work/insilico_translation/encode_base_type1_maxlen95_trainratio80/train_check -m /home/jovyan/work/insilico_translation/encode_base_type1_maxlen95_trainratio80 -e 10 -g 0 -p 'transformer' -l 9700

## training with dataset max_len=5034 && 0.75
#export CUDA_LAUNCH_BLOCKING=1
#python /home/jovyan/work/insilico_translation/script/Training_Transformer.py -d /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/train -m /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80 -e 10 -g 1 -p 'trans' -l 5034 -b 4


#python /home/jovyan/work/insilico_translation/script/Training_RNN.py -d  /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/train_check  -m  /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80 -e 10 -g 1 -p check_onehot_type1_RNN

#python /home/jovyan/work/insilico_translation/script/Training_LSTMWithAttention.py -d  /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/train_check  -m  /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80 -e 10 -g 0 -p check_onehot_type1


#python /home/jovyan/work/insilico_translation/script/Training_RNN.py -d  /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/train  -m  /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80 -e 10 -g 1 -p onehot_type1_RNN

#python /home/jovyan/work/insilico_translation/script/Training_LSTMWithAttention.py -d  /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/train_check  -m  /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80 -e 10 -g 0 -p train_check_onehot_type1_LSTMwAttention

#python /home/jovyan/work/insilico_translation/script/Training_CNN.py  --data_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/train   --output_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM   --model_type  'TRANSAID_v3' --batch_size 32   --learning_rate 0.001   --epochs 50   --patience 5     --gpu 0   --seed 42 --prefix 'TRANSAID_v3'

#python /home/jovyan/work/insilico_translation/script/Training_RNN_mod.py -d  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/train  -m  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM -b 32 -e 50 -g 1 -p RNN_mod -s 42

#python /home/jovyan/work/insilico_translation/script/Training_CNN.py  --data_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen98_ratio80_NM/train   --output_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen98_ratio80_NM   --model_type  'TRANSAID_v2' --batch_size 32   --learning_rate 0.001   --epochs 50   --patience 5     --gpu 1   --seed 42   --max_len  11451 --prefix 'TRANSAID_v2'

#python /home/jovyan/work/insilico_translation/script/Training_CNN.py  --data_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM/train   --output_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM   --model_type  'TRANSAID_v2' --batch_size 32   --learning_rate 0.001   --epochs 50   --patience 5     --gpu 1   --seed 42   --max_len  27308 --prefix 'TRANSAID_v2'

#python /home/jovyan/work/insilico_translation/script/Training_CNN.py  --data_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM/train   --output_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM   --model_type  'TRANSAID_v1' --batch_size 32   --learning_rate 0.001   --epochs 50   --patience 5     --gpu 0   --seed 42   --max_len  27308 --prefix 'TRANSAID_v1'

#python /home/jovyan/work/insilico_translation/script/Training_CNN.py  --data_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM/train   --output_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM   --model_type  'TRANSAID_v2' --batch_size 32   --learning_rate 0.001   --epochs 50   --patience 5     --gpu 0   --seed 42   --max_len  27308 --prefix 'TRANSAID_v2'

#python /home/jovyan/work/insilico_translation/script/Training_CNN.py  --data_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/train   --output_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM   --model_type  'TRANSAID_v2' --batch_size 32   --learning_rate 0.0001   --epochs 50   --patience 5     --gpu 1   --seed 42  --prefix 'TRANSAID_v2_lr4zero'

python /home/jovyan/work/insilico_translation/script/Training_CNN.py  --data_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM/train   --output_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM   --model_type  'TRANSAID_v3' --batch_size 32   --learning_rate 0.001   --epochs 50   --patience 5     --gpu 1   --seed 42   --max_len  27308 --prefix 'TRANSAID_v3'