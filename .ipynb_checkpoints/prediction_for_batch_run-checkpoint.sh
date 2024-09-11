#python /home/jovyan/work/insilico_translation/script/prediction_for_batch3.py -t 'TransformerModel' -m /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/transformer_transformer_model.pth  -d /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/validation/ -o /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/ -p 'validation' -g 1 -l 5034


#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'SimpleRNN' -m /home/jovyan/work/insilico_translation/encoded_data_start2_end1/simple_rnn_model.pth  -d /home/jovyan/work/insilico_translation/encoded_data_start2_end1/test_2/ -o /home/jovyan/work/insilico_translation/encoded_data_start2_end1/ -p 'test_2' -g 0 -l 5034

#python /home/jovyan/work/insilico_translation/encoded_data_start2_end1/prediction_for_batch.py -m /home/jovyan/work/insilico_translation/encoded_data_start2_end1/simple_rnn_model.pth  -d /home/jovyan/work/insilico_translation/encoded_data_start2_end1/test_2/ -o /home/jovyan/work/insilico_translation/encoded_data_start2_end1/ -p 'test_2' -g 0 -l 5034


#python /home/jovyan/work/insilico_translation/encoded_data_start2_end1/prediction_for_batch.py -m /home/jovyan/work/insilico_translation/encoded_data_start2_end1/simple_rnn_model.pth  -d /home/jovyan/work/insilico_translation/encoded_data_start2_end1/test/ -o /home/jovyan/work/insilico_translation/encoded_data_start2_end1/ -p 'test' -g 1 -l 5034

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'TransformerModel' -m /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/transformer_transformer_model.pth  -d /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/train_check/ -o /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80 -p 'train_check' -g 0 -l 5034


#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'TransformerModel' -m /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/transformer_transformer_model.pth  -d /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/train/ -o /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80 -p 'train' -g 0 -l 5034

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'SimpleLSTM' -m /home/jovyan/work/insilico_translation/encoded_data_start2_end1_long/SimpleLSTM_model.pth  -d /home/jovyan/work/insilico_translation/encoded_data_start2_end1_long/validation_check/ -o /home/jovyan/work/insilico_translation/encoded_data_start2_end1_long -p 'validation_check' -g 0 -l 5034

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'SimpleRNN' -m /home/jovyan/work/insilico_translation/encoded_data_start2_end1/simple_rnn_model.pth  -d /home/jovyan/work/insilico_translation/encoded_data_start2_end1/validation_check/ -o /home/jovyan/work/insilico_translation/encoded_data_start2_end1 -p 'validation_check' -g 0 -l 5034

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'TransformerModel' -m /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/trans_transformer_model.pth  -d /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80/validation/ -o /home/jovyan/work/insilico_translation/encode_base_type1_maxlen75_trainratio80 -p 'validation' -g 0 -l 5034

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'LSTMWithAttention' -m /home/jovyan/work/insilico_translation/encode_onehot_type1_maxlen75_trainratio80/onehot_type1_Epoch7_model.pth  -d /home/jovyan/work/insilico_translation/encode_onehot_type1_maxlen75_trainratio80/train_check -o /home/jovyan/work/insilico_translation/encode_onehot_type1_maxlen75_trainratio80 -p 'train_check' -g 0 -l 5034


#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'LSTMWithAttention' -m /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/check_onehot_type1_model.pth  -d /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/validation_check -o /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80 -p 'validation_check' -g 0 -l 5034

#for i in {1..10}
#do
#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'SimpleRNN' -m /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/onehot_type1_RNN_Epoch${i}_model.pth  -d /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/validation/ -o /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80 -p "validation_RNN_Epoch${i}" -g 1 -l 5034

#for i in {1..10}
#do
#python /home/jovyan/work/insilico_translation/script/prediction_for_batch.py -t 'LSTMWithAttention' -m /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/train_check_onehot_type1_LSTMwAttention_Epoch${i}_model.pth  -d /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80/validation/ -o /home/jovyan/work/insilico_translation/encoded_onehot_type1_maxlen75_trainratio80 -p "validation_check_LSTMwAttention_Epoch${i}" -g 1 -l 5034
#done

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch_latest.py  --model_path  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/RNN_mod_Epoch35_model.pth  --model_type 'SimpleRNN'  --data_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/validation  --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM --batch_size 32 --gpu 0 --max_len 5049  --prefix 'validation_SimpleRNN_Epoch35'

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch_latest.py  --model_path  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/TranslationAI_v2_best_model.pth  --model_type 'TranslationAI_v2'  --data_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/validation  --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM --batch_size 32 --gpu 1 --max_len 5049  --prefix 'validation_TranslationAI_v2'

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch_latest.py  --model_path  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/TranslationAI_v3_best_model.pth  --model_type 'TranslationAI_v3'  --data_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/validation  --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM --batch_size 32 --gpu 1 --max_len 5049  --prefix 'validation_TranslationAI_v3'

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch_latest.py  --model_path  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/TranslationAI_v2_lr4zero_best_model.pth  --model_type 'TranslationAI_v2'  --data_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM/validation  --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen75_ratio80_NM --batch_size 32 --gpu 1 --max_len 5049  --prefix 'validation_TranslationAI_v2_lr4zero'

#python /home/jovyan/work/insilico_translation/script/prediction_for_batch_latest.py  --model_path  /home/jovyan/work/insilico_translation/onehot_type3_maxlen98_ratio80_NM/TranslationAI_v2_best_model.pth  --model_type 'TranslationAI_v2'  --data_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen98_ratio80_NM/validation  --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen98_ratio80_NM --batch_size 32 --gpu 1 --max_len 11451  --prefix 'validation_TranslationAI_v2'

python /home/jovyan/work/insilico_translation/script/prediction_for_batch_latest.py  --model_path  /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM/TranslationAI_v2_best_model.pth  --model_type 'TranslationAI_v2'  --data_dir  /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM/validation  --output_dir /home/jovyan/work/insilico_translation/onehot_type3_maxlen9995_ratio80_NM --batch_size 32 --gpu 1 --max_len 27308  --prefix 'validation_TranslationAI_v2'

