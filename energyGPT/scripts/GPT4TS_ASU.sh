export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Forecasting" ]; then
    mkdir ./logs/Forecasting
fi

model_name=EnergyGPT



root_path_name=/home/lm/energyGPT/dataset/
data_path_name=ASUh.csv
model_id_name=ASUh
data_name=ASU


for dim in 32
do
for seq_len in 48
do
for lr in 0.0002
do
for pred_len in 1
do
for scale in 10000
do
    python -u run_Exp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len'your model name'\
    --model $model_name \
    --data $data_name \
    --task_name forecast \
    --features MS \
    --target 'Tamp_E'\
    --seq_len $seq_len \
    --label_len $seq_len \
    --pred_len $pred_len \
    --enc_in 35 \
    --e_layers 2 \
    --n_heads 16 \
    --d_model 768 \
    --d_ff 32 \
    --head_dropout 0 \
    --adapter_dropout 0.1 \
    --patch_len 48 \
    --stride 8 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 --batch_size 1 --learning_rate $lr \
    --warmup_epochs 10 \
    --scale $scale \
    --gpt_layers 6 \
    --spect_adapter_layer 6 \
    --adapter_layer 6 \
    --T_type 1 \
    --C_type 1 \
    --adapter_dim $dim \
    --use_multi_gpu

done
done
done
done
done

