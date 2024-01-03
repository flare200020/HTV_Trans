export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --root_path './data/illness/' \
  --data_path 'national_illness.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 2 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ill_24' \
  --log_path 'log_trainer/ill_24' \
  --batch_size 32 \
  --epochs 100 \
  --coeff 0.1 \
  --patience 12 \
  --learning_rate 3e-3 \


python -u run.py \
  --root_path './data/illness/' \
  --data_path 'national_illness.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 2 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ill_36' \
  --log_path 'log_trainer/ill_36' \
  --batch_size 32 \
  --epochs 100 \
  --coeff 0.1 \
  --patience 12 \
  --learning_rate 3e-3 \

python -u run.py \
  --root_path './data/illness/' \
  --data_path 'national_illness.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 2 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ill_48' \
  --log_path 'log_trainer/ill_48' \
  --batch_size 32 \
  --epochs 100 \
  --coeff 0.1 \
  --patience 12 \
  --learning_rate 3e-3 \


python -u run.py \
  --root_path './data/illness/' \
  --data_path 'national_illness.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 2 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ill_60' \
  --log_path 'log_trainer/ill_60' \
  --batch_size 32 \
  --epochs 100 \
  --coeff 0.1 \
  --patience 12 \
  --learning_rate 3e-3 \



