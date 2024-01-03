export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --root_path './data/exchange_rate/' \
  --data_path 'exchange_rate.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --x_dim 8 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ex_96' \
  --log_path 'log_trainer/ex_96' \
  --batch_size 128 \
  --epochs 30 \
  --coeff 0.1 \
  --learning_rate 1e-3 \

python -u run.py \
  --root_path './data/exchange_rate/' \
  --data_path 'exchange_rate.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --x_dim 8 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ex_192' \
  --log_path 'log_trainer/ex_192' \
  --batch_size 128 \
  --epochs 20\
  --coeff 0.1 \
  --learning_rate 1e-3 \

python -u run.py \
  --root_path './data/exchange_rate/' \
  --data_path 'exchange_rate.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --x_dim 8 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ex_336' \
  --log_path 'log_trainer/ex_336' \
  --batch_size 128 \
  --epochs 24 \
  --coeff 0.1 \
  --learning_rate 1e-3 \

python -u run.py \
  --root_path './data/exchange_rate/' \
  --data_path 'exchange_rate.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --x_dim 8 \
  --z_dim 25 \
  --h_dim 128 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ex_720' \
  --log_path 'log_trainer/ex_720' \
  --batch_size 128 \
  --epochs 20 \
  --coeff 0.01 \
  --learning_rate 2e-4 \




