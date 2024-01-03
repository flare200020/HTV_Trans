export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTm1.csv' \
  --dataset 'ETTm1' \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ettm1_96' \
  --log_path 'log_trainer/ettm1_96' \
  --epochs 50


python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTm1.csv' \
  --dataset 'ETTm1' \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --x_dim 7 \
  --coeff 0.5 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ettm1_192' \
  --log_path 'log_trainer/ettm1_192' \
  --epochs 50


python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTm1.csv' \
  --dataset 'ETTm1' \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ettm1_336' \
  --log_path 'log_trainer/ettm1_336' \
  --epochs 50

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTm1.csv' \
  --dataset 'ETTm1' \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ettm1_720' \
  --log_path 'log_trainer/ettm1_720' \
  --epochs 50

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTm2.csv' \
  --dataset 'ETTm2' \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ettm2_96' \
  --log_path 'log_trainer/ettm2_96' \
  --epochs 50


python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTm2.csv' \
  --dataset 'ETTm2' \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ettm2_192' \
  --log_path 'log_trainer/ettm2_192' \
  --epochs 50


python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTm2.csv' \
  --dataset 'ETTm2' \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ettm2_336' \
  --log_path 'log_trainer/ettm2_336' \
  --epochs 50

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTm2.csv' \
  --dataset 'ETTm2' \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/ettm2_720' \
  --log_path 'log_trainer/ettm2_720' \
  --epochs 50



