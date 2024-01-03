export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTh1.csv' \
  --dataset 'ETTh1' \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/etth1_96' \
  --log_path 'log_trainer/etth1_96' \
  --batch_size 128 \
  --epochs 60

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTh1.csv' \
  --dataset 'ETTh1' \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/etth1_192' \
  --log_path 'log_trainer/etth1_192' \
  --batch_size 128 \
  --epochs 64

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTh1.csv' \
  --dataset 'ETTh1' \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/etth1_336' \
  --log_path 'log_trainer/etth1_336' \
  --batch_size 128 \
  --epochs 60

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTh1.csv' \
  --dataset 'ETTh1' \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/etth1_720' \
  --log_path 'log_trainer/etth1_720' \
  --batch_size 128 \
  --epochs 60

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTh2.csv' \
  --dataset 'ETTh2' \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/etth2_96' \
  --log_path 'log_trainer/etth2_96' \
  --batch_size 128 \
  --epochs 60

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTh2.csv' \
  --dataset 'ETTh2' \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 48 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 48 \
  --checkpoints_path 'model/etth2_192' \
  --log_path 'log_trainer/etth2_192' \
  --batch_size 128 \
  --epochs 60

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTh2.csv' \
  --dataset 'ETTh2' \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 96 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 8 \
  --checkpoints_path 'model/etth2_336' \
  --log_path 'log_trainer/etth2_336' \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --epochs 50

python -u run.py \
  --root_path './data/ETT/' \
  --data_path 'ETTh2.csv' \
  --dataset 'ETTh2' \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --x_dim 7 \
  --z_dim 25 \
  --h_dim 96 \
  --layer_xz 1 \
  --z_layers 1 \
  --embd_h 8 \
  --checkpoints_path 'model/etth2_720' \
  --log_path 'log_trainer/etth2_720' \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --epochs 50