export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --root_path './data/weather/' \
  --data_path 'weather.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --x_dim 21 \
  --layer_xz 1 \
  --z_layers 1 \
  --h_dim 128 \
  --checkpoints_path 'model/weather_96' \
  --log_path 'log_trainer/weather_96' \
  --batch_size 128 \
  --epochs 100 \
  --learning_rate 2e-4\


python -u run.py \
  --root_path './data/weather/' \
  --data_path 'weather.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --x_dim 21 \
  --layer_xz 1 \
  --z_layers 1 \
  --h_dim 128 \
  --checkpoints_path 'model/weather_192' \
  --log_path 'log_trainer/weather_192' \
  --batch_size 128 \
  --epochs 70 \
  --learning_rate 2e-4\


python -u run.py \
  --root_path './data/weather/' \
  --data_path 'weather.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --x_dim 21 \
  --layer_xz 1 \
  --z_layers 1 \
  --h_dim 128 \
  --checkpoints_path 'model/weather_336' \
  --log_path 'log_trainer/weather_336' \
  --batch_size 128 \
  --epochs 70 \
  --learning_rate 2e-4\

python -u run.py \
  --root_path './data/weather/' \
  --data_path 'weather.csv' \
  --dataset 'custom' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --x_dim 21 \
  --layer_xz 1 \
  --z_layers 1 \
  --h_dim 128 \
  --checkpoints_path 'model/weather_720' \
  --log_path 'log_trainer/weather_720' \
  --batch_size 128 \
  --epochs 70 \
  --learning_rate 2e-4\


