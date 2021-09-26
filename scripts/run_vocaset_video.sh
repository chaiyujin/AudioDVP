DATA_DIR=data/vocaset_video

# prepare data
python3 scripts/tools_vocaset_video.py prepare --dataset_dir $DATA_DIR

# -------------------------------------------------------------------------------------------------------------------- #
#                                                3D face reconstruction                                                #
# -------------------------------------------------------------------------------------------------------------------- #

python3 train.py \
  --net_dir $DATA_DIR/checkpoints \
  --data_dir $DATA_DIR \
  --dataset_mode multi \
  --num_epoch 20 \
  --lambda_photo 2.0 \
  --lambda_land 1.0 \
  --serial_batches False \
  --display_freq 400 \
  --print_freq 400 \
  --batch_size 5 \
;

# python3 scripts/tools_vocaset_video.py visualize_reconstruction --dataset_dir $DATA_DIR
