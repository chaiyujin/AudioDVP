DATA_DIR=data/vocaset_video
SPEAKER="FaceTalk_170725_00137_TA"
NET_DIR="$DATA_DIR/$SPEAKER/checkpoints"

# prepare data
python3 utils/tools_vocaset_video.py prepare --dataset_dir $DATA_DIR --speakers $SPEAKER

# -------------------------------------------------------------------------------------------------------------------- #
#                                                3D face reconstruction                                                #
# -------------------------------------------------------------------------------------------------------------------- #

# python3 train.py \
#   --dataset_mode multi \
#   --num_epoch 20 \
#   --lambda_photo 2.0 \
#   --lambda_land 1.0 \
#   --serial_batches False \
#   --display_freq 400 \
#   --print_freq 400 \
#   --batch_size 5 \
#   --data_dir $DATA_DIR \
#   --net_dir $NET_DIR \
# ;

# visualize by generating debug videos
# python3 utils/tools_vocaset_video.py visualize_reconstruction --dataset_dir $DATA_DIR


# -------------------------------------------------------------------------------------------------------------------- #
#                                                Train Audio2Expression                                                #
# -------------------------------------------------------------------------------------------------------------------- #

# # train audio2expression network
# python train_exp.py \
#   --dataset_mode multi_audio_expr \
#   --num_epoch 40 \
#   --serial_batches False \
#   --display_freq 800 \
#   --print_freq 800 \
#   --batch_size 5 \
#   --lr 1e-3 \
#   --lambda_delta 1.0 \
#   --data_dir $DATA_DIR \
#   --net_dir $NET_DIR \
# ;

# -------------------------------------------------------------------------------------------------------------------- #
#                                              Train neural face renderer                                              #
# -------------------------------------------------------------------------------------------------------------------- #

# data generation
python3 utils/tools_vocaset_video.py build_nfr_dataset --dataset_dir $DATA_DIR --speakers $SPEAKER

# train neural face renderer
python3 vendor/neural-face-renderer/train.py \
  --checkpoints_dir "$NET_DIR/nfr" \
  --dataroot "$DATA_DIR/$SPEAKER/nfr/AB" \
  --dataset_mode temporal \
  --preprocess none --load_size 256 --Nw 7 \
  --name nfr --model nfr \
  --netG unet_256 --direction BtoA --norm batch --pool_size 0 --use_refine --input_nc 21 \
  --num_threads 4 --batch_size 16 --lambda_L1 100 \
  --n_epochs 250 --n_epochs_decay 0 \
;

