DATA_DIR=data/vocaset_video

# prepare data
python3 scripts/prepare_data_vocaset.py --output_dir $DATA_DIR

# -------------------------------------------------------------------------------------------------------------------- #
#                                                3D face reconstruction                                                #
# -------------------------------------------------------------------------------------------------------------------- #

python3 train.py \
  --data_dir $DATA_DIR \
  --dataset_mode multi \
  --num_epoch 20 \
  --lambda_photo 2.0 \
  --lambda_land 2.0 \
  --serial_batches False \
  --display_freq 400 \
  --print_freq 400 \
  --batch_size 5 \
;

# # create reconstruction debug video
# /usr/bin/ffmpeg -hide_banner -y -loglevel warning \
#     -thread_queue_size 8192 -i $DATA_DIR/render/%05d.png \
#     -thread_queue_size 8192 -i $DATA_DIR/crop/%05d.png \
#     -thread_queue_size 8192 -i $DATA_DIR/overlay/%05d.png \
#     -i $DATA_DIR/audio/audio.aac \
