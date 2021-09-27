SPEAKER="FaceTalk_170725_00137_TA"
DATA_DIR=data/vocaset
NET_DIR="$DATA_DIR/$SPEAKER/checkpoints"

# -------------------------------------------------------------------------------------------------------------------- #
#                                                    Data Preparing                                                    #
# -------------------------------------------------------------------------------------------------------------------- #

# prepare data
python3 utils/tools_vocaset_video.py prepare --dataset_dir $DATA_DIR --speakers $SPEAKER

# -------------------------------------------------------------------------------------------------------------------- #
#                                                3D face reconstruction                                                #
# -------------------------------------------------------------------------------------------------------------------- #

if [ ! -f "${NET_DIR}/recons3d_net.pth" ]; then
  python3 train.py \
    --dataset_mode multi \
    --num_epoch 20 \
    --lambda_photo 2.0 \
    --lambda_land 1.0 \
    --serial_batches False \
    --display_freq 400 \
    --print_freq 400 \
    --batch_size 5 \
    --data_dir $DATA_DIR \
    --net_dir $NET_DIR \
  ;
else
  echo "3D Face Reconstruction is already trained, checkpoint is found at: ${NET_DIR}/recons3d_net.pth"
fi

# generate masks
python3 utils/tools_vocaset_video.py generate_masks --dataset_dir $DATA_DIR --speakers $SPEAKER

# visualize by generating debug videos
python3 utils/tools_vocaset_video.py visualize_reconstruction --dataset_dir $DATA_DIR --speakers $SPEAKER

# -------------------------------------------------------------------------------------------------------------------- #
#                                                Train Audio2Expression                                                #
# -------------------------------------------------------------------------------------------------------------------- #

if [ ! -f "${NET_DIR}/delta_net.pth" ]; then
  # train audio2expression network
  python train_exp.py \
    --dataset_mode multi_audio_expr \
    --num_epoch 40 \
    --serial_batches False \
    --display_freq 800 \
    --print_freq 800 \
    --batch_size 5 \
    --lr 1e-3 \
    --lambda_delta 1.0 \
    --data_dir $DATA_DIR \
    --net_dir $NET_DIR \
  ;
else
  echo "Audio2Expression is already trained, checkpoint is found at: ${NET_DIR}/delta_net.pth"
fi

# -------------------------------------------------------------------------------------------------------------------- #
#                                              Train neural face renderer                                              #
# -------------------------------------------------------------------------------------------------------------------- #

# data generation
python3 utils/tools_vocaset_video.py build_nfr_dataset --dataset_dir $DATA_DIR --speakers $SPEAKER

# training
EPOCH_NFR=200
# check times of 25
if [ "$(( ${EPOCH_NFR} % 25 ))" -ne 0 ]; then
  echo "EPOCH_NFR=${EPOCH_NFR}, which is not times of 25!"
  exit 1
fi

if [ ! -f "${NET_DIR}/nfr/${EPOCH_NFR}_net_G.pth" ]; then
  # train neural face renderer
  python3 vendor/neural-face-renderer/train.py \
    --checkpoints_dir "$NET_DIR" \
    --dataroot "$DATA_DIR/$SPEAKER/nfr/AB" \
    --dataset_mode temporal \
    --preprocess none --load_size 256 --Nw 7 \
    --name nfr --model nfr \
    --netG unet_256 --direction BtoA --norm batch --pool_size 0 --use_refine --input_nc 21 \
    --num_threads 4 --batch_size 16 --lambda_L1 100 \
    --n_epochs ${EPOCH_NFR} --n_epochs_decay 0 \
  ;

  # remove other checkpoints of nfr
  REGEX_CKPT=".*_net_.*"
  REGEX_SAVE=".*${EPOCH_NFR}_net_.*"
  for entry in "$NET_DIR/nfr"/*
  do
    if [[ $entry =~ $REGEX_CKPT ]]; then
      # don't delete the one we need
      if [[ $entry =~ $REGEX_SAVE ]]; then
        continue
      fi
      # delete other checkpoints
      echo delete "$entry"
      rm "$entry"
    fi
  done
else
  echo "Neural Face Renderer is already trained!"
fi

# -------------------------------------------------------------------------------------------------------------------- #
#                                                        Testing                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

for ((i=20;i<40;i++)); do
  y=$(printf clip"%02d" $i)
  audio_dir=$DATA_DIR/$SPEAKER/test/$y
  results_dir=$DATA_DIR/$SPEAKER/results
  result_vpath=$results_dir/test-$y.mp4

  if [ -f $result_vpath ]; then
    echo "$result_vpath is already generated! Skip."
    continue
  fi

  if [ -d $audio_dir ]; then
    echo "Generating results for $audio_dir"

    # extract high-level feature from test audio
    mkdir -p $audio_dir/feature
    python vendor/ATVGnet/code/test.py -i $audio_dir/

    # predict expression parameter fron audio feature
    python3 test_exp.py \
      --dataset_mode audio_expression \
      --data_dir $audio_dir \
      --net_dir  $NET_DIR \
    ;

    # reenact face using predicted expression parameter
    python3 reenact.py --src_dir $audio_dir --tgt_dir $audio_dir

    # neural rendering the reenact face sequence
    python3 vendor/neural-face-renderer/test.py --model test \
      --netG unet_256 \
      --direction BtoA \
      --dataset_mode temporal_single \
      --norm batch \
      --input_nc 21 \
      --Nw 7 \
      --preprocess none \
      --eval \
      --use_refine \
      --name nfr \
      --checkpoints_dir $NET_DIR \
      --dataroot $audio_dir/reenact \
      --results_dir $audio_dir \
      --epoch $EPOCH_NFR \
    ;

    # composite lower face back to original video
    python3 comp.py --src_dir $audio_dir --tgt_dir $audio_dir

    # create video result
    mkdir -p "$results_dir"
    ffmpeg -y -loglevel error \
      -thread_queue_size 8192 -i $audio_dir/audio/audio.wav \
      -thread_queue_size 8192 -i $audio_dir/comp/%05d.png \
      -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest $result_vpath \
    ;
  fi
done
