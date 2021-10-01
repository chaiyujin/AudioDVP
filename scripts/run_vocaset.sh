function DRAW_DIVIDER() {
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" ' ' | tr ' ' '-'
}

function RUN_VOCASET() {
  ERROR='\033[0;31m[ERROR]\033[0m'

  local CWD=${PWD}
  local SPEAKER="FaceTalk_170725_00137_TA"
  local EPOCH_NFR=200
  local DATA_DIR=data/vocaset

  # Override from arguments
  for var in "$@"
  do
    if [[ $var =~ --speaker\=(.*) ]]; then
      SPEAKER="${BASH_REMATCH[1]}"
    elif [[ $var =~ --epoch_nfr\=(.*) ]]; then
      EPOCH_NFR="${BASH_REMATCH[1]}"
    fi
  done

  # Check variables
  # - Check speaker is FaceTalk
  if [[ ! ${SPEAKER} =~ FaceTalk_.* ]]; then
    printf "${ERROR} SPEAKER=${SPEAKER}, is not one from VOCASET!\n"
    exit 1
  fi
  # - Check EPOCH_NFR is times of 25
  if [ "$(( ${EPOCH_NFR} % 25 ))" -ne 0 ]; then
    printf "${ERROR} EPOCH_NFR=${EPOCH_NFR}, which is not times of 25!\n"
    exit 1
  fi

  # The checkpoints directory for this speaker
  local NET_DIR="$DATA_DIR/$SPEAKER/checkpoints"

  # Print arguments
  DRAW_DIVIDER;
  printf "Speaker        : $SPEAKER\n"
  printf "Epoch for NFR  : $EPOCH_NFR\n"
  printf "Data directory : $DATA_DIR\n"
  printf "Checkpoints    : $NET_DIR\n"

  # *------------------------------------------------ Data Preparing ------------------------------------------------* #

  DRAW_DIVIDER;

  # prepare data
  python3 utils/tools_vocaset_video.py prepare --dataset_dir $CWD/$DATA_DIR --speakers $SPEAKER

  # *-------------------------------------------- 3D face reconstruction --------------------------------------------* #

  DRAW_DIVIDER;

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
      --data_dir "$DATA_DIR/$SPEAKER" \
      --net_dir $NET_DIR \
    ;
    cd ${CWD}
  else
    printf "3D Face Reconstruction is already trained, checkpoint is found at: ${NET_DIR}/recons3d_net.pth\n"
  fi

  # generate masks
  python3 utils/tools_vocaset_video.py generate_masks --dataset_dir $CWD/$DATA_DIR --speakers $SPEAKER

  # visualize by generating debug videos
  python3 utils/tools_vocaset_video.py visualize_reconstruction --dataset_dir $CWD/$DATA_DIR --speakers $SPEAKER

  # *-------------------------------------------- Train Audio2Expression --------------------------------------------* #

  DRAW_DIVIDER;

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
      --data_dir "$DATA_DIR/$SPEAKER" \
      --net_dir $NET_DIR \
    ;
    cd ${CWD}
  else
    printf "Audio2Expression is already trained, checkpoint is found at: ${NET_DIR}/delta_net.pth\n"
  fi

  # *------------------------------------------ Train neural face renderer ------------------------------------------* #

  DRAW_DIVIDER;

  # data generation
  python3 utils/tools_vocaset_video.py build_nfr_dataset --dataset_dir $CWD/$DATA_DIR --speakers $SPEAKER

  # training
  if [ ! -d "${NET_DIR}/nfr" ]; then
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
    cd ${CWD}

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
  fi
  
  if [ ! -f "${NET_DIR}/nfr/${EPOCH_NFR}_net_G.pth" ]; then
    printf "${ERROR} Neural Face Renderer is trained, but we failed to find expected checkpoint at: ${NET_DIR}/nfr/${EPOCH_NFR}_net_G.pth\n"
    exit 1
  else
    printf "Neural Face Renderer is already trained!\n"
  fi

  # *---------------------------------------------------- Testing ---------------------------------------------------* #

  DRAW_DIVIDER;

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
      cd ${CWD}

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
}

# RUN_VOCASET --speaker=FaceTalk_170725_00137_TA --epoch_nfr=200;
RUN_VOCASET --speaker=FaceTalk_170908_03277_TA --epoch_nfr=200;
