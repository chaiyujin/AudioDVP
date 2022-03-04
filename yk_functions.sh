#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

# * Global variables
ERROR='\033[0;31m[ERROR]\033[0m'
CWD=${PWD}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                  Tool Functions                                                  * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function DRAW_DIVIDER() {
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" ' ' | tr ' ' '-'
}

function RUN_WITH_LOCK_GUARD() {
  local _end=
  local lock_file=
  local tag=
  local debug=
  local cmd=
  for i in "$@"; do
    if [ -z "${_end}" ]; then
      if [ "$i" = "--" ]; then
        _end=true
        continue
      fi
      case $i in
        -l=* | --lock_file=* ) lock_file=${i#*=}  ;;
        -t=* | --tag=*       ) tag=${i#*=}        ;;
        -d   | --debug       ) debug=true         ;;
        *) echo "[LockFileGuard]: Wrong argument ${i}"; exit 1;;
      esac
    else
      cmd="$cmd '$i'"
    fi
  done

  # lock_file must be given
  if [ -z "$lock_file" ]; then
    printf "lock_file is not set!\n"
    exit 1
  fi

  if [ -f "$lock_file" ]; then
    printf "[Skip]: Command '${tag}' is already done. (lock_file: '${lock_file}')\n"
    return
  fi

  # to abspath
  mkdir -p "$(dirname $lock_file)"
  lock_file="$(cd $(dirname $lock_file) && pwd)/$(basename $lock_file)"

  # debug message
  if [ -n "$debug" ]; then
    echo "--------------------------------------------------------------------------------"
    echo "LOCK_FILE: $lock_file"
    echo "COMMAND:   $cmd"
    echo "--------------------------------------------------------------------------------"
  fi

  # Run command in subshell and creat lock file if success
  if (eval "$cmd") ; then
    [ -f "$lock_file" ] || touch "$lock_file"
  else
    echo "Failed to run command '${tag}'"
    exit 1
  fi
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                              Functions for Each Step                                             * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function PrepareData() {
  local DATA_SRC=
  local SPEAKER=
  local EXP_DIR=
  local EPOCH=
  local USE_SEQS=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_src=* ) DATA_SRC=${var#*=}  ;;
      --speaker=*  ) SPEAKER=${var#*=}   ;;
      --exp_dir=*  ) EXP_DIR=${var#*=}   ;;
      --epoch=*    ) EPOCH=${var#*=}     ;;
      --use_seqs=* ) USE_SEQS=${var#*=}  ;;
      --debug      ) DEBUG="--debug"     ;;
    esac
  done

  [ -n "$DATA_SRC" ] || { echo "data_src is not set!"; exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";  exit 1; }
  [ -n "$EXP_DIR"  ] || { echo "exp_dir is not set!";  exit 1; }
  [ -n "$EPOCH"    ] || { echo "epoch is not set!";    exit 1; }

  local NET_DIR="$EXP_DIR/checkpoints"
  local DATA_DIR="$EXP_DIR/data"
  local RECONS_DIR="$EXP_DIR/reconstructed"
  local JOB_LOCK="$RECONS_DIR/all_done.lock"

  # * Guard lock file
  if [ -f "$JOB_LOCK" ]; then
    printf "[SKIP]: Reconstruction is already done before!\n"
    return
  fi

  # prepare data
  python3 yk_tools.py "prepare_$DATA_SRC" \
    --data_dir $DATA_DIR \
    --speaker  $SPEAKER  \
    --use_seqs $USE_SEQS \
    ${DEBUG} \
  || {
    printf "${ERROR} Failed to prepare data for source: ${DATA_SRC}!\n";
    exit 1;
  }
  
  RUN_WITH_LOCK_GUARD --tag="Reconstruct" --lock_file="${NET_DIR}/recons3d_net.pth" -- \
  python3 train.py \
    --dataset_mode    multi \
    --num_epoch       ${EPOCH} \
    --serial_batches  False \
    --display_freq    400 \
    --print_freq      400 \
    --net_dir         $NET_DIR \
    --data_dir        $DATA_DIR \
    --recons_dir      $RECONS_DIR \
  ;

  # generate masks
  python3 yk_tools.py generate_masks \
    --recons_dir $RECONS_DIR \
    ${DEBUG} \
  || {
    printf "${ERROR} Failed to generate masks!\n";
    exit 1;
  }

  # visualize by generating debug videos
  python3 yk_tools.py visualize_reconstruction \
    --data_dir $DATA_DIR \
    --recons_dir $RECONS_DIR \
    ${DEBUG} \
  || {
    printf "${ERROR} Failed to visualize reconstruction!\n";
    exit 1;
  }

  # * Create lock file if failed to find lock file
  [ -f "$JOB_LOCK" ] || touch "$JOB_LOCK"
}

# * -------------------------------------------- Train Audio2Expression -------------------------------------------- * #

function TrainA2E() {
  local EXP_DIR=
  local EPOCH=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --exp_dir=* ) EXP_DIR=${var#*=}   ;;
      --epoch=*   ) EPOCH=${var#*=}     ;;
    esac
  done

  [ -n "$EXP_DIR" ] || { echo "exp_dir is not set!";  exit 1; }
  [ -n "$EPOCH"   ] || { echo "epoch is not set!";    exit 1; }

  local NET_DIR="$EXP_DIR/checkpoints"
  local DATA_DIR="$EXP_DIR/data"
  local RECONS_DIR="$EXP_DIR/reconstructed"

  # train
  RUN_WITH_LOCK_GUARD --tag="Audio2Expression" --lock_file="${NET_DIR}/net_a2e.pth" -- \
  python train_exp.py \
    --dataset_mode    multi_audio_expr \
    --num_epoch       $EPOCH \
    --serial_batches  False \
    --display_freq    800 \
    --print_freq      800 \
    --batch_size      5 \
    --lr              1e-3 \
    --lambda_delta    1.0 \
    --net_dir         $NET_DIR \
    --data_dir        $DATA_DIR \
    --recons_dir      $RECONS_DIR \
  ;
}

# * ------------------------------------------ Train neural face renderer ------------------------------------------ * #

function TrainNFR() {
  local EXP_DIR=
  local EPOCH=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --exp_dir=*   ) EXP_DIR=${var#*=}   ;;
      --epoch=*     ) EPOCH=${var#*=}     ;;
      --debug       ) DEBUG="--debug"     ;;
    esac
  done

  [ -n "$EXP_DIR"  ] || { echo "exp_dir is not set!";  exit 1; }
  [ -n "$EPOCH"    ] || { echo "epoch is not set!";    exit 1; }

  local NET_DIR="$EXP_DIR/checkpoints"
  local DATA_DIR="$EXP_DIR/data"
  local RECONS_DIR="$EXP_DIR/reconstructed"
  local NFR_DATA_DIR="$EXP_DIR/nfr"

  # data generation
  python3 yk_tools.py build_nfr_dataset \
    --data_dir     $DATA_DIR \
    --recons_dir   $RECONS_DIR \
    --nfr_data_dir $NFR_DATA_DIR \
    ${DEBUG} \
  || {
    printf "${ERROR} Failed to build nfr dataset!\n";
    exit 1;
  }

  # training
  if [ ! -d "${NET_DIR}/nfr" ]; then
    # train neural face renderer
    if ! python3 vendor/neural-face-renderer/train.py \
      --checkpoints_dir "$NET_DIR" \
      --dataroot        "$NFR_DATA_DIR/AB" \
      --dataset_mode    temporal \
      --save_epoch_freq 20 \
      --preprocess none --load_size 256 --Nw 7 \
      --name nfr --model nfr \
      --netG unet_256 --direction BtoA --norm batch --pool_size 0 --use_refine --input_nc 21 \
      --num_threads 4 --batch_size 16 --lambda_L1 100 \
      --n_epochs ${EPOCH} --n_epochs_decay 0 \
    ; then
      printf "${ERROR} Failed to train NFR!"
      exit 1
    fi
    cd ${CWD}

    # # remove other checkpoints of nfr
    # REGEX_CKPT=".*_net_.*"
    # REGEX_SAVE=".*${EPOCH}_net_.*/$SPEAKER"
    # for entry in "$NET_DIR/nfr"/*
    # do
    #   if [[ $entry =~ $REGEX_CKPT ]]; then
    #     # don't delete the one we need
    #     if [[ $entry =~ $REGEX_SAVE ]]; then
    #       continue
    #     fi
    #     # delete other checkpoints
    #     echo delete "$entry"
    #     rm "$entry"
    #   fi
    # done
  fi
  
  if [ ! -f "${NET_DIR}/nfr/${EPOCH}_net_G.pth" ]; then
    printf "${ERROR} Neural Face Renderer is trained, but we failed to find expected checkpoint at: ${NET_DIR}/nfr/${EPOCH}_net_G.pth\n"
    exit 1
  else
    printf "Neural Face Renderer is already trained!\n"
  fi
}

# * ------------------------------------------------- Test Function ------------------------------------------------ * #

function TestClip() {
  local EXP_DIR=
  local EPOCH_NFR=
  local SRC_DIR=
  local TGT_DIR=
  local TGT_REC_DIR=
  local RES_DIR=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --exp_dir=*       ) EXP_DIR=${var#*=}   ;;
      --epoch_nfr=*     ) EPOCH_NFR=${var#*=} ;;
      --src_audio_dir=* ) SRC_DIR=${var#*=}   ;;
      --tgt_video_dir=* ) TGT_DIR=${var#*=}   ;;
      --tgt_recons_dir=*) TGT_REC_DIR=${var#*=} ;;
      --result_dir=*    ) RES_DIR=${var#*=}   ;;
    esac
  done
  # check
  [ -n "$EXP_DIR"     ] || { echo "exp_dir is not set!";       exit 1; }
  [ -n "$SRC_DIR"     ] || { echo "src_audio_dir is not set!"; exit 1; }
  [ -n "$TGT_DIR"     ] || { echo "tgt_video_dir is not set!"; exit 1; }
  [ -n "$TGT_REC_DIR" ] || { echo "tgt_recons_dir is not set!"; exit 1; }
  [ -n "$RES_DIR"     ] || { echo "result_dir is not set!";    exit 1; }

  local NET_DIR="$EXP_DIR/checkpoints"

  # guard
  if [ ! -f "$SRC_DIR/audio/audio.wav" ]; then
    printf "${ERROR} Failed to find '$SRC_DIR/audio/audio.wav'!\n"
    return
  fi

  printf "> Generating results for '$SRC_DIR', reenacting video from '$TGT_DIR'\n"
  
  # Audio feature
  mkdir -p $SRC_DIR/feature;
  RUN_WITH_LOCK_GUARD --tag="audio_feat" --lock_file="$SRC_DIR/audio_feat.lock" -- \
  python3 vendor/ATVGnet/code/test.py -i $SRC_DIR/ ;

  # Predict coeffs
  RUN_WITH_LOCK_GUARD --tag="audio2expr" --lock_file="$RES_DIR/pred_a2e.lock" -- \
  python3 test_exp.py --dataset_mode audio_expression --data_dir $SRC_DIR --result_dir $RES_DIR --net_dir $NET_DIR;

  # reenact face using predicted expression parameter
  RUN_WITH_LOCK_GUARD --tag="reenact and render" --lock_file="$RES_DIR/reenact.lock" -- \
  python3 reenact.py --src_dir "$RES_DIR" --tgt_dir "$TGT_REC_DIR";

  # create video result
  local vpath="$RES_DIR-render.mp4"
  if [ ! -f "$vpath" ]; then
    mkdir -p "$(dirname $vpath)" && \
    ffmpeg -y -loglevel error \
      -thread_queue_size 8192 -i $SRC_DIR/audio/audio.wav \
      -thread_queue_size 8192 -i $RES_DIR/reenact/%05d.png \
      -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest "$vpath" \
    ;
  fi

  # * If NFR is not trained, we directly return
  if [ -z "$EPOCH_NFR" ]; then
    return
  fi

  # neural rendering the reenact face sequence
  RUN_WITH_LOCK_GUARD --tag="NFR for reenacted" --lock_file="$RES_DIR/nfr.lock" -- \
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
    --epoch           $EPOCH_NFR \
    --dataroot        $RES_DIR/reenact \
    --results_dir     $RES_DIR \
  ;

  # composite lower face back to original video
  RUN_WITH_LOCK_GUARD --tag="comp" --lock_file="$RES_DIR/comp.lock" -- \
  python3 comp.py --src_dir "$RES_DIR" --tgt_dir "$TGT_DIR" --recons_dir "$TGT_REC_DIR";

  # create video result
  local vpath="$RES_DIR-nfr.mp4"
  if [ ! -f "$vpath" ]; then
    mkdir -p "$(dirname $vpath)" && \
    ffmpeg -y -loglevel error \
      -thread_queue_size 8192 -i $SRC_DIR/audio/audio.wav \
      -thread_queue_size 8192 -i $RES_DIR/comp/%05d.png \
      -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest "$vpath" \
    ;
  fi

}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                Collection of steps                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function RUN_YK_EXP() {
  local DATA_SRC=
  local SPEAKER=
  local EPOCH_D3D=60
  local EPOCH_A2E=60
  local EPOCH_NFR=
  local USE_SEQS=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_src=*  ) DATA_SRC=${var#*=}  ;;
      --data_dir=*  ) DATA_DIR=${var#*=}  ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --epoch_d3d=* ) EPOCH_D3D=${var#*=} ;;
      --epoch_a2e=* ) EPOCH_A2E=${var#*=} ;;
      --epoch_nfr=* ) EPOCH_NFR=${var#*=} ;;
      --use_seqs=*  ) USE_SEQS=${var#*=}  ;;
      --debug       ) DEBUG="--debug"     ;;
    esac
  done

  # Check variables
  [ -n "$DATA_SRC" ] || { echo "data_src is not set!"; exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";  exit 1; }
  # Check EPOCH_NFR is none or times of 20
  if [ -n "${EPOCH_NFR}" ] && [ "$(( ${EPOCH_NFR} % 20 ))" -ne 0 ]; then
    printf "${ERROR} EPOCH_NFR=${EPOCH_NFR}, which is not times of 20!\n"
    exit 1
  fi
  DATA_SRC=${DATA_SRC,,}

  # Some preset dirs
  local EXP_DIR="yk_exp/$DATA_SRC/$SPEAKER"
  # Shared arguments
  local SHARED="--data_src=$DATA_SRC --speaker=$SPEAKER --exp_dir=$EXP_DIR ${DEBUG}"

  # Print arguments
  DRAW_DIVIDER;
  printf "Data source    : $DATA_SRC\n"
  printf "Speaker        : $SPEAKER\n"
  printf "Epoch for D3D  : $EPOCH_D3D\n"
  printf "Epoch for A2E  : $EPOCH_A2E\n"
  printf "Epoch for NFR  : $EPOCH_NFR\n"

  # * Step 1: Prepare data
  DRAW_DIVIDER; PrepareData $SHARED --epoch=$EPOCH_D3D --use_seqs=$USE_SEQS

  # * Step 2: Train audio to expression
  DRAW_DIVIDER; TrainA2E ${SHARED} --epoch=$EPOCH_A2E

  # * Step 3: (Optional) train neural renderer
  if [ -n "${EPOCH_NFR}" ]; then
    DRAW_DIVIDER; TrainNFR ${SHARED} --epoch=$EPOCH_NFR
  fi

  # * Test
  DRAW_DIVIDER;
  for d in "$EXP_DIR/data/test"/*; do
    if [ ! -d "$d" ]; then continue; fi
    local clip_id="$(basename ${d})"

    TestClip \
      --exp_dir="$EXP_DIR" \
      --epoch_nfr="$EPOCH_NFR" \
      --src_audio_dir="$d" \
      --tgt_video_dir="$d" \
      --tgt_recons_dir="$EXP_DIR/reconstructed/test/$clip_id" \
      --result_dir="$EXP_DIR/results/self-reenact/$clip_id" \
    ;
  done
}
