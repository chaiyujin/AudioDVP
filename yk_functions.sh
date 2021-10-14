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
    printf "Command '${tag}' is skipped due to existance of lock_file ${lock_file}\n"
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
  local DATA_DIR=
  local NET_DIR=
  local SPEAKER=
  local EPOCH=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_source=*) DATA_SRC=${var#*=}  ;;
      --data_dir=*   ) DATA_DIR=${var#*=}  ;;
      --net_dir=*    ) NET_DIR=${var#*=}   ;;
      --speaker=*    ) SPEAKER=${var#*=}   ;;
      --epoch=*      ) EPOCH=${var#*=}     ;;
      --debug        ) DEBUG="--debug"     ;;
    esac
  done

  [ -n "$DATA_SRC" ] || { echo "data_source is not set!"; exit 1; }
  [ -n "$DATA_DIR" ] || { echo "data_dir is not set!";    exit 1; }
  [ -n "$NET_DIR"  ] || { echo "net_dir is not set!";     exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";     exit 1; }
  [ -n "$EPOCH"    ] || { echo "epoch is not set!";       exit 1; }

  # prepare data
  if ! python3 yk_tools.py "prepare_$DATA_SRC" --dataset_dir $DATA_DIR --speakers $SPEAKER ${DEBUG} ; then
    printf "${ERROR} Failed to prepare data for source: ${DATA_SRC}!\n"
    exit 1
  fi

  RUN_WITH_LOCK_GUARD --tag="Reconstruct" --lock_file="${NET_DIR}/recons3d_net.pth" -- \
  python3 train.py \
    --dataset_mode multi \
    --num_epoch ${EPOCH} \
    --lambda_photo 2.0 \
    --lambda_land 0.1 \
    --serial_batches False \
    --display_freq 400 \
    --print_freq 400 \
    --batch_size 5 \
    --data_dir "$DATA_DIR/$SPEAKER" \
    --net_dir $NET_DIR \
  ;

  # generate masks
  if ! python3 yk_tools.py generate_masks --dataset_dir $CWD/$DATA_DIR --speakers $SPEAKER ${DEBUG} ; then
    printf "${ERROR} Failed to generate masks!\n"
    exit 1
  fi

  # visualize by generating debug videos
  if ! python3 yk_tools.py visualize_reconstruction --dataset_dir $CWD/$DATA_DIR --speakers $SPEAKER ${DEBUG} ; then
    printf "${ERROR} Failed to visualize reconstruction!\n"
    exit 1
  fi
}

# * -------------------------------------------- Train Audio2Expression -------------------------------------------- * #

function TrainAudio2Expression() {
  local DATA_DIR=
  local NET_DIR=
  local SPEAKER=
  local EPOCH=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_dir=*  ) DATA_DIR=${var#*=}  ;;
      --net_dir=*   ) NET_DIR=${var#*=}   ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --epoch=*     ) EPOCH=${var#*=}     ;;
    esac
  done

  [ -n "$DATA_DIR" ] || { echo "data_dir is not set!"; exit 1; }
  [ -n "$NET_DIR"  ] || { echo "net_dir is not set!";  exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";  exit 1; }
  [ -n "$EPOCH"    ] || { echo "epoch is not set!";    exit 1; }

  # train
  RUN_WITH_LOCK_GUARD --tag="Reconstruct" --lock_file="${NET_DIR}/delta_net.pth" -- \
  python train_exp.py \
    --dataset_mode multi_audio_expr \
    --num_epoch ${EPOCH} \
    --serial_batches False \
    --display_freq 800 \
    --print_freq 800 \
    --batch_size 5 \
    --lr 1e-3 \
    --lambda_delta 1.0 \
    --data_dir "$DATA_DIR/$SPEAKER" \
    --net_dir $NET_DIR \
  ;
}

# * ------------------------------------------ Train neural face renderer ------------------------------------------ * #

function TrainNeuralFaceRenderer() {
  local DATA_DIR=
  local NET_DIR=
  local SPEAKER=
  local EPOCH=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_dir=*  ) DATA_DIR=${var#*=}  ;;
      --net_dir=*   ) NET_DIR=${var#*=}   ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --epoch=*     ) EPOCH=${var#*=}     ;;
      --debug       ) DEBUG="--debug"     ;;
    esac
  done

  [ -n "$DATA_DIR" ] || { echo "data_dir is not set!"; exit 1; }
  [ -n "$NET_DIR"  ] || { echo "net_dir is not set!";  exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";  exit 1; }
  [ -n "$EPOCH"    ] || { echo "epoch is not set!";    exit 1; }

  # data generation
  if ! python3 yk_tools.py build_nfr_dataset --dataset_dir $CWD/$DATA_DIR --speakers $SPEAKER ${DEBUG}; then
    printf "${ERROR} Failed to build nfr dataset!\n"
    exit 1
  fi

  # training
  if [ ! -d "${NET_DIR}/nfr" ]; then
    # train neural face renderer
    if ! python3 vendor/neural-face-renderer/train.py \
      --checkpoints_dir "$NET_DIR" \
      --dataroot "$DATA_DIR/$SPEAKER/nfr/AB" \
      --dataset_mode temporal \
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

    # remove other checkpoints of nfr
    REGEX_CKPT=".*_net_.*"
    REGEX_SAVE=".*${EPOCH}_net_.*"
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
  
  if [ ! -f "${NET_DIR}/nfr/${EPOCH}_net_G.pth" ]; then
    printf "${ERROR} Neural Face Renderer is trained, but we failed to find expected checkpoint at: ${NET_DIR}/nfr/${EPOCH}_net_G.pth\n"
    exit 1
  else
    printf "Neural Face Renderer is already trained!\n"
  fi
}

# * ------------------------------------------------- Test Function ------------------------------------------------ * #

function TestClip() {
  local clip_dir=
  local reenact_tar=
  local result_dir=
  local result_vpath=
  local net_dir=
  local epoch_nfr=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --clip_dir=*     ) clip_dir=${var#*=}     ;;
      --reenact_tar=*  ) reenact_tar=${var#*=}     ;;
      --result_dir=*   ) result_dir=${var#*=}   ;;
      --result_vpath=* ) result_vpath=${var#*=} ;;
      --net_dir=*      ) net_dir=${var#*=}      ;;
      --epoch_nfr=*    ) epoch_nfr=${var#*=}    ;;
    esac
  done
  # check
  [ -n "$clip_dir"     ] || { echo "clip_dir is not set!";     exit 1; }
  [ -n "$result_dir"   ] || { echo "result_dir is not set!";   exit 1; }
  [ -n "$result_vpath" ] || { echo "result_vpath is not set!"; exit 1; }
  [ -n "$net_dir"      ] || { echo "net_dir is not set!";      exit 1; }

  # guard
  if [ ! -f "$clip_dir/audio/audio.wav" ]; then
    printf "${ERROR} Failed to find '$clip_dir/audio/audio.wav'!\n"
    return
  fi

  printf "> Generating results for '$clip_dir'\n"
  cd ${CWD} && mkdir -p $clip_dir/feature

  RUN_WITH_LOCK_GUARD --tag="audio_feat" --lock_file="${clip_dir}/audio_feat.lock" -- \
  python3 vendor/ATVGnet/code/test.py -i $clip_dir/ ;

  RUN_WITH_LOCK_GUARD --tag="audio2expr" --lock_file="${clip_dir}/pred_a2e.lock" -- \
  python3 test_exp.py --dataset_mode audio_expression --data_dir $clip_dir --net_dir $net_dir;

  # reenact face using predicted expression parameter
  RUN_WITH_LOCK_GUARD --tag="reenact and render" --lock_file="${clip_dir}/reenact.lock" -- \
  python3 reenact.py --src_dir "$clip_dir" --tgt_dir "$reenact_tar";

  # create video result
  mkdir -p "$result_dir"
  ffmpeg -n -loglevel error \
    -thread_queue_size 8192 -i $clip_dir/audio/audio.wav \
    -thread_queue_size 8192 -i $clip_dir/reenact/%05d.png \
    -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest "${result_vpath}-render.mp4" \
  ;

  # * If NFR is not trained, we directly return
  if [ -z "${epoch_nfr}" ]; then
    return
  fi

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
    --checkpoints_dir $net_dir \
    --dataroot $clip_dir/reenact \
    --results_dir $clip_dir \
    --epoch $epoch_nfr \
  ;

  # composite lower face back to original video
  python3 comp.py --src_dir $audio_dir --tgt_dir $audio_dir

  # create video result
  mkdir -p "$result_dir"
  ffmpeg -y -loglevel error \
    -thread_queue_size 8192 -i $audio_dir/audio/audio.wav \
    -thread_queue_size 8192 -i $audio_dir/comp/%05d.png \
    -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest "${result_vpath}-nfr.mp4" \
  ;
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                Collection of steps                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function RUN_VOCASET() {
  local DATA_DIR=data/vocaset
  local SPEAKER="FaceTalk_170725_00137_TA"
  local EPOCH_D3D=20
  local EPOCH_A2E=40
  local EPOCH_NFR=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_dir=*  ) DATA_DIR=${var#*=}  ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --epoch_d3d=* ) EPOCH_D3D=${var#*=} ;;
      --epoch_a2e=* ) EPOCH_A2E=${var#*=} ;;
      --epoch_nfr=* ) EPOCH_NFR=${var#*=} ;;
      --debug       ) DEBUG="--debug"     ;;
    esac
  done
  # Check variables
  # - Check speaker is FaceTalk
  if [[ ! ${SPEAKER} =~ FaceTalk_.* ]]; then
    printf "${ERROR} SPEAKER=${SPEAKER}, is not one from VOCASET!\n"
    exit 1
  fi
  # - Check EPOCH_NFR is none or times of 25
  if [ -n "${EPOCH_NFR}" ] && [ "$(( ${EPOCH_NFR} % 25 ))" -ne 0 ]; then
    printf "${ERROR} EPOCH_NFR=${EPOCH_NFR}, which is not times of 25!\n"
    exit 1
  fi

  # The checkpoints directory for this speaker
  local NET_DIR="$DATA_DIR/$SPEAKER/checkpoints"

  # Print arguments
  DRAW_DIVIDER;
  printf "Speaker        : $SPEAKER\n"
  printf "Data directory : $DATA_DIR\n"
  printf "Checkpoints    : $NET_DIR\n"
  printf "Epoch for D3D  : $EPOCH_D3D\n"
  printf "Epoch for A2E  : $EPOCH_A2E\n"
  printf "Epoch for NFR  : $EPOCH_NFR\n"

  # Shared arguments
  local SHARED="--data_dir=$DATA_DIR --net_dir=$NET_DIR --speaker=$SPEAKER ${DEBUG}"

  DRAW_DIVIDER;
  PrepareData $SHARED --data_source=vocaset --epoch=$EPOCH_D3D

  DRAW_DIVIDER;
  TrainAudio2Expression ${SHARED} --epoch=$EPOCH_A2E

  # (Optional) train neural renderer
  if [ -n "${EPOCH_NFR}" ]; then
    DRAW_DIVIDER;
    TrainNeuralFaceRenderer ${SHARED} --epoch=$EPOCH_NFR
  fi

  # * Test
  DRAW_DIVIDER;
  for i in `seq 21 40`; do
    local clip_id=$(printf sentence"%02d" $i)

    TestClip \
      --clip_dir="$DATA_DIR/$SPEAKER/test/$clip_id" \
      --reenact_tar="$DATA_DIR/$SPEAKER/test/$clip_id" \
      --result_dir="$DATA_DIR/$SPEAKER/results" \
      --result_vpath="$DATA_DIR/$SPEAKER/results/test-$clip_id" \
      --net_dir="$NET_DIR" \
      --epoch_nfr="$EPOCH_NFR" \
    ;
  done
}



function RUN_CELEBTALK() {
  local DATA_DIR=data/celebtalk
  local SPEAKER="m000_obama"
  local EPOCH_D3D=20
  local EPOCH_A2E=40
  local EPOCH_NFR=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_dir=*  ) DATA_DIR=${var#*=}  ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --epoch_d3d=* ) EPOCH_D3D=${var#*=} ;;
      --epoch_a2e=* ) EPOCH_A2E=${var#*=} ;;
      --epoch_nfr=* ) EPOCH_NFR=${var#*=} ;;
      --debug       ) DEBUG="--debug"     ;;
    esac
  done
  # Check variables
  # - Check EPOCH_NFR is none or times of 25
  if [ -n "${EPOCH_NFR}" ] && [ "$(( ${EPOCH_NFR} % 25 ))" -ne 0 ]; then
    printf "${ERROR} EPOCH_NFR=${EPOCH_NFR}, which is not times of 25!\n"
    exit 1
  fi

  # The checkpoints directory for this speaker
  local NET_DIR="$DATA_DIR/$SPEAKER/checkpoints"

  # Print arguments
  DRAW_DIVIDER;
  printf "Speaker        : $SPEAKER\n"
  printf "Data directory : $DATA_DIR\n"
  printf "Checkpoints    : $NET_DIR\n"
  printf "Epoch for D3D  : $EPOCH_D3D\n"
  printf "Epoch for A2E  : $EPOCH_A2E\n"
  printf "Epoch for NFR  : $EPOCH_NFR\n"

  # Shared arguments
  local SHARED="--data_dir=$DATA_DIR --net_dir=$NET_DIR --speaker=$SPEAKER ${DEBUG}"

  DRAW_DIVIDER;
  PrepareData $SHARED --data_source=celebtalk --epoch=$EPOCH_D3D

  DRAW_DIVIDER;
  TrainAudio2Expression ${SHARED} --epoch=$EPOCH_A2E

  # (Optional) train neural renderer
  if [ -n "${EPOCH_NFR}" ]; then
    DRAW_DIVIDER;
    TrainNeuralFaceRenderer ${SHARED} --epoch=$EPOCH_NFR
  fi

  # * Test
  DRAW_DIVIDER;
  for d in "$DATA_DIR/$SPEAKER/test"/*; do
    if [ ! -d "$d" ]; then continue; fi
    local clip_id="$(basename ${d})"

    TestClip \
      --clip_dir="$d" \
      --reenact_tar="$d" \
      --result_dir="$DATA_DIR/$SPEAKER/results" \
      --result_vpath="$DATA_DIR/$SPEAKER/results/test-$clip_id" \
      --net_dir="$NET_DIR" \
      --epoch_nfr="$EPOCH_NFR" \
    ;
  done
}
