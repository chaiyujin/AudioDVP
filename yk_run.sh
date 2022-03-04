source yk_functions.sh

# RUN_VOCASET --speaker=FaceTalk_170725_00137_TA --epoch_nfr=200;
# RUN_VOCASET --speaker=FaceTalk_170908_03277_TA --epoch_nfr=200;
# RUN_VOCASET --speaker=FaceTalk_170908_03277_TA --epoch_nfr=200;
# RUN_VOCASET --epoch_nfr=200 --speaker=FaceTalk_170811_03275_TA --debug
# RUN_VOCASET --epoch_nfr=200 --speaker=FaceTalk_170904_03276_TA
# RUN_VOCASET --epoch_nfr=200 --speaker=FaceTalk_170908_03277_TA --debug
# RUN_VOCASET --epoch_nfr=200 --speaker=FaceTalk_170913_03279_TA
# RUN_VOCASET --epoch_d3d=1 --epoch_a2e=10 --epoch_nfr=25 --speaker=FaceTalk_170904_03276_TA --debug
# RUN_VOCASET --epoch_d3d=20 --epoch_a2e=40 --epoch_nfr='' --speaker=FaceTalk_170908_03277_TA --debug

# RUN_YK_EXP --data_src=celebtalk --epoch_d3d=60 --epoch_a2e=60 --epoch_nfr=80 --speaker=m000_obama  # --debug

source nohup_run.sh

# CUDA_VISIBLE_DEVICES=1 \
# NOHUP_RUN --device=1 --include=yk_functions.sh -- \
# RUN_YK_EXP --data_src=celebtalk --epoch_d3d=100 --epoch_a2e=100 --speaker=m001_trump --use_seqs="trn-000,trn-001,vld-000,vld-001";
