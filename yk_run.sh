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


RUN_CELEBTALK --epoch_d3d=100 --epoch_a2e=40 --epoch_nfr=100 --speaker=m000_obama  # --debug
# RUN_VOCASET   --epoch_d3d=40 --epoch_a2e=40 --epoch_nfr=250 --speaker=FaceTalk_170908_03277_TA  # --debug
