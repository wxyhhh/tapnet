
# Testing script for TapNet in 30 multivariate dataset
Parameters:
 --use_lstm 
 --use_cnn 
 --lr 1e-5
 --use_rp False
 --rp_params 2,1
 

<!-- 
1. 
ArticularyWordRecognition
AtrialFibrillation
BasicMotions
CharacterTrajectories
Cricket
-->
python3 train.py --dataset ArticularyWordRecognition
python3 train.py --dataset AtrialFibrillation
python3 train.py --dataset BasicMotions
python3 train.py --dataset CharacterTrajectories
python3 train.py --dataset Cricket

<!--
2. 
DuckDuckGeese
EigenWorms
Epilepsy
ERing
EthanolConcentration
-->
# python3 train.py --dataset DuckDuckGeese --filters 16,8,8
python3 train.py --dataset EigenWorms
python3 train.py --dataset Epilepsy
python3 train.py --dataset ERing
python3 train.py --dataset EthanolConcentration

<!--
3. 
FaceDetection
FingerMovements
HandMovementDirection
Handwriting
Heartbeat
-->
# python3 train.py --dataset FaceDetection --filters 16,8,8
python3 train.py --dataset FingerMovements
python3 train.py --dataset HandMovementDirection
python3 train.py --dataset Handwriting
python3 train.py --dataset Heartbeat

<!--
4. 
InsectWingbeat
JapaneseVowels
Libras
LSST
MotorImagery
-->
# python3 train.py --dataset InsectWingbeat --filters 16,8,8
python3 train.py --dataset JapaneseVowels
python3 train.py --dataset Libras
python3 train.py --dataset LSST
python3 train.py --dataset MotorImagery

<!--
5.
NATOPS
PEMS-SF
PenDigits
Phoneme
RacketSports
-->
python3 train.py --dataset NATOPS
python3 train.py --dataset PEMS-SF
# python3 train.py --dataset PenDigits --kernels 3,2,1
python3 train.py --dataset Phoneme
python3 train.py --dataset RacketSports

<!--
6.
SelfRegulationSCP1
SelfRegulationSCP2
SpokenArabicDigits
StandWalkJump
UWaveGestureLibrary
-->
python3 train.py --dataset SelfRegulationSCP1
python3 train.py --dataset SelfRegulationSCP2
python3 train.py --dataset SpokenArabicDigits
python3 train.py --dataset StandWalkJump
python3 train.py --dataset UWaveGestureLibrary

# ============== Semi-TapNet ===================
python3 train.py --dataset ArticularyWordRecognition --dilation 10 --use_ss
python3 train.py --dataset AtrialFibrillation --use_ss
python3 train.py --dataset EigenWorms --dilation 200 --use_ss
python3 train.py --dataset Epilepsy --use_ss
python3 train.py --dataset Handwriting --use_ss
python3 train.py --dataset Heartbeat --lr 1e-6 --use_ss
python3 train.py --dataset JapaneseVowels --use_ss
python3 train.py --dataset RacketSports --use_ss
python3 train.py --dataset StandWalkJump --rp_params 3,1 --use_ss
python3 train.py --dataset UWaveGestureLibrary --use_ss
# ============================================================================================================================
# Run lstmfcn
<!--
ArticularyWordRecognition
AtrialFibrillation
BasicMotions
CharacterTrajectories
Cricket
-->
python3 mlstam_classifer.py --dataset ArticularyWordRecognition
python3 mlstam_classifer.py --dataset AtrialFibrillation
python3 mlstam_classifer.py --dataset BasicMotions
python3 mlstam_classifer.py --dataset CharacterTrajectories
python3 mlstam_classifer.py --dataset Cricket

<!--
DuckDuckGeese
EigenWorms
Epilepsy
ERing
EthanolConcentration
-->
# python3 mlstam_classifer.py --dataset DuckDuckGeese
python3 mlstam_classifer.py --dataset EigenWorms
python3 mlstam_classifer.py --dataset Epilepsy
python3 mlstam_classifer.py --dataset ERing
python3 mlstam_classifer.py --dataset EthanolConcentration

<!--
FaceDetection
FingerMovements
HandMovementDirection
Handwriting
Heartbeat
-->
# python3 mlstam_classifer.py --dataset FaceDetection
python3 mlstam_classifer.py --dataset FingerMovements
python3 mlstam_classifer.py --dataset HandMovementDirection
python3 mlstam_classifer.py --dataset Handwriting
python3 mlstam_classifer.py --dataset Heartbeat

<!--
InsectWingbeat
JapaneseVowels
Libras
LSST
MotorImagery
-->
# python3 mlstam_classifer.py --dataset InsectWingbeat
python3 mlstam_classifer.py --dataset JapaneseVowels
python3 mlstam_classifer.py --dataset Libras
python3 mlstam_classifer.py --dataset LSST
python3 mlstam_classifer.py --dataset MotorImagery

<!--
NATOPS
PEMS-SF
PenDigits
Phoneme
RacketSports
-->
python3 mlstam_classifer.py --dataset NATOPS
# python3 mlstam_classifer.py --dataset PEMS-SF
python3 mlstam_classifer.py --dataset PenDigits
python3 mlstam_classifer.py --dataset Phoneme
python3 mlstam_classifer.py --dataset RacketSports

<!--
SelfRegulationSCP1
SelfRegulationSCP2
SpokenArabicDigits
StandWalkJump
UWaveGestureLibrary
-->
python3 mlstam_classifer.py --dataset SelfRegulationSCP1
python3 mlstam_classifer.py --dataset SelfRegulationSCP2
python3 mlstam_classifer.py --dataset SpokenArabicDigits
python3 mlstam_classifer.py --dataset StandWalkJump
python3 mlstam_classifer.py --dataset UWaveGestureLibrary

# ================
nohup ./run_muse.sh FaceDetection 100 & # 0.503
nohup ./run_muse.sh FaceDetection 50 & # 0.54499
nohup ./run_muse.sh FaceDetection 30 & 0.54499
nohup ./run_muse.sh FaceDetection 10 & 0.556

nohup ./run_muse.sh HandMovementDirection 1 & # 0.365
nohup ./run_muse.sh Heartbeat 1 & # 0.727
nohup ./run_muse.sh PEMS-SF 1 &
nohup ./run_muse.sh SpokenArabicDigits 1 & # 0.982
nohup ./run_muse.sh Phoneme 1 &