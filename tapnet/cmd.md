
python3 train.py --dataset PenDigits --lr 1e-5 --wd 1e-3 --layers 2000,300 --epoch 3000

python3 train.py --dataset SelfRegulationSCP1 --lr 1e-4 --wd 0.1 --layers 2000,300 --epoch 30000

python3 train.py --dataset SpokenArabicDigits --lr 1e-4 --wd 0.1 --layers 2000,300 --epoch 1000

python3 train.py --dataset PenDigits --lr 1e-4 --wd 0.1 --layers 2000,300 --epoch 3000


# ========= raw data =========
python3 train.py --dataset NATOPS --lr 5e-5 --wd 1e-3 --layers 2000,300 --use_ss --use_metric
python3 train.py --dataset UWaveGestureLibrary --lr 1e-5 --wd 1e-3 --layers 2000,300 --use_raw
python3 train.py --dataset Handwriting --lr 1e-5 --wd 1e-3 --layers 2000,300 --use_raw --use_ss
Handwriting
PenDigits

# pure test

<!--
ArticularyWordRecognition
AtrialFibrillation
BasicMotions
CharacterTrajectories
Cricket
-->
python3 train.py --dataset ArticularyWordRecognition --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset AtrialFibrillation --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset BasicMotions --use_lstm --use_cnn --lr 1e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset CharacterTrajectories --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset Cricket --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300

<!--
DuckDuckGeese
EigenWorms
Epilepsy
ERing
EthanolConcentration
-->
python3 train.py --dataset DuckDuckGeese --use_lstm --use_cnn --lr 5e-6 --wd 1e-3 --layers 1000,300
python3 train.py --dataset EigenWorms --use_lstm --lr 1e-5 --wd 1e-3 --layers 1000,300  # outofmemory, so use lstm only
python3 train.py --dataset Epilepsy --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset ERing --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset EthanolConcentration --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300

<!--
FaceDetection
FingerMovements
HandMovementDirection
Handwriting
Heartbeat
-->
python3 train.py --dataset FaceDetection --use_lstm --use_cnn --lr 1e-5 --wd 1e-3 --layers 1000,300 --filters 16,8,8
python3 train.py --dataset FingerMovements --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset HandMovementDirection --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset Handwriting --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset Heartbeat --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300

<!--
InsectWingbeat
JapaneseVowels
Libras
LSST
MotorImagery
-->
python3 train.py --dataset InsectWingbeat --use_lstm --use_cnn --lr 1e-4 --wd 1e-3 --layers 1000,300 --filters 16,8,8
python3 train.py --dataset JapaneseVowels --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset Libras --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset LSST --use_lstm --use_cnn --lr 5e-4 --wd 1e-3 --layers 1000,300
python3 train.py --dataset MotorImagery --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
<!--
NATOPS
PEMS-SF
PenDigits
Phoneme
RacketSports
-->
python3 train.py --dataset NATOPS --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset PEMS-SF --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset PenDigits --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300 --kernels 3,2,1
python3 train.py --dataset Phoneme --use_lstm --use_cnn --lr 5e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset RacketSports --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300

<!--
SelfRegulationSCP1
SelfRegulationSCP2
SpokenArabicDigits
StandWalkJump
UWaveGestureLibrary
-->
python3 train.py --dataset SelfRegulationSCP1 --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset SelfRegulationSCP2 --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset SpokenArabicDigits --use_lstm --use_cnn --lr 5e-5 --wd 1e-3 --layers 1000,300
python3 train.py --dataset StandWalkJump --use_lstm --use_cnn --lr 5e-6 --wd 1e-3 --layers 1000,300
python3 train.py --dataset UWaveGestureLibrary --use_lstm --use_cnn --lr 2e-5 --wd 1e-3 --layers 1000,300


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