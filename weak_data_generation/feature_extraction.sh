
#!/bin/sh
# ahocoder feature extraction Set low and high f0 range

# pass the following argument when run this script
## $1 = directory path of the dataset which contains .wav files
## $2 = path where you want to save the features
## $3 = path of an empty folder

wav_directory=$1
feat_directory=$2
empty_folder=$3


# create the directories to store the features
# (AHOCODER returns the 3 features for any wavfile; in this research work, we need only MCC features)
mkdir -p $feat_directory/f0
mkdir -p $feat_directory/mcc
mkdir -p $feat_directory/fv


# extract features
for entry in `ls $wav_directory/*.wav`; do

    # echo $entry
    fname=`basename $entry .wav`
    echo $fname

    # convert and save wav file to 16k hz sampling frequency
    sox -r 16000 -b 16 -c 1 $entry $empty_folder/$fname.wav 

    # feaure extraction : f0, mcc, fv 
    ./ahocoder16_64 $empty_folder/$fname.wav $feat_directory/f0/$fname.f0 $feat_directory/mcc/$fname.mcc $feat_directory/fv/$fname.fv

    rm -r $empty_folder/*.wav

done
exit
