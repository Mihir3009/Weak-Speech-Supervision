# Weak-Speech-Supervision
A first-of-its-kind system in speech domain that helps users to train state-of-the-art classification models without hand-labeling training data. This research work "Weak Speech Supervision: A case study of Dysarthria Severity Classification" is pubished in the EUSIPCO-2020 conference.

## Table of contents
1. [Introduction](#intro)
2. [Citation](#cite)
3. [Implementation](#implement)
4. [Conclusion](#summary)

<a name="intro"></a>
## Introduction

A huge amount of labeled data is the largest bottleneck in the deployment of deep learning-based classification systems including speech area. It is know that hand-labeling training data is an intensively laborious task even for the speech domain. To overcome this problems, we introduce a new paradigm "Weak Speech Supervision" first time in speech domain to produce a labeled data without user intervention, and utilize this data to improve perfomance of classifiers in various speech domain.

<a name="cite"></a>
## Citation

If you use this implementation idea in your research or industrial work, please cite:

```
@article{WSS2020,
	title = {Weak Speech Supervision: A case study of Dysarthria Severity Classification},
	author = {Mirali Purohit and Mihir Parmar and Maitreya Patel and Harshit Malaviya and Hemant A. Patil},
	booktitle={$$28^{th}$$ European Signal Processing Conference (EUSIPCO)},
	year = {2020},
	Address = {Amsterdam, Netherlands}
}
```
<a name="implement"></a>
## Implementation

### QuickStart

To use the implemented Weak Speech Supervision Model model, please use following cmd to clone the code:

```
$ git clone https://github.com/Mihir3009/Weak-Speech-Supervision
```

### Setup Environment

1. python3.6 <br /> Reference to download and install : https://www.python.org/downloads/release/python-360/
2. Install requirements <br /> 
```
$ pip3 install -r requirements.txt
```

For implementation follow the following 4 steps:

### Generate weak label
To generate weak labels for your unlabeled data, you need to write weak rules. In this reserch work, we have used energy-based parameter to label the unlabeled dysarthric wav files. You can use our predefine weak rule.

1. To generate weak labels for the unlabeled dataset -
```
$ python3 energy_extractor.py -wave_file_path ../path/of/the/unlabeled/dataset/wavefiles
```

### Feature extraction
In training and testing, Mel Cepstral Coefficient (MCC) are used as features for original (i.e., labeled) as well as weakly labeled data. To extract features (i.e., MCC) from wavfile, [AHOCODER](https://aholab.ehu.eus/ahocoder/info.html) is used. For feature extraction, download the .exe/executable file of AHOCODER from [here](https://aholab.ehu.eus/ahocoder/info.html).

2. To extract the feature from the labeled and weakly generated data, run the following script:
```
$ ./feature_extraction.sh ../path/of/the/wavfile/folder ../path/where/to/save/the/features ../path/of/any/empty/directory
```

### Training
In the training, we utilize the weakly labeled data for the severity-based binary classification of dysarthric speech. You can see schematic representation of our training procedure below.

![weak_figure_final_4-Page-7](https://user-images.githubusercontent.com/47143544/92290254-567b1c00-eec8-11ea-9ddf-048a47aa8486.jpeg)

3. For training run the following command -
```
$ python3 main.py --data_dir ../path/of/the/training_data --output_dir ../path/where/to/save/the/checkpoint --do_train
```

### Testing
4. For testing the model, run the following command -
```
$ python3 main.py --data_dir ..path/of/the/testing_data --output_dir ../path/where/to/save/the/checkpoint --do_test
```

<a name="summary"></a>
## Conclusion

For more, you can read the blog - 

Also, you can refer our publication in EUSIPCO 2020 by on this [link](https://drive.google.com/file/d/1L-UWr23O_sFBI43Pe_-cbJZajHdzWXOo/view).
