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

To use the implemented GA for Richardson arm race model, please use following cmd to clone the code:

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

### Define weak label:
To define the weak label in this reserch work we have used energy-based parameter:

1. To generate the weak labels for the unlabeled dataset -
```
python3 energy_extractor.py -wave_file_path ../path/of/the/unlabeled/dataset/wavefiles
```

### Feature extraction:
In training and testing, Mel Cepstral Coefficient (MCC) features are used for original as well as weka data. In feature extraction, the MCC from wavfile [AHOCODER](https://aholab.ehu.eus/ahocoder/info.html) is used.

2. To extract the feature from the weak generated data and original data run the following script:
```
$ 
```
