
# Libraries
import pandas as pd
import numpy as np
import glob
import argparse
from os import listdir
from os.path import join
from tqdm import trange

# Matlab 
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()


parser= argparse.ArgumentParser()

parser.add_argument("-wave_file_path", default="./wav",
                    type=str, required=True,
                    help="Provide path of folder with .wav files for generate a weak data")

args= parser.parse_args()

wavfile_list = sorted(glob.glob(join(args.wave_file_path,'*.wav')))
ignore_index = len(args.wave_file_path)

result = list()

for index in trange(len(wavfile_list)):
    energy = eng.energy_extractor(wavfile_list[index])
    energy = np.array(energy)
    sum_energy = np.sum(energy)
    threshold = sum_energy/(10**9)

    if threshold < 120:
        weak_label = 'Low-to-Mid'
    else:
        weak_label = 'Mid-to-High'
    
    result.append((wavfile_list[index][ignore_index+1:], weak_label))

df = pd.DataFrame(result, columns=['wavfile', 'weak_label'])
df.to_csv('./weak_labels.csv', index=False)
