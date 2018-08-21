import librosa
import argparse
import numpy as np
from tqdm import tqdm
from os import path, listdir
from utils import removeExt, toNp

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./input')
parser.add_argument('--output_dir', type=str, default='./input_np')

args = parser.parse_args()

root = args.input_dir
output = args.output_dir
if not path.exists(root) or not path.isdir(root):
    print('Invalid input directory')
elif not path.exists(output) or not path.isdir(output):
    print('Invallid output directory')
else:
    for filename in tqdm(listdir(root)):
        if removeExt(filename) == '':
            continue

        full_path = path.join(root, filename)
        out_file = path.join(output, removeExt(filename) + '.npz')

        if path.exists(out_file):
            continue
        
        np.savez_compressed(out_file, arr=toNp(full_path))
        