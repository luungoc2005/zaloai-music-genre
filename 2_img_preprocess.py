import librosa
import librosa.display
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import path, listdir
from utils import removeExt

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./input_np')
parser.add_argument('--output_dir', type=str, default='./input_img')

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
        out_file = path.join(output, removeExt(filename) + '.jpg')
        y = np.load(full_path)['arr']
        D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
            sr=22050, y_axis='mel', x_axis='time')
        plt.axis('off')
        plt.savefig(out_file, bbox_inches='tight')