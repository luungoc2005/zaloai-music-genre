import librosa
import librosa.display
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from os import path, listdir
from utils import removeExt, sliding_window

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
        out_file = path.join(output, removeExt(filename))
        y = np.load(full_path)['arr']
        
        # https://arxiv.org/pdf/1804.01149.pdf
        # central moments
        c_mean = np.mean(y)
        c_std = np.std(y)
        c_skew = skew(y)
        c_kurtosis = kurtosis(y)

        # zcr, rmse
        zcr_list = []
        rmse_list = []
        for window in sliding_window(y, 2048, 512):
            zcr = librosa.zero_crossings(window)
            zcr_list.append(np.sum(zcr))
            rmse_list.append(np.sqrt(np.mean(window ** 2)))
        
        zcr_mean = np.mean(zcr_list)
        zcr_std = np.std(zcr_list)

        rmse_mean = np.mean(rmse_list)
        rmse_std = np.std(rmse_list)

        # tempo
        onset_env = librosa.onset.onset_strength(y)
        tempo = librosa.beat.tempo(onset_envelope=onset_env)

        # mfcc
        mfcc = librosa.feature.mfcc(y, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # chroma_cqt
        chroma_cq = librosa.feature.chroma_cqt(y)
        chroma_mean = np.mean(chroma_cq, axis=1)
        chroma_std = np.std(chroma_cq, axis=1)

        # spectral_centroid
        spectral_centroid = librosa.feature.spectral_centroid(y, n_fft=2048, hop_length=512)[0]
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)

        # spectral_bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y, n_fft=2048, hop_length=512)[0]
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_std = np.std(spectral_bandwidth)

        # spectral_contrast
        spectral_contrast = librosa.feature.spectral_contrast(y, n_fft=2048, hop_length=512)[0]
        spectral_contrast_mean = np.mean(spectral_contrast)
        spectral_contrast_std = np.std(spectral_contrast)

        # spectral_rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y, n_fft=2048, hop_length=512, roll_percent=0.85)[0]
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_std = np.std(spectral_rolloff)

        feats = np.array([
            c_mean, c_std, c_skew, c_kurtosis, 
            zcr_mean, zcr_std, 
            rmse_mean, rmse_std, 
            tempo,
            *mfcc_mean, *mfcc_std,
            *chroma_mean, *chroma_std,
            spectral_centroid_mean, spectral_centroid_std,
            spectral_bandwidth_mean, spectral_bandwidth_std,
            spectral_contrast_mean, spectral_contrast_std,
            spectral_rolloff_mean, spectral_rolloff_std
        ], dtype='float32')
        # print(feats.shape) # 81 features

        np.save(out_file, feats)



