import argparse
import pandas as pd
from tqdm import tqdm
from os import path, listdir, makedirs
import shutil
from utils import removeExt

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./input_img')
parser.add_argument('--input_csv', type=str, default='./train.csv')
args = parser.parse_args()

root = args.input_dir
input_csv = args.input_csv

if not path.exists(root) or not path.isdir(root):
    print('Invalid input directory')
if not path.exists(input_csv):
    print('Invalid input csv path')
else:
    input_csv = pd.read_csv(args.input_csv, header=None)
    all_files = listdir(root)
    all_labels = list(input_csv[0])
    for filename in tqdm(all_files):
        full_path = path.join(root, filename)
        item_name = removeExt(filename) + '.mp3'
        if item_name in all_labels:
            move_idx = all_labels.index(item_name)
            dest_name = str(input_csv[1][move_idx])
            dest_path = path.join(root, dest_name)

            if not path.exists(dest_path):
                makedirs(dest_path)
            
            shutil.move(full_path, path.join(dest_path, filename))