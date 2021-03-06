import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from os import path, listdir
from utils import removeExt

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./input_feats')
parser.add_argument('--input_test_dir', type=str, default='./test_feats')
parser.add_argument('--input_csv', type=str, default='./train.csv')
parser.add_argument('--sub_csv', type=str, default='./sample_submission.csv')
args = parser.parse_args()

root = args.input_dir
root_test = args.input_test_dir
input_csv = args.input_csv

if not path.exists(root) or not path.isdir(root):
    print('Invalid input directory')
if not path.exists(root_test) or not path.isdir(root_test):
    print('Invalid test directory')
if not path.exists(input_csv):
    print('Invalid input csv path')
else:
    class_lookup = np.eye(10)
    input_csv = pd.read_csv(args.input_csv, header=None)
    all_files = listdir(root)
    all_labels = list(input_csv[0])
    X_train = []
    X_test = []
    y_train = []
    for filename in tqdm(all_files):
        full_path = path.join(root, filename)
        item_name = removeExt(filename) + '.mp3'
        if item_name in all_labels:
            item_idx = all_labels.index(item_name)
            class_idx = input_csv[1][item_idx] - 1
            X_train.append(np.load(full_path)['arr_0'].tolist())
            # y_train.append(class_lookup[class_idx])
            y_train.append(class_idx)
    
    all_files = listdir(root_test)
    for filename in tqdm(all_files):
        full_path = path.join(root_test, filename)
        X_test.append(np.load(full_path)['arr_0'].tolist())

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    print('Loaded %s train items' % str(X_train.shape))
    print('Loaded %s test items' % str(X_test.shape))

    # Begin training code
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, 
        test_size = 0.25, random_state = 197)
    
    import lightgbm as lgb
    d_train = lgb.Dataset(X_train, label=y_train)
    d_valid = lgb.Dataset(X_dev, label=y_dev)
    params = {}
    # params['learning_rate'] = 0.01
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'multiclass'
    params['metric'] = 'multi_logloss'
    params['num_class'] = 10
    # params['max_depth'] = 5
    # params['num_leaves'] = 30
    # params['max_bin'] = 128
    # params['metric_freq'] = 20
    # params['is_training_metric'] = True
    params['early_stopping'] = 20
    clf = lgb.train(params, d_train, 20000, valid_sets=[d_valid])

    from sklearn.metrics import accuracy_score
    y_pred = np.round(np.argmax(clf.predict(X_dev), axis=1)).astype(int)
    score = accuracy_score(y_pred, y_dev)
    print('Model score: %s' % score)

    y_probs = clf.predict(X_test)
    y_test = np.round(np.argmax(y_probs, axis=1)).astype(int) + 1
    
    sub = pd.read_csv(args.sub_csv)
    sub[' Genre'] = y_test
    sub.to_csv('lightgbm_output.csv', index=False)

    xgb_proba = pd.DataFrame(y_probs)
    xgb_proba.to_csv('lightgbm_proba.csv', index=False)

    import json
    if path.exists('scores.json'):
        with open('scores.json', 'r') as scores_file:
            scores = json.load(scores_file)
    else:
        scores = {}
    scores['lightgbm_proba'] = score
    with open('scores.json', 'w') as scores_file:
        json.dump(scores, scores_file)


