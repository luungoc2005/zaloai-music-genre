import keras
import argparse
import pandas as pd
import numpy as np
import math
from os import path, listdir
from utils import removeExt
from keras_preprocessing.image import ImageDataGenerator
from keras import applications
from keras import optimizers

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./input_img')
parser.add_argument('--input_test_dir', type=str, default='./test_img')
parser.add_argument('--input_csv', type=str, default='./train.csv')
parser.add_argument('--sub_csv', type=str, default='./sample_submission.csv')
parser.add_argument('--batch_size', type=int, default=8)
args = parser.parse_args()

root = args.input_dir
root_test = args.input_test_dir
input_csv = args.input_csv
batch_size = args.batch_size

if not path.exists(root) or not path.isdir(root):
    print('Invalid input directory')
if not path.exists(root_test) or not path.isdir(root_test):
    print('Invalid test directory')
if not path.exists(input_csv):
    print('Invalid input csv path')
else:
    class_lookup = np.eye(10)
    input_csv = pd.read_csv(args.input_csv, header=None)
    sub = pd.read_csv(args.sub_csv)
    
    nb_train_samples = len(input_csv[0])
    nb_test_samples = len(sub['Id'])

    datagen = ImageDataGenerator(rescale=1./255)

    model_name = 'vgg16'

    train_generator = datagen.flow_from_directory(
        root,  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    label_map = dict((v, k) for k, v in train_generator.class_indices.items())
    
    print(label_map)
    
    test_generator = datagen.flow_from_directory(
        root_test,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    model = applications.VGG16(classes=10, weights=None)
    
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced', np.unique(input_csv[1]), input_csv[1])

    from keras.callbacks import EarlyStopping

    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(train_generator,
        steps_per_epoch=math.ceil(nb_train_samples / batch_size),
        epochs=100,
        callbacks=[
            EarlyStopping(monitor='loss', patience=5, verbose=1)
        ],
        class_weight=class_weights
    )
    model.save_weights(model_name + 'best_model.h5')
    best_score = history.history['acc'][-1]
    print('Model score %s' % best_score)
    
    y_probs = model.predict_generator(
        test_generator, math.ceil(nb_test_samples / batch_size), verbose=1)
    y_test = np.round(np.argmax(y_probs, axis=1)).astype(int)
    y_test = np.array([int(label_map[str(label)]) for label in y_test])
    
    sub[' Genre'] = y_test
    sub.to_csv(model_name + '_output.csv', index=False)

    res_proba = pd.DataFrame(y_probs)
    res_proba.to_csv(model_name + '_proba.csv', index=False)
    
    import json
    if path.exists('scores.json'):
        with open('scores.json', 'r') as scores_file:
            scores = json.load(scores_file)
    else:
        scores = {}
    scores[model_name + '_proba'] = best_score
    with open('scores.json', 'w') as scores_file:
        json.dump(scores, scores_file)
