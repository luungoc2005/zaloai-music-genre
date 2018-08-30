import keras
import argparse
import pandas as pd
import numpy as np
import math
from os import path, listdir
from utils import removeExt
from keras_preprocessing.image import ImageDataGenerator
from keras import applications

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./input_img')
parser.add_argument('--input_test_dir', type=str, default='./test_img')
parser.add_argument('--input_csv', type=str, default='./train.csv')
parser.add_argument('--sub_csv', type=str, default='./sample_submission.csv')
args = parser.parse_args()

root = args.input_dir
root_test = args.input_test_dir
input_csv = args.input_csv
batch_size = 16

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

    train_generator = datagen.flow_from_directory(
        root,  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    
    model = applications.VGG16(include_top=False, weights='imagenet')
    
    if not path.exists('bottleneck_features_train.npy'):
        bottleneck_features_train = model.predict_generator(
            train_generator, math.ceil(nb_train_samples / batch_size), verbose=1)
        
        np.save('bottleneck_features_train.npy', bottleneck_features_train)
        X_train = bottleneck_features_train
    else:
        X_train = np.load('bottleneck_features_train.npy')

    if not path.exists('bottleneck_features_test.npy'):
        test_generator = datagen.flow_from_directory(
            root_test,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
        bottleneck_features_test = model.predict_generator(
            test_generator, math.ceil(nb_test_samples / batch_size), verbose=1)

        np.save('bottleneck_features_test.npy', bottleneck_features_test)
        X_test = bottleneck_features_test
    else:
        X_test = np.load('bottleneck_features_test.npy')

    lookup = np.eye(10)
    y_train = np.array([lookup[class_idx - 1] for class_idx in input_csv[1]])

    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced', np.unique(input_csv[1]), input_csv[1])

    from keras.models import Sequential
    from keras.layers import Dropout, Flatten, Dense
    from keras.callbacks import EarlyStopping

    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
        loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train,
        epochs=1000,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        ],
        class_weight=class_weights,
        validation_split=0.25
    )
    model.save_weights('best_model.h5')
    print('Model score %s' % history.history['val_acc'][-1])

    y_probs = model.predict(X_test)
    y_test = np.round(np.argmax(y_probs, axis=1)).astype(int)
    
    sub['Genre'] = y_test
    sub.to_csv('vgg_output.csv', index=False)

    vgg_proba = pd.DataFrame(y_probs)
    vgg_proba.to_csv('vgg16_proba.csv', index=False)
