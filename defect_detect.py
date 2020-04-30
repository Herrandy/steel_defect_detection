#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from utils import mask2contour
from keras import backend as K
import segmentation_models as sm
from keras.callbacks import ModelCheckpoint

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def get_model():
    preprocess = sm.get_preprocessing('resnet34')
    model = sm.Unet('resnet34', input_shape=(128, 800, 3), classes=4, activation='sigmoid')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    return model, preprocess

def load_data(path):
    df = pd.read_csv(path + 'train.csv')
    df = df[df['EncodedPixels'].notna()]  # remove possible NaNs
    data = {}
    for index, row in df.iterrows():
        imgID = row['ImageId']
        if imgID in data:
            data[imgID]['defectIDs'].append(row['ClassId'] - 1)
            data[imgID]['masks'].append(row['EncodedPixels'])
        else:
            dic = {'defectIDs': [row['ClassId'] - 1], 'masks': [row['EncodedPixels']]}
            data[imgID] = dic

    data = np.array(list(data.items()))
    return data

def visualize_data(path):
    filenames = {}
    data = load_data(path)

    dg = DataGenerator(np.array(data), path, batch_size=16, info=filenames)
    for batch_idx, (X, Y) in enumerate(dg):  # loop batches one by one
        fig = plt.figure(figsize=(16, 25))
        for idx, (img, masks) in enumerate(zip(X, Y)):  # loop of images
            for m in range(4):  # loop different defects
                mask = masks[:, :, m]
                mask = mask2contour(mask, width=2)
                if m == 0:  # yellow
                    img[mask == 1, 0] = 235
                    img[mask == 1, 1] = 235
                elif m == 1:
                    img[mask == 1, 1] = 210  # green
                elif m == 2:
                    img[mask == 1, 2] = 255  # blue
                elif m == 3:  # magenta
                    img[mask == 1, 0] = 255
                    img[mask == 1, 2] = 255
            plt.axis('off')
            fig.add_subplot(8, 2, idx+1)
            plt.imshow(img/255.0)
            plt.title(filenames[16 * batch_idx + idx])
        plt.show()

def train(path, save_file):
    filenames = {}
    data = load_data(path)
    model, preprocess = get_model()
    idx = int(0.8 * len(data))
    train_batches = DataGenerator(data[:idx], path, shuffle=True, preprocess=preprocess, info=filenames)
    valid_batches = DataGenerator(data[idx:], path, preprocess=preprocess, info=filenames)


    '''
    checkpoint = ModelCheckpoint(save_file, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit_generator(train_batches, validation_data=valid_batches, callbacks=callbacks_list, epochs=5, verbose=2)
    '''
    history = model.fit_generator(train_batches, validation_data=valid_batches, epochs=5, verbose=1)
    model.save(save_file)

def eval(data_path, model_weights):
    model, preprocess = get_model()
    model.load_weights(model_weights)
    data = load_data(data_path)
    idx = int(0.8 * len(data))
    filenames = {}
    valid_batches = DataGenerator(data[idx:], data_path, preprocess=preprocess, info=filenames)
    preds = model.predict_generator(valid_batches, verbose=1)

    for batch_idx, start_idx in enumerate(range(len(preds))[::16]):
        fig = plt.figure(figsize=(16, 25))
        (X, Y) = valid_batches[batch_idx]
        for idx, pred in enumerate(preds[start_idx:start_idx+8]):
            mask = np.zeros((pred.shape[0], pred.shape[1]))
            for m in range(4):
                mask += pred[:, :, m]
            plt.axis('off')
            fig.add_subplot(8, 2, (idx * 2) + 1)
            plt.imshow(mask)
            plt.title(filenames[16 * batch_idx + idx])
            fig.add_subplot(8, 2, (idx * 2) + 2)
            plt.imshow(X[idx, :, :, :] / 255.0)
            plt.title(filenames[16 * batch_idx + idx])
        plt.show()

if __name__== "__main__":

    data_path = '/mnt/sda/data/severstal-steel-defect-detection/'
    save_file = '/home/antti/work/surface_defect/severstal/saved_models/model.h5'
    model_weights = '/home/antti/work/surface_defect/severstal/saved_models/model.h5'
    # visualize_data(data_path)
    eval(data_path, model_weights)
    # train(data_path, save_file)
