import numpy as np
import keras
from utils import rle2mask
import matplotlib.pyplot as plt

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, path, img_size=(128, 800), batch_size=16, subset="train", shuffle=False,
                 preprocess=None, info={}):
        super().__init__()
        self.data = data
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        self.path = path
        self.img_size = img_size

        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # (batch size, image height, image width, number of channels (RGB=3))
        X = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        # (batch size, image height, image width, number of classes (one hot coded))
        Y = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 4), dtype=np.int8)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        for idx, (imgID, masks) in enumerate(self.data[indexes]):
            self.info[index * self.batch_size + idx] = imgID
            X[idx, ] = plt.imread(self.data_path + imgID)[::2, ::2]
            if self.subset == 'train':
                defectsIDs = masks['defectIDs']
                masks = masks['masks']
                for m in range(len(defectsIDs)):
                    Y[idx, :, :, defectsIDs[m]] = rle2mask(masks[m])[::2, ::2]
        if self.preprocess != None: X = self.preprocess(X)
        if self.subset == 'train':
            return X, Y
        else:
            return X