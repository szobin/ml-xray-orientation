# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
import numpy as np
import logging
import os
from .settings import MODEL_DIR, TRAIN_DIR, VAL_DIR, TYPES, IMG_WIDTH, IMG_HEIGHT


logger = logging.getLogger(__name__)


def load_images(path, mode='file', grayscale=False, files_extension='.png'):
    _images, _labels = image_preloader(path, image_shape=(IMG_WIDTH, IMG_HEIGHT), mode=mode,
                                       categorical_labels=True, normalize=True, files_extension=files_extension,
                                       grayscale=grayscale)
    return _images, _labels


def load_dataset(folder):
    return load_images(folder, mode="folder")


def get_class_str(p):
    dp = p[1] - p[0]
    if abs(dp) > 0.3:
        if dp < 0:
            return "class: {} [{} / {}]".format(TYPES[0]["name"], p[0], p[1])
        else:
            return "class: {} [{} / {}]".format(TYPES[1]["name"], p[1], p[0])
    else:
        if dp < 0:
            return "probably class: {} [{} / {}]".format(TYPES[0]["name"], p[0], p[1])
        else:
            return "probably class: {} [{} / {}]".format(TYPES[1]["name"], p[1], p[0])



class Classifier:

    def __init__(self, model_run_id='xray-classifier', n_epoch=100):
        self.model_run_id = model_run_id
        self.n_epoch = n_epoch
        self.data_set = None

        tf.logging.set_verbosity(tf.logging.ERROR)

        # Make sure the data is normalized
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

        # Create extra synthetic training data by flipping, rotating and blurring the
        # images on our data set.
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)
        img_aug.add_random_blur(sigma_max=3.)

        # Define our network architecture:

        # Input is a 32x32 image with 3 color channels (red, green and blue)
        self.network = input_data(shape=[None, IMG_WIDTH, IMG_HEIGHT, 1],
                                  data_preprocessing=img_prep,
                                  data_augmentation=img_aug)

        # Step 1: Convolution
        self.network = conv_2d(self.network, IMG_WIDTH, 3, activation='relu')

        # Step 2: Max pooling
        self.network = max_pool_2d(self.network, 2)

        # Step 3: Convolution again
        self.network = conv_2d(self.network, 2 * IMG_WIDTH, 3, activation='relu')

        # Step 4: Convolution yet again
        self.network = conv_2d(self.network, 2 * IMG_WIDTH, 3, activation='relu')

        # Step 5: Max pooling again
        self.network = max_pool_2d(self.network, 2)

        # Step 6: Fully-connected 512 node neural network
        self.network = fully_connected(self.network, 512, activation='relu')

        # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
        self.network = dropout(self.network, 0.5)

        # Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final
        # prediction
        self.network = fully_connected(self.network, 2, activation='softmax')

        # Tell tflearn how we want to train the network
        self.network = regression(self.network, optimizer='adam',
                                  loss='categorical_crossentropy',
                                  learning_rate=0.001)

        # Wrap the network in a model object
        self.model = tflearn.DNN(self.network, tensorboard_verbose=0, checkpoint_path='xray-classifier.tfl.ckpt')

    def fit(self, n_epoch=100, train_dir=TRAIN_DIR, val_dir=VAL_DIR):
        images, labels_ = load_dataset(train_dir)
        x = np.reshape(images, (-1, IMG_WIDTH, IMG_HEIGHT, 1))

        images_test, labels_test = load_dataset(val_dir)
        x_test = np.reshape(images_test, (-1, IMG_WIDTH, IMG_HEIGHT, 1))

        n_samples = x.shape[0]

        self.model.fit(x, labels_, n_epoch=n_epoch, shuffle=True, validation_set=(x_test, labels_test),
                       show_metric=True, batch_size=n_samples,
                       snapshot_epoch=False,
                       run_id=self.model_run_id)

        model_fn = os.path.join(MODEL_DIR, "{}.tfl".format(self.model_run_id))
        self.model.save(model_fn)
        print("Network trained and saved as: {}".format(model_fn))

    def predict(self, image_arg):
        model_fn = os.path.join(MODEL_DIR, "{}.tfl".format(self.model_run_id))
        if os.path.isfile(model_fn+'.index'):
            logger.debug('model was found, loading from {}'.format(model_fn))
            self.model.load(model_fn)
        else:
            logger.debug('model was not found, network training starting ...')
            self.fit(n_epoch=self.n_epoch)

        if image_arg.endswith(".png"):
            images, _labels = load_images(image_arg)
        elif os.path.isdir(image_arg):
            images, _labels = load_images(image_arg, mode="folder")
        else:
            print("Unrecognized image argument: {}".format(image_arg))
            return None

        image_filenames = list(images.array)
        x = np.reshape(images, (-1, IMG_WIDTH, IMG_HEIGHT, 1))

        prediction = self.model.predict(x)
        resp = list()
        for f, p in zip(image_filenames, prediction):
            resp.append([f, p])
        return resp

    @staticmethod
    def print_prediction(prediction):
        print("prediction result:")
        print("=====================")
        for p in prediction:
            print("{} - {}".format(p[0], get_class_str(p[1])))
        print("=====================")
