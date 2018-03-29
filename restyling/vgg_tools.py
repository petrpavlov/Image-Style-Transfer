import os
import requests
import tarfile

import numpy as np
import tensorflow as tf

from io import BytesIO

from settings import FILES_DIR
from .vgg import vgg_19


VGG_19_CHECKPOINT_URL = 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'
VGG_19_CHECKPOINT_FILENAME = os.path.join(FILES_DIR, 'vgg_19.ckpt')

MEAN_PIXEL = np.array([123.68, 116.779, 103.939])


def maybe_download_checkpoint():
    if not os.path.exists(VGG_19_CHECKPOINT_FILENAME):
        print(f'Checkpoint does not exist. Download from: {VGG_19_CHECKPOINT_URL}')
        response = requests.get(VGG_19_CHECKPOINT_URL)

        print(f'Extract checkpoint into {FILES_DIR}')
        with tarfile.open(fileobj=BytesIO(response.content)) as tar:
            tar.extractall(FILES_DIR)


def get_layers(inputs, reuse_variables):
    _, layers = vgg_19(inputs, num_classes=None, reuse=reuse_variables)
    return layers


def get_layers_values(image, layer_names, reuse_variables):
    inputs = tf.expand_dims(tf.constant(image, tf.float32), 0)
    _, end_points = vgg_19(inputs, num_classes=None, reuse=reuse_variables)

    layers = [end_points[name] for name in layer_names]

    saver = tf.train.Saver(tf.get_collection('model_variables'))
    with tf.Session() as sess:
        saver.restore(sess, VGG_19_CHECKPOINT_FILENAME)
        layers_values = sess.run(layers)

        return layers_values


def pre_process(image):
    return image - MEAN_PIXEL


def post_process(image):
    return image + MEAN_PIXEL
