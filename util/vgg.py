import os
import requests
import tarfile

import tensorflow as tf

from io import BytesIO

from settings import FILES_DIR
from vgg import vgg_19


VGG_19_CHECKPOINT_URL = 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'
VGG_19_CHECKPOINT_FILENAME = os.path.join(FILES_DIR, 'vgg_19.ckpt')

VGG_STYLE_LAYERS_NAMES = [
    'vgg_19/conv1/conv1_1',
    'vgg_19/conv2/conv2_1',
    'vgg_19/conv3/conv3_1',
    'vgg_19/conv4/conv4_1',
    'vgg_19/conv5/conv5_1',
]
VGG_STYLE_LAYERS_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
VGG_CONTENT_LAYER_NAME = 'vgg_19/conv4/conv4_2'


def maybe_download_checkpoint():
    if not os.path.exists(VGG_19_CHECKPOINT_FILENAME):
        print(f'Checkpoint does not exist. Download from: {VGG_19_CHECKPOINT_URL}')
        response = requests.get(VGG_19_CHECKPOINT_URL)

        print(f'Extract checkpoint into {FILES_DIR}')
        with tarfile.open(fileobj=BytesIO(response.content)) as tar:
            tar.extractall(FILES_DIR)


def get_layers(inputs, reuse_variables):
    _, end_points = vgg_19(inputs, num_classes=None, reuse=reuse_variables)
    content_layer = end_points[VGG_CONTENT_LAYER_NAME]
    style_layers = [end_points[name] for name in VGG_STYLE_LAYERS_NAMES]
    return content_layer, style_layers


def get_layers_values(image, layer_names, reuse_variables):
    maybe_download_checkpoint()

    inputs = tf.expand_dims(tf.constant(image, tf.float32), 0)
    _, end_points = vgg_19(inputs, num_classes=None, reuse=reuse_variables)

    layers = [end_points[name] for name in layer_names]

    saver = tf.train.Saver(tf.get_collection('model_variables'))
    with tf.Session() as sess:
        saver.restore(sess, VGG_19_CHECKPOINT_FILENAME)
        layers_values = sess.run(layers)

        return layers_values


def get_style_layers_targets(style_image, reuse_variables):
    return get_layers_values(style_image, VGG_STYLE_LAYERS_NAMES, reuse_variables)


def gram_matrix(x, M, N):
    F = tf.reshape(x, (M, N))
    return tf.matmul(tf.transpose(F), F)


def style_layer_loss(a, x):
    _, h, w, d = x.shape
    M = h.value * w.value
    N = d.value
    K = 1. / (4. * N ** 2 * M ** 2)

    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    return K * tf.reduce_sum(tf.pow(G - A, 2))


def get_style_loss(style_layers, style_layers_targets):
    loss = 0
    for layer, value, weight in zip(style_layers, style_layers_targets, VGG_STYLE_LAYERS_WEIGHTS):
        loss += style_layer_loss(value, layer) * weight
    loss /= float(len(style_layers))
    return loss


def get_content_layer_target(content_image, reuse_variables):
    return get_layers_values(content_image, [VGG_CONTENT_LAYER_NAME], reuse_variables)[0]


def get_content_loss(content_layer, content_layer_target):
    _, h, w, d = content_layer.shape
    M = h.value * w.value
    N = d.value
    K = 1. / (2. * N ** 0.5 * M ** 0.5)

    return K * tf.reduce_sum(tf.pow(content_layer - content_layer_target, 2))