import numpy as np
import tensorflow as tf

from settings import VGG_19_CHECKPOINT_FILENAME, VGG_19_CONTENT_LAYER_NAME, VGG_19_STYLE_LAYERS_NAMES
from .vgg import vgg_19

MEAN_PIXEL = np.array([123.68, 116.779, 103.939])


def get_layers(inputs, reuse_variables):
    _, layers = vgg_19(inputs, num_classes=None, reuse=reuse_variables)
    content_layer = layers[VGG_19_CONTENT_LAYER_NAME]
    style_layers = [layers[name] for name in VGG_19_STYLE_LAYERS_NAMES]
    return content_layer, style_layers


def get_style_layers_values(style_image, reuse_variables):
    return get_layers_values(style_image, VGG_19_STYLE_LAYERS_NAMES, reuse_variables)


def get_content_layer_values(content_image, reuse_variables):
    return get_layers_values(content_image, [VGG_19_CONTENT_LAYER_NAME], reuse_variables)[0]


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
