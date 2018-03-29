import numpy as np
import tensorflow as tf

from . import vgg_tools


VGG_STYLE_LAYERS_NAMES = [
    'vgg_19/conv1/conv1_1',
    'vgg_19/conv2/conv2_1',
    'vgg_19/conv3/conv3_1',
    'vgg_19/conv4/conv4_1',
    'vgg_19/conv5/conv5_1',
]
VGG_STYLE_LAYERS_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
VGG_CONTENT_LAYER_NAME = 'vgg_19/conv4/conv4_2'


def get_layers(inputs, reuse_variables):
    layers = vgg_tools.get_layers(inputs, reuse_variables)
    content_layer = layers[VGG_CONTENT_LAYER_NAME]
    style_layers = [layers[name] for name in VGG_STYLE_LAYERS_NAMES]
    return content_layer, style_layers


def get_style_layers_values(style_image, reuse_variables):
    return vgg_tools.get_layers_values(style_image, VGG_STYLE_LAYERS_NAMES, reuse_variables)


def gram_matrix(x):
    if isinstance(x, np.ndarray):
        _, h, w, d = x.shape
        x = np.reshape(x, (-1, h * w, d))
        return np.matmul(np.transpose(x, axes=[0, 2, 1]), x) / (h * w * d)
    elif isinstance(x, tf.Tensor):
        _, h, w, d = (dim.value for dim in x.shape)
        x = tf.reshape(x, (-1, h * w, d))
        return tf.matmul(tf.transpose(x, perm=[0, 2, 1]), x) / (h * w * d)
    else:
        raise ValueError(f'Invalid type of x: {type(x)}')


def style_layer_loss(a, x):
    gram_a = gram_matrix(a)
    gram_x = gram_matrix(x)
    return tf.nn.l2_loss(gram_a - gram_x)


def get_style_loss(style_layers, style_layers_targets):
    loss = 0
    for layer, value, weight in zip(style_layers, style_layers_targets, VGG_STYLE_LAYERS_WEIGHTS):
        loss += style_layer_loss(value, layer) * weight
    loss /= float(len(style_layers))
    return loss


def get_content_layer_values(content_image, reuse_variables):
    return vgg_tools.get_layers_values(content_image, [VGG_CONTENT_LAYER_NAME], reuse_variables)[0]


def get_content_loss(content_layer, content_layer_target):
    assert isinstance(content_layer, tf.Tensor)

    return tf.nn.l2_loss(content_layer_target - content_layer) / \
           np.prod([dim.value for dim in content_layer.shape if dim.value is not None])


def get_total_variation_loss(images):
    return tf.reduce_sum(tf.image.total_variation(images))
