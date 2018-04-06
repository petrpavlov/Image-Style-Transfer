import numpy as np
import tensorflow as tf

from settings import VGG_19_STYLE_LAYERS_WEIGHTS


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

    loss = 2 * tf.nn.l2_loss(gram_a - gram_x) / tf.cast(tf.size(gram_a), tf.float32)

    return loss


def get_style_loss(style_layers, style_layers_targets):
    loss = 0
    for layer, value, weight in zip(style_layers, style_layers_targets, VGG_19_STYLE_LAYERS_WEIGHTS):
        loss += weight * style_layer_loss(value, layer)

    return loss


def get_content_loss(content_layer, content_layer_target):
    loss = 2 * tf.nn.l2_loss(content_layer_target - content_layer) / tf.cast(tf.size(content_layer), tf.float32)

    return loss


def get_total_variation_loss(images):
    loss = tf.reduce_sum(tf.image.total_variation(images)) / tf.cast(tf.shape(images)[0], tf.float32)

    return loss
