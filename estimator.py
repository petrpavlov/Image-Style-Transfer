import os
import tensorflow as tf
import numpy as np

from PIL import Image
from vgg import vgg_19


VGG_STYLE_LAYERS_NAMES = [
    'vgg_19/conv1/conv1_1',
    'vgg_19/conv2/conv2_1',
    'vgg_19/conv3/conv3_1',
    'vgg_19/conv4/conv4_1',
    'vgg_19/conv5/conv5_1',
]
VGG_STYLE_LAYERS_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
VGG_CONTENT_LAYER_NAME = 'vgg_19/conv4/conv4_2'

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
if not os.path.exists(FILES_DIR):
    os.mkdir(FILES_DIR)

CHECKPOINT_URL = 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'
CHECKPOINT_FILENAME = os.path.join(FILES_DIR, 'vgg_19.ckpt')


def read_style_image(style_image_filename, content_image_size):
    style_image = Image.open(style_image_filename)
    style_image = style_image.resize(content_image_size, resample=Image.LANCZOS)

    return np.asarray(style_image)


def get_vgg_19_layers_values(image, layer_names, reuse_variables):
    inputs = tf.expand_dims(tf.constant(image, tf.float32), 0)
    _, end_points = vgg_19(inputs, num_classes=None, reuse=reuse_variables)

    layers = [end_points[name] for name in layer_names]

    saver = tf.train.Saver(tf.get_collection('model_variables'))
    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_FILENAME)
        layers_values = sess.run(layers)

        return layers_values


def get_style_layers_targets(style_image, reuse_vgg_variables):
    return get_vgg_19_layers_values(style_image, VGG_STYLE_LAYERS_NAMES, reuse_vgg_variables)


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


def get_style_loss(style_layers, style_layers_targets, style_layers_weights):
    loss = 0
    for layer, value, weight in zip(style_layers, style_layers_targets, style_layers_weights):
        loss += style_layer_loss(value, layer) * weight
    loss /= float(len(style_layers))
    return loss


def get_content_layer_target(content_image, reuse_vgg_variables):
    return get_vgg_19_layers_values(content_image, [VGG_CONTENT_LAYER_NAME], reuse_vgg_variables)[0]


def get_content_loss(content_layer, content_layer_target):
    _, h, w, d = content_layer.shape
    M = h.value * w.value
    N = d.value
    K = 1. / (2. * N ** 0.5 * M ** 0.5)

    return K * tf.reduce_sum(tf.pow(content_layer - content_layer_target, 2))


def training_input_fn(image_filenames, batch_size, repeat_count):
    dataset = tf.data.Dataset.from_tensor_slices(image_filenames)

    def decode_image(filename):
        content = tf.read_file(filename)
        image = tf.image.decode_jpeg(content, channels=3)
        image = tf.cast(image, tf.float32)
        return image

    dataset = dataset.shuffle(10 * batch_size)
    dataset = dataset.map(decode_image, num_parallel_calls=10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_count)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.layers.conv2d(net, filters=32, kernel_size=9, strides=1, activation=tf.nn.relu)
    net = tf.layers.conv2d(net, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu)
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=2, activation=tf.nn.relu)

    def residual_block(x):
        f = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1)
        f = tf.layers.batch_normalization(f)
        f = tf.nn.relu(f)
        f = tf.layers.conv2d(f, filters=128, kernel_size=3, strides=1)
        f = tf.layers.batch_normalization(f)
        return x + f

    for i in range(5):
        net = residual_block(net)

    net = tf.layers.conv2d_transpose(net, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu)
    net = tf.layers.conv2d_transpose(net, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu)
    net = tf.layers.conv2d(net, filters=3, kernel_size=9, strides=1)
    images = 127.5 + 127.5 * tf.nn.tanh(net)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        _, end_points = vgg_19(images, num_classes=None)

        content_layer = end_points[VGG_CONTENT_LAYER_NAME]
        content_layer_target = get_content_layer_target(images, True)
        content_loss = get_content_loss(content_layer, content_layer_target)

        style_layers = [end_points[name] for name in VGG_STYLE_LAYERS_NAMES]
        style_image = read_style_image(params['style_image_filename'], features.shape[1:3])
        style_layers_targets = get_style_layers_targets(style_image, True)
        style_loss = get_style_loss(style_layers, style_layers_targets, VGG_STYLE_LAYERS_WEIGHTS)

        total_variation_loss = tf.reduce_sum(tf.image.total_variation(images))

        loss = params['content_loss_weight'] * content_loss + \
               params['style_loss_weight'] * style_loss + \
               params['total_variation_loss_weight'] * total_variation_loss

        train_op = tf.train.AdamOptimizer(learning_rate=1e0).minimize(loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'images': tf.cast(images, tf.uint8)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

