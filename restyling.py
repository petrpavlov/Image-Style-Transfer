import os
import requests
import tarfile
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from io import BytesIO
from PIL import Image

from vgg import vgg_19

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
if not os.path.exists(FILES_DIR):
    os.mkdir(FILES_DIR)

CHECKPOINT_URL = 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'
CHECKPOINT_FILENAME = os.path.join(FILES_DIR, 'vgg_19.ckpt')

MAX_IMAGE_SIZE = 512

VGG_STYLE_LAYERS_NAMES = [
    'vgg_19/conv1/conv1_1',
    'vgg_19/conv2/conv2_1',
    'vgg_19/conv3/conv3_1',
    'vgg_19/conv4/conv4_1',
    'vgg_19/conv5/conv5_1',
]
VGG_STYLE_LAYERS_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
VGG_CONTENT_LAYER_NAME = 'vgg_19/conv4/conv4_2'


def download_checkpoint():
    if not os.path.exists(CHECKPOINT_FILENAME):
        print(f'Checkpoint does not exist. Download from: {CHECKPOINT_URL}')
        response = requests.get(CHECKPOINT_URL)

        print(f'Extract checkpoint into {FILES_DIR}')
        with tarfile.open(fileobj=BytesIO(response.content)) as tar:
            tar.extractall(FILES_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple implementation fo "A Neural Algorithm for Artistic Style"')

    parser.add_argument('content_image_filename', type=str, help='Path to content image')
    parser.add_argument('style_image_filename', type=str, help='Path to style image')
    parser.add_argument('result_image_filename', type=str, help='Path to result image')
    parser.add_argument('--content_loss_weight', type=float, default=5e0, help='Weight for content loss function')
    parser.add_argument('--style_loss_weight', type=float, default=1e4, help='Weight for style loss function')
    parser.add_argument('--total_variation_loss_weight', type=float, default=1e-3,
                        help='Weight for total variance loss function')
    parser.add_argument('--max_iterations', type=int, default=1000, help='Maximum training iterations count')
    parser.add_argument('--verbose', action='store_true',
                        help='Boolean flag indicating if training information should be printed.')

    return vars(parser.parse_args())


def read_content_image(content_image_filename):
    content_image = Image.open(content_image_filename)

    size = np.asarray(content_image.size)
    size = (size * MAX_IMAGE_SIZE / max(size)).astype(int)

    content_image = content_image.resize(size, resample=Image.LANCZOS)

    return np.asarray(content_image)


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


def write_result_image(result, result_image_filename):
    result = np.clip(result, 0, 255)
    result = result.astype(np.uint8)
    result = np.squeeze(result)

    result_image = Image.fromarray(result, mode='RGB')
    result_image.save(result_image_filename)


def transfer_style(content_image_filename, style_image_filename, result_image_filename, content_loss_weight,
                   style_loss_weight, total_variation_loss_weight, max_iterations, verbose):
    content_image = read_content_image(content_image_filename)
    style_image = read_style_image(style_image_filename, content_image.shape[:2])

    image = slim.variable('input', initializer=tf.constant(np.expand_dims(content_image, 0), dtype=tf.float32),
                          trainable=True)
    _, end_points = vgg_19(image, num_classes=None)

    content_layer = end_points[VGG_CONTENT_LAYER_NAME]
    content_layer_target = get_content_layer_target(content_image, True)
    content_loss = get_content_loss(content_layer, content_layer_target)

    style_layers = [end_points[name] for name in VGG_STYLE_LAYERS_NAMES]
    style_layers_targets = get_style_layers_targets(style_image, True)
    style_loss = get_style_loss(style_layers, style_layers_targets, VGG_STYLE_LAYERS_WEIGHTS)

    total_variation_loss = tf.image.total_variation(image)

    loss = content_loss_weight * content_loss + style_loss_weight * style_loss + total_variation_loss_weight * \
           total_variation_loss
    train_operation = tf.train.AdamOptimizer(learning_rate=1e0).minimize(loss)

    saver = tf.train.Saver(tf.get_collection('model_variables'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, CHECKPOINT_FILENAME)

        for i in range(max_iterations):
            sess.run(train_operation)
            if verbose and (i % 50 == 0):
                print(f'Iteration: {i}, Loss: {sess.run(loss)}')

        result = sess.run(image)
        write_result_image(result, result_image_filename)


def main():
    transfer_style_kwargs = parse_args()
    download_checkpoint()
    transfer_style(**transfer_style_kwargs)


if __name__ == '__main__':
    main()
