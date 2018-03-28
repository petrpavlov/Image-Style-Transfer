import argparse
import argparse

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

from util import vgg
from settings import MAX_IMAGE_SIZE


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
    content_layer, style_layers = vgg.get_layers(image, reuse_variables=False)
    content_layer_target = vgg.get_content_layer_target(content_image, True)
    content_loss = vgg.get_content_loss(content_layer, content_layer_target)

    style_layers_targets = vgg.get_style_layers_targets(style_image, True)
    style_loss = vgg.get_style_loss(style_layers, style_layers_targets)

    total_variation_loss = tf.reduce_sum(tf.image.total_variation(image))

    loss = content_loss_weight * content_loss + style_loss_weight * style_loss + total_variation_loss_weight * \
           total_variation_loss
    train_operation = tf.train.AdamOptimizer(learning_rate=1e0).minimize(loss)

    saver = tf.train.Saver(tf.get_collection('model_variables'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, vgg.VGG_19_CHECKPOINT_FILENAME)

        for i in range(max_iterations):
            sess.run(train_operation)
            if verbose and (i % 50 == 0):
                print(f'Iteration: {i}, Loss: {sess.run(loss):.4}')

        result = sess.run(image)
        write_result_image(result, result_image_filename)


def main():
    transfer_style_kwargs = parse_args()
    transfer_style(**transfer_style_kwargs)


if __name__ == '__main__':
    main()
