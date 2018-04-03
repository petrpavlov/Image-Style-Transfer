import argparse

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

from restyling import losses, vgg_tools
from settings import MAX_IMAGE_SIZE, VGG_19_CHECKPOINT_FILENAME
from prepare import prepare_vgg_19_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Simple implementation fo "A Neural Algorithm for Artistic Style"')

    parser.add_argument('content_image_filename', type=str, help='Path to content image')
    parser.add_argument('style_image_filename', type=str, help='Path to style image')
    parser.add_argument('result_image_filename', type=str, help='Path to result image')
    parser.add_argument('--content_loss_weight', type=float, default=1e0, help='Weight for content loss function')
    parser.add_argument('--style_loss_weight', type=float, default=1e2, help='Weight for style loss function')
    parser.add_argument('--total_variation_loss_weight', type=float, default=1e-2,
                        help='Weight for total variance loss function')
    parser.add_argument('--max_iterations', type=int, default=1000, help='Maximum training iterations count')

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
    result_image.show()


def transfer_style(content_image_filename, style_image_filename, result_image_filename, content_loss_weight,
                   style_loss_weight, total_variation_loss_weight, max_iterations):
    content_image = read_content_image(content_image_filename)
    style_image = read_style_image(style_image_filename, content_image.shape[:2])

    image = slim.variable('input', initializer=tf.constant(np.expand_dims(content_image, 0), dtype=tf.float32),
                          trainable=True)
    content_layer, style_layers = vgg_tools.get_layers(vgg_tools.pre_process(image), reuse_variables=False)

    content_layer_target = vgg_tools.get_content_layer_values(vgg_tools.pre_process(content_image), True)
    content_loss = content_loss_weight * losses.get_content_loss(content_layer, content_layer_target)

    style_layers_targets = vgg_tools.get_style_layers_values(vgg_tools.pre_process(style_image), True)
    style_loss = style_loss_weight * losses.get_style_loss(style_layers, style_layers_targets)

    total_variation_loss = total_variation_loss_weight * losses.get_total_variation_loss(image)

    total_loss = content_loss + style_loss + total_variation_loss
    train_operation = tf.train.AdamOptimizer(learning_rate=1e0).minimize(total_loss)

    saver = tf.train.Saver(tf.get_collection('model_variables'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, VGG_19_CHECKPOINT_FILENAME)

        for i in range(max_iterations):
            content_loss_value, style_loss_value, total_variation_loss_value, total_loss_value, _ = sess.run([
                content_loss,
                style_loss,
                total_variation_loss,
                total_loss,
                train_operation
            ])
            if i % 50 == 0:
                print(f'Iteration: {i}, Content loss: {content_loss_value:.4}. Style loss: {style_loss_value:.4}. '
                      f'Total variation loss: {total_variation_loss_value:.4}. Total loss: {total_loss_value:.4}')

        result = sess.run(image)
        write_result_image(result, result_image_filename)


def main():
    prepare_vgg_19_checkpoint()

    transfer_style_kwargs = parse_args()
    transfer_style(**transfer_style_kwargs)


if __name__ == '__main__':
    main()
