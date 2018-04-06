import argparse
import os
import shutil

import numpy as np
import tensorflow as tf
from PIL import Image

from restyling import losses, vgg_tools
from settings import TRAIN_IMAGE_SIZE, COCO_DATASET_PATH, MODEL_DIR, VGG_19_CHECKPOINT_FILENAME


class RestoreVgg19Hook(tf.train.SessionRunHook):
    def __init__(self, saver):
        self._saver = saver

    def after_create_session(self, session, coord):
        self._saver.restore(session, VGG_19_CHECKPOINT_FILENAME)


def train_input_fn(batch_size, repeat_count):
    filenames = [os.path.join(COCO_DATASET_PATH, f) for f in os.listdir(COCO_DATASET_PATH)]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    def decode_image(filename):
        content = tf.read_file(filename)
        image = tf.image.decode_jpeg(content, channels=3)
        image = tf.cast(image, tf.float32)
        image.set_shape((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3))
        return image

    dataset = dataset.shuffle(buffer_size=batch_size * 100)
    dataset = dataset.map(decode_image, num_parallel_calls=10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_count)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def predict_input_fn(image_filename):
    content = tf.read_file(image_filename)
    image = tf.image.decode_jpeg(content, channels=3)
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)


def model_fn(features, labels, mode, params):
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    def conv_block(x, filters, kernel_size, strides, relu=True):
        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        if relu:
            return tf.nn.relu(x)
        else:
            return x

    def residual_block(x):
        f = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same')
        f = tf.layers.batch_normalization(f, training=training)
        f = tf.nn.relu(f)
        f = tf.layers.conv2d(f, filters=128, kernel_size=3, strides=1, padding='same')
        f = tf.layers.batch_normalization(f, training=training)
        return x + f

    def conv_transpose_block(x, filters, kernel_size, strides):
        x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        return tf.nn.relu(x)

    net = conv_block(features / 255.0, filters=32, kernel_size=9, strides=1)
    net = conv_block(net, filters=64, kernel_size=3, strides=2)
    net = conv_block(net, filters=128, kernel_size=3, strides=2)

    for _ in range(5):
        net = residual_block(net)

    net = conv_transpose_block(net, filters=64, kernel_size=3, strides=2)
    net = conv_transpose_block(net, filters=32, kernel_size=3, strides=2)

    net = conv_block(net, filters=3, kernel_size=9, strides=1, relu=False)
    images = 127.5 + 127.5 * tf.nn.tanh(net)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        content_layer, style_layers = vgg_tools.get_layers(images, reuse_variables=False)

        content_layer_target, _ = vgg_tools.get_layers(features, reuse_variables=True)
        content_loss = params['content_loss_weight'] * losses.get_content_loss(content_layer, content_layer_target)

        style_image = np.asarray(Image.open(params['style_image_filename']))
        style_layers_targets = vgg_tools.get_style_layers_values(style_image, reuse_variables=True)
        style_loss = params['style_loss_weight'] * losses.get_style_loss(style_layers, style_layers_targets)

        total_variation_loss = params['total_variation_loss_weight'] * losses.get_total_variation_loss(images)

        total_loss = content_loss + style_loss + total_variation_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())

        training_hooks = [
            RestoreVgg19Hook(tf.train.Saver(tf.get_collection('model_variables'))),
            tf.train.LoggingTensorHook({
                'content_loss': content_loss,
                'style_loss': style_loss,
                'total_variation_loss': total_variation_loss,
                'total_loss': total_loss
            }, every_n_iter=10)
        ]
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op, training_hooks=training_hooks)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'images': images
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def train(args):
    if args.clean:
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)

    config = tf.estimator.RunConfig(
        save_checkpoints_secs=30
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, params={
        'style_image_filename': args.style_image_filename,
        'content_loss_weight': args.content_loss_weight,
        'style_loss_weight': args.style_loss_weight,
        'total_variation_loss_weight': args.total_variation_loss_weight
    }, model_dir=MODEL_DIR, config=config)
    try:
        estimator.train(input_fn=lambda: train_input_fn(4, 2))
    except KeyboardInterrupt:
        pass


def predict(args):
    estimator = tf.estimator.Estimator(model_fn=model_fn, params={}, model_dir=MODEL_DIR)

    image = next(estimator.predict(input_fn=lambda: predict_input_fn(args.content_image_filename)))['images']
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image, mode='RGB')
    image.save(args.result_filename)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple implementation of "A Neural Algorithm for Artistic Style"')
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train', help='Train model')
    parser_train.add_argument('style_image_filename', type=str, help='Path to style image')
    parser_train.add_argument('--content_loss_weight', type=float, default=1e0, help='Weight for content loss function')
    parser_train.add_argument('--style_loss_weight', type=float, default=1e2, help='Weight for style loss function')
    parser_train.add_argument('--total_variation_loss_weight', type=float, default=1e-4,
                              help='Weight for total variance loss function')
    parser_train.add_argument('--clean', action='store_true',
                              help='Should perform clean training (remove saved checkpoints)')
    parser_train.set_defaults(func=train)

    parser_predict = subparsers.add_parser('predict', help='Transfer style of image')
    parser_predict.add_argument('content_image_filename', type=str, help='Path to content image')
    parser_predict.add_argument('result_image_filename', type=str, help='Path to result image')
    parser_predict.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    parse_args()

    
if __name__ == '__main__':
    main()
