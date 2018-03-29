import os
import shutil

import numpy as np
import tensorflow as tf

from PIL import Image

from restyling import losses, vgg_tools
from settings import FILES_DIR, TRAIN_IMAGE_SIZE


MODEL_DIR = os.path.join(FILES_DIR, 'model')

TRAIN_DATASET_PATH = os.path.join(FILES_DIR, 'net-train-dataset')
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DATASET_PATH, 'train-images')
STYLE_IMAGE_FILENAME = os.path.join(TRAIN_DATASET_PATH, 'style.jpg')
CONTENT_IMAGE_FILENAME = os.path.join(TRAIN_DATASET_PATH, 'content.jpg')


class RestoreVgg19Hook(tf.train.SessionRunHook):
    def __init__(self, saver):
        self._saver = saver

    def after_create_session(self, session, coord):
        self._saver.restore(session, vgg_tools.VGG_19_CHECKPOINT_FILENAME)


def train_input_fn(images_dir, batch_size, repeat_count):
    filenames = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1] == '.jpg']
    filenames = list(map(lambda x: os.path.join(images_dir, x), filenames))

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    def decode_image(filename):
        content = tf.read_file(filename)
        image = tf.image.decode_jpeg(content, channels=3)
        image = tf.cast(image, tf.float32)
        image.set_shape((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3))
        return image

    dataset = dataset.map(decode_image, num_parallel_calls=10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_count)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def predict_input_fn(image_filename):
    content = tf.read_file(image_filename)
    image = tf.image.decode_jpeg(content, channels=3)
    image = tf.cast(image, tf.float32)
    image.set_shape((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3))
    return tf.expand_dims(image, 0)


def model_fn(features, labels, mode, params):

    def conv_block(x, filters, kernel_size, strides):
        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
        x = tf.layers.batch_normalization(x)
        return tf.nn.relu(x)

    def residual_block(x):
        f = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same')
        f = tf.layers.batch_normalization(f)
        f = tf.nn.relu(f)
        f = tf.layers.conv2d(f, filters=128, kernel_size=3, strides=1, padding='same')
        f = tf.layers.batch_normalization(f)
        return x + f

    def conv_transpose_block(x, filters, kernel_size, strides):
        x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
        x = tf.layers.batch_normalization(x)
        return tf.nn.relu(x)

    net = conv_block(features / 255.0, filters=32, kernel_size=9, strides=1)
    net = conv_block(net, filters=64, kernel_size=3, strides=2)
    net = conv_block(net, filters=128, kernel_size=3, strides=2)

    for i in range(5):
        net = residual_block(net)

    net = conv_transpose_block(net, filters=64, kernel_size=3, strides=2)
    net = conv_transpose_block(net, filters=32, kernel_size=3, strides=2)

    net = tf.layers.conv2d(net, filters=3, kernel_size=9, strides=1, padding='same')
    images = 127.5 + 150 * tf.nn.tanh(net)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        content_layer, style_layers = losses.get_layers(vgg_tools.pre_process(images), reuse_variables=False)

        content_layer_target, _ = losses.get_layers(vgg_tools.pre_process(features), reuse_variables=True)
        content_loss = losses.get_content_loss(content_layer, content_layer_target)

        style_image = np.asarray(Image.open(params['style_image_filename']))
        style_layers_targets = losses.get_style_layers_values(vgg_tools.pre_process(style_image), reuse_variables=True)
        style_loss = losses.get_style_loss(style_layers, style_layers_targets)

        total_variation_loss = losses.get_total_variation_loss(images)

        total_loss = params['content_loss_weight'] * content_loss + \
                     params['style_loss_weight'] * style_loss + \
                     params['total_variation_loss_weight'] * total_variation_loss

        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(total_loss,
                                                                       global_step=tf.train.get_global_step())

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


def train(clean=True):
    if clean:
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)

    estimator = tf.estimator.Estimator(model_fn=model_fn, params={
        'style_image_filename': STYLE_IMAGE_FILENAME,
        'content_loss_weight': 7.5e0,
        'style_loss_weight': 1e2,
        'total_variation_loss_weight': 2e2
    }, model_dir=MODEL_DIR)
    estimator.train(input_fn=lambda: train_input_fn(TRAIN_IMAGES_DIR, 4, None), steps=100_000)


def predict():
    estimator = tf.estimator.Estimator(model_fn=model_fn, params={}, model_dir=MODEL_DIR)

    image = next(estimator.predict(input_fn=lambda: predict_input_fn(CONTENT_IMAGE_FILENAME)))['images']
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode='RGB').show()


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    vgg_tools.maybe_download_checkpoint()
    try:
        train()
    except KeyboardInterrupt:
        pass
    predict()

    
if __name__ == '__main__':
    main()