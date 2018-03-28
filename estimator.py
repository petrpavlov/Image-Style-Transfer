import os

import tensorflow as tf

from PIL import Image

from util import vgg
from settings import FILES_DIR, TRAIN_IMAGE_SIZE


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

    dataset = dataset.shuffle(10 * batch_size)
    dataset = dataset.map(decode_image, num_parallel_calls=10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_count)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def model_fn(features, labels, mode, params):
    print(features.shape)
    net = tf.layers.conv2d(features, filters=32, kernel_size=9, strides=1, activation=tf.nn.relu, padding='same')
    print(net.shape)
    net = tf.layers.conv2d(net, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    print(net.shape)
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    print(net.shape)

    def residual_block(x):
        f = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same')
        f = tf.layers.batch_normalization(f)
        f = tf.nn.relu(f)
        f = tf.layers.conv2d(f, filters=128, kernel_size=3, strides=1, padding='same')
        f = tf.layers.batch_normalization(f)
        return x + f

    for i in range(5):
        net = residual_block(net)
        print(net.shape)

    net = tf.layers.conv2d_transpose(net, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    print(net.shape)
    net = tf.layers.conv2d_transpose(net, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    print(net.shape)
    net = tf.layers.conv2d(net, filters=3, kernel_size=9, strides=1, padding='same')
    print(net.shape)
    images = 127.5 + 127.5 * tf.nn.tanh(net)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        content_layer, style_layers = vgg.get_layers(images, reuse_variables=False)

        content_layer_target, _ = vgg.get_layers(features, reuse_variables=True)
        content_loss = vgg.get_content_loss(content_layer, content_layer_target)

        style_image = Image.open(params['style_image_filename'])
        style_layers_targets = vgg.get_style_layers_targets(style_image, reuse_variables=True)
        style_loss = vgg.get_style_loss(style_layers, style_layers_targets)

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


def train():
    dataset_path = os.path.join(FILES_DIR, 'net-train-dataset')

    estimator = tf.estimator.Estimator(model_fn=model_fn, params={
        'style_image_filename': os.path.join(dataset_path, 'style.jpg'),
        'content_loss_weight' : 1,
        'style_loss_weight': 1e4,
        'total_variation_loss_weight': 1e-3
    }, model_dir=os.path.join(FILES_DIR, 'model'))
    estimator.train(input_fn=lambda : train_input_fn(os.path.join(dataset_path, 'train'), 4, 2), steps=100)

    # image = train_input_fn(os.path.join(dataset_path, 'train'), 1, 1)
    # print(image.shape)
    # with tf.Session() as sess:
    #     print(sess.run(image).shape)


if __name__ == '__main__':
    train()


