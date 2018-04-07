import os


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
if not os.path.exists(FILES_DIR):
    os.mkdir(FILES_DIR)

# Max image size for slow algorithm
MAX_IMAGE_SIZE = 512

COCO_DATASET_URL = 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'
COCO_DATASET_ZIP_FILE = os.path.join(FILES_DIR, 'train2014.zip')
COCO_DATASET_PATH = os.path.join(FILES_DIR, 'train2014')
TRAIN_IMAGE_SIZE = 256

MODEL_DIR = os.path.join(FILES_DIR, 'model')

VGG_19_CHECKPOINT_URL = 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'
VGG_19_CHECKPOINT_FILENAME = os.path.join(FILES_DIR, 'vgg_19.ckpt')
VGG_19_STYLE_LAYERS_NAMES = [
    'vgg_19/conv1/conv1_1',
    'vgg_19/conv2/conv2_1',
    'vgg_19/conv3/conv3_1',
    'vgg_19/conv4/conv4_1',
    'vgg_19/conv5/conv5_1',
]
VGG_19_STYLE_LAYERS_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
VGG_19_CONTENT_LAYER_NAME = 'vgg_19/conv4/conv4_2'
