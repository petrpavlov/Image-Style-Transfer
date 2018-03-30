import os


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
if not os.path.exists(FILES_DIR):
    os.mkdir(FILES_DIR)


TRAIN_IMAGE_SIZE = 256

MAX_IMAGE_SIZE = 512

VGG_19_CHECKPOINT_FILENAME = os.path.join(FILES_DIR, 'vgg_19.ckpt')
