import os
import tarfile
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import requests
from PIL import Image

from settings import FILES_DIR, COCO_DATASET_URL, COCO_DATASET_ZIP_FILE, COCO_DATASET_PATH, TRAIN_IMAGE_SIZE, \
    VGG_19_CHECKPOINT_URL, VGG_19_CHECKPOINT_FILENAME


def prepare_vgg_19_checkpoint():
    if not os.path.exists(VGG_19_CHECKPOINT_FILENAME):
        print(f'Checkpoint does not exist. Download from: {VGG_19_CHECKPOINT_URL}')
        response = requests.get(VGG_19_CHECKPOINT_URL)

        print(f'Extract checkpoint into {FILES_DIR}')
        with tarfile.open(fileobj=BytesIO(response.content)) as tar:
            tar.extractall(FILES_DIR)
    else:
        print('Checkpoint already exists')


def prepare_dataset():
    _download_dataset()
    _rescale_dataset()


def _download_dataset():
    if not os.path.exists(COCO_DATASET_ZIP_FILE):
        print(f'Download dataset from {COCO_DATASET_URL}')

        with open(COCO_DATASET_ZIP_FILE, 'wb') as f:
            r = requests.get(COCO_DATASET_URL, stream=True)
            for chunk in r.iter_content(chunk_size=1_000_000):
                f.write(chunk)
    else:
        print(f'Dataset file ({COCO_DATASET_ZIP_FILE}) already exists')


def _rescale_image(image):
    size = np.asarray(image.size)
    size = (size * TRAIN_IMAGE_SIZE / min(size)).astype(int)
    image = image.resize(size, resample=Image.LANCZOS)
    w, h = image.size
    image = image.crop((
        (w - TRAIN_IMAGE_SIZE) // 2,
        (h - TRAIN_IMAGE_SIZE) // 2,
        (w + TRAIN_IMAGE_SIZE) // 2,
        (h + TRAIN_IMAGE_SIZE) // 2)
    )

    return image


def _rescale_dataset():
    if not os.path.exists(COCO_DATASET_PATH):
        print(f'Rescale images in COCO dataset to size: ({TRAIN_IMAGE_SIZE}, {TRAIN_IMAGE_SIZE})')

        os.mkdir(COCO_DATASET_PATH)

        with ZipFile(COCO_DATASET_ZIP_FILE) as zip_file:
            for image_filename in zip_file.namelist():
                with zip_file.open(image_filename) as image_file:
                    image = Image.open(image_file)
                    image = _rescale_image(image)
                    image.save(os.path.join(COCO_DATASET_PATH, image_filename))
    else:
        print('Scaled dataset already exists.')


if __name__ == '__main__':
    prepare_vgg_19_checkpoint()
    prepare_dataset()
