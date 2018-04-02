import os
import requests
import tarfile
import time

import numpy as np

from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, wait
from PIL import Image

from settings import FILES_DIR, TRAIN_IMAGE_SIZE, VGG_19_CHECKPOINT_URL, VGG_19_CHECKPOINT_FILENAME


def prepare_vgg_19_checkpoint():
    if not os.path.exists(VGG_19_CHECKPOINT_FILENAME):
        print(f'Checkpoint does not exist. Download from: {VGG_19_CHECKPOINT_URL}')
        response = requests.get(VGG_19_CHECKPOINT_URL)

        print(f'Extract checkpoint into {FILES_DIR}')
        with tarfile.open(fileobj=BytesIO(response.content)) as tar:
            tar.extractall(FILES_DIR)


def prepare_dataset():
    pass


def _download_dataset():
    pass


def _rescale_image(filename, source_path, dest_path):
    image = Image.open(os.path.join(source_path, filename))

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

    image.save(os.path.join(dest_path, filename))


def _rescale_dataset(source_path, dest_path):
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    files = [f for f in os.listdir(source_path) if os.path.splitext(f)[1] == '.jpg']
    executor = ThreadPoolExecutor()
    batch_size = 2
    start = time.time()
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        futures = [executor.submit(_rescale_image, filename, source_path, dest_path) for filename in batch]
        wait(futures)

        if (i + batch_size) % 100 == 0:
            print(f'Processed {i + batch_size} images. Elapsed time per 100 images: {time.time() - start}')
            start = time.time()
