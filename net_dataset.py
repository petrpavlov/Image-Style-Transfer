import os
import time

import numpy as np

from concurrent.futures import ThreadPoolExecutor, wait
from PIL import Image

from settings import TRAIN_IMAGE_SIZE


def make_square_image(image, side_size):
    size = np.asarray(image.size)
    size = (size * side_size / min(size)).astype(int)
    image = image.resize(size, resample=Image.LANCZOS)
    w, h = image.size
    image = image.crop((
        (w - TRAIN_IMAGE_SIZE) // 2,
        (h - TRAIN_IMAGE_SIZE) // 2,
        (w + TRAIN_IMAGE_SIZE) // 2,
        (h + TRAIN_IMAGE_SIZE) // 2)
    )
    return image


def prepare_train_image(filename, source_path, dest_path):
    image = Image.open(os.path.join(source_path, filename))
    image = make_square_image(image, TRAIN_IMAGE_SIZE)
    image.save(os.path.join(dest_path, filename))


def prepare_train_dataset(source_path, dest_path):
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    files = [f for f in os.listdir(source_path) if os.path.splitext(f)[1] == '.jpg']
    executor = ThreadPoolExecutor()
    batch_size = 2
    start = time.time()
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        futures = [executor.submit(prepare_train_image, filename, source_path, dest_path) for filename in batch]
        wait(futures)

        if (i + batch_size) % 100 == 0:
            print(f'Processed {i + batch_size} images. Elapsed time per 100 images: {time.time() - start}')
            start = time.time()
