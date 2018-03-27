import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor, wait
from PIL import Image

from settings import FILES_DIR

DATASET_DIR = os.path.join(FILES_DIR, 'dataset')
if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

IMAGE_SIZE = 256


def prepare_image(filename, source_path, dest_path):
    image = Image.open(os.path.join(source_path, filename))
    size = np.asarray(image.size)
    size = (size * IMAGE_SIZE / min(size)).astype(int)
    image = image.resize(size, resample=Image.LANCZOS)
    w, h = image.size
    image = image.crop(((w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2, (w + IMAGE_SIZE) // 2, (h + IMAGE_SIZE) // 2))
    image.save(os.path.join(dest_path, filename))


def prepare_dataset(source_path, dest_path):
    files = (f for f in os.listdir(source_path) if os.path.splitext(f)[1] == '.jpg')

    executor = ThreadPoolExecutor()
    processed_images_count = 0
    finished = False
    while not finished:
        batch = []
        for i in range(10):
            try:
                batch.append(next(files))
                processed_images_count += 1
            except StopIteration:
                finished = True
                break
        futures = [executor.submit(prepare_image, filename, source_path, dest_path) for filename in batch]
        wait(futures)

        if processed_images_count % 100 == 0:
            print(f'Processed {processed_images_count} images')
    print(f'Processed {processed_images_count} images')


if __name__ == '__main__':
    prepare_dataset('/media/petr/Data/Downloads/train2017', DATASET_DIR)
