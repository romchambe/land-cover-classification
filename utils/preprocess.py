from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join

LAND_COVER = [
    ['AnnualCrop', 0],
    ['Forest', 1],
    ['HerbaceousVegetation', 2],
    ['Highway', 3],
    ['Industrial', 4],
    ['Pasture', 5],
    ['PermanentCrop', 6],
    ['Residential', 7],
    ['River', 8],
    ['SeaLake', 9]
]

INTERMEDIARY_FILE_PATH = "training_data/labelled_dataset.npz"


def image_to_array(path):
    image = Image.open(path)
    return np.array(image) / 255.0


def preprocess_data():
    image_dataset = []
    label_dataset = []

    # Convert
    for land_cover_class, land_cover_code in LAND_COVER:
        land_cover_directory = f'./raw_data/{land_cover_class}'

        for image_file_name in listdir(land_cover_directory):
            image_path = join(land_cover_directory, image_file_name)
            if (isfile(image_path)):
                image_dataset.append(
                    np.array(image_to_array(image_path))
                )

                label_dataset.append(int(land_cover_code))

    x_train = np.array(image_dataset)
    y_train = np.array(label_dataset)

    # Save to a file
    np.savez(
        INTERMEDIARY_FILE_PATH,
        x_train=x_train,
        y_train=y_train
    )

    print(
        "The shape of training images: {} and training labels: {}".format(
            x_train.shape, y_train.shape
        )
    )
