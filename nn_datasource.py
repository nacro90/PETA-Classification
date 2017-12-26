import numpy as np
import os
from PIL import Image
from random import shuffle
import json

POSITIVE_ROOT = "positive/"
NEGATIVE_ROOT = "negative/"

DATASET_FOLDER = "processed_dataset_1/"

TRAINING_JSON = "train_data.json"
TEST_JSON = "test_data.json"


def training_batch_generator(batch_size):
    training_data = json.load(open(DATASET_FOLDER + TRAINING_JSON))

    for batch_index in range(int(len(training_data) / batch_size)):

        image_list = []
        labels = []

        for training_row in training_data[batch_index * batch_size:(batch_index + 1) * batch_size]:
            im = Image.open(DATASET_FOLDER + training_row['image'])
            width, height = im.size
            image_np = np.array(im.getdata()).reshape(width, height, 3)
            image_list.append(image_np)

            labels.append([training_row['label']])

        yield batch_index, np.array(image_list), np.array(labels)


def get_training_length():
    return len(json.load(open(DATASET_FOLDER + TRAINING_JSON)))


def test_batch_generator(batch_size):
    test_data = json.load(open(DATASET_FOLDER + TEST_JSON))

    image_list = []
    labels = []

    for batch_index in range(len(test_data) // batch_size):
        for test_row in test_data[batch_index * batch_size: (batch_index + 1) * batch_size]:
            im = Image.open(DATASET_FOLDER + test_row['image'])
            width, height = im.size
            image_np = np.array(im.getdata()).reshape(width, height, 3)
            image_list.append(image_np)

            labels.append([test_row['label']])

        yield np.array(image_list), np.array(labels)


def main():
    pass


if __name__ == '__main__':
    main()
