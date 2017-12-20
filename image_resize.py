from PIL import Image
from os import listdir
from random import shuffle, choice

RESIZED_IMAGE_FOLDER = 'average_resized_dataset/positive'
RAW_IMAGE_FOLDER = "raw_dataset/positive"

TARGET_SIZE = (72, 170)

def main():
    image_names = listdir(RAW_IMAGE_FOLDER)
    numbers = list(range(len(image_names*2)))
    for image_name in image_names:
        image = Image.open(RAW_IMAGE_FOLDER + '/' + image_name)
        image = image.resize(TARGET_SIZE, Image.ANTIALIAS)
        number = choice(numbers)
        numbers.remove(number)
        image.save(RESIZED_IMAGE_FOLDER + '/' + str(number) + '.png')
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        number = choice(numbers)
        numbers.remove(number)
        image.save(RESIZED_IMAGE_FOLDER + '/' + str(number) + '.png')
        print(image_name, str(number) + '.png', sep=' => ')

if __name__ == '__main__':
    main()
