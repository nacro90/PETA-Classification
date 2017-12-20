from PIL import Image
from os import listdir
from random import choice
from image_resize import TARGET_SIZE

RAW_IMAGE_FOLDER = "raw_dataset/negative"
PROCESSED_IMAGE_FOLDER = "average_resized_dataset/negative"

VERTICAL_STRIDE_RATIO = 2/3
HORIZONTAL_STRIDE_RATIO = 2/3


def main():

    image_names = listdir(RAW_IMAGE_FOLDER)
    total = len(image_names)
    number = 0

    for index, image_name in enumerate(image_names):
        image = Image.open(RAW_IMAGE_FOLDER + '/' + image_name)
        width, height = image.size
        left_margin = 0
        top_margin = 0
        cropped_image = None

        checkpoint = number

        # Base striding top to down
        while top_margin + TARGET_SIZE[1] < height:

            left_margin = 0
            
            # Base striding left to right at every vertical step
            while left_margin + TARGET_SIZE[0] < width:
                cropped_image = image.crop((left_margin, top_margin,
                                            left_margin + TARGET_SIZE[0],
                                            top_margin + TARGET_SIZE[1]))

                cropped_image.save(PROCESSED_IMAGE_FOLDER +
                                '/temp/' + str(number) + '.png')

                # One step right
                left_margin += TARGET_SIZE[0] * HORIZONTAL_STRIDE_RATIO

                number += 1
            
            # One step down
            top_margin += TARGET_SIZE[1] * VERTICAL_STRIDE_RATIO

        # Fixing the top margin to very bottom of the image
        top_margin = height - TARGET_SIZE[1]
        left_margin = 0

        # Very bottom row of images
        while left_margin + TARGET_SIZE[0] < width:
            cropped_image = image.crop((left_margin, top_margin,
                                        left_margin + TARGET_SIZE[0],
                                        top_margin + TARGET_SIZE[1]))

            cropped_image.save(PROCESSED_IMAGE_FOLDER +
                            '/temp/' + str(number) + '.png')

            left_margin += TARGET_SIZE[0] * HORIZONTAL_STRIDE_RATIO

            number += 1

        # The last crop at bottom right of image
        left_margin = width - TARGET_SIZE[0]
        cropped_image = image.crop((left_margin, top_margin, width, height))
        cropped_image.save(PROCESSED_IMAGE_FOLDER +
                           '/temp/' + str(number) + '.png')

        number += 1

        print("{}/{}".format(index+1, total), '=>', str(number - checkpoint) + ' images')


if __name__ == '__main__':
    main()
