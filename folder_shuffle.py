from os import listdir, rename
from random import shuffle

SHUFFLE_FOLDER = "average_resized_dataset/negative/temp"
TARGET_FOLDER = "average_resized_dataset/negative/shuffled"

def main():
    file_names = listdir(SHUFFLE_FOLDER)
    numbers = list(range(len(file_names)))
    shuffle(numbers)

    for index, file_name in enumerate(file_names):
        new_name = str(numbers[index])
        rename(SHUFFLE_FOLDER + '/' + file_name, 
               TARGET_FOLDER + '/' + new_name + '.png')

        print(file_name, '->', new_name + '.png')

if __name__ == '__main__':
    main()
