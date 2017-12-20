from PIL import Image
from os import listdir

FOLDER_NAME = "raw_dataset"

def main():
    total_width = 0
    total_height = 0

    image_names = listdir(FOLDER_NAME)
    for image_name in image_names:
        print(image_name)
        image = Image.open(FOLDER_NAME + '/' + image_name)

        total_width += image.size[0]
        total_height += image.size[1]
    
    print("\n\Avg width: {}\nAvg height: {}\n".format(
        total_width / len(image_names), total_height / len(image_names)))

if __name__ == '__main__':
    main()
