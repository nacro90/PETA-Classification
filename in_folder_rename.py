from os import listdir, rename

RENAME_FOLDER = "average_resized_dataset/negative"

def main():
    for file_name in listdir(RENAME_FOLDER):
        number = int(file_name.split('_')[1][:-4])
        number_str = "%05d" % number
        old_name = RENAME_FOLDER + '/' + file_name
        new_name = RENAME_FOLDER + '/neg_' + number_str + '.png'
        rename(old_name, new_name)
        print(old_name, "->", new_name)

if __name__ == '__main__':
    main()
