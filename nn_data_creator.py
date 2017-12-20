from os import listdir
from random import shuffle
import json

POSITIVE_FOLDER = "average_resized_dataset/positive"
NEGATIVE_FOLDER = "average_resized_dataset/negative"

def main():
    positive_files = listdir(POSITIVE_FOLDER)
    negative_files = listdir(NEGATIVE_FOLDER)

    file_names = positive_files + negative_files
    shuffle(file_names) 

    dataset = []
    for file_name in file_names:
        label = None
        if file_name[0] is 'p':
            file_name = 'positive/' + file_name
            label = 1
        elif file_name[0] is 'n':
            file_name = 'negative/' + file_name
            label = 0
        else:
            raise ValueError("Invalid file name!")

        dataset.append({
            'image': file_name,
            'label': label
        })

    train_dataset = dataset[: round(0.8 * len(dataset))]
    test_dataset = dataset[round(0.8 * len(dataset)) :]

    f = open("average_resized_dataset/train_data.json", "w")
    f.write(json.dumps(train_dataset, indent=4, sort_keys=True))
    
    f = open("average_resized_dataset/test_data.json", 'w')
    f.write(json.dumps(test_dataset, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
