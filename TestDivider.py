import os
import shutil
import random
from pathlib import Path

images_train = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Supervisely/Train/img')
annotations_train = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Supervisely/Train/ann')

images_test = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Supervisely/Test/img')
annotations_test = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Supervisely/Test/ann')

files = [file for file in os.listdir(images_train) if file.endswith(".png")]

test = open("test.txt","w")

test_amount = round(0.15*len(files))

for x in range(test_amount):

    file = random.choice(files)
    files.remove(file)
    shutil.move(os.path.join(images_train, file), images_test)
    shutil.move(os.path.join(annotations_train, file), annotations_test)
    test.write(file + "\n")