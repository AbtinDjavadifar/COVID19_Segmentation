import cv2 as cv
import numpy as np
import random
import pandas as pd
from skimage.io import imread
import os
import sys
import imageio

sys.path.append("..")

from BaseDataset import BaseDataset
from enhancements import contrast_stretch

current_dir = os.path.dirname(os.path.realpath(__file__))

class PathologyDataset(BaseDataset):

    def __init__(self, path=current_dir):
        super()
        self.path = os.path.abspath(path)
        # reading in the training set
        data = pd.read_csv(os.path.join(self.path, 'train.csv'))

        # keep only the images with labels
        self.filtered = data.dropna(subset=['ClassId'], axis='rows')
        self.filtered['ClassId'] = self.filtered['ClassId'].astype(np.uint8)

        # squash multiple rows per image into a list
        self.squashed = self.filtered[['ImageId', 'ClassId']] \
            .groupby('ImageId', as_index=False) \
            .agg(list)

        self.no_defects = data[['ImageId']].drop_duplicates().set_index('ImageId').drop(
            self.squashed.ImageId.tolist()).index.tolist()
        print("No pathology instances: %s" % len(self.no_defects))

    def get_classes(self):
        return list(map(lambda x: str(x), sorted(self.filtered['ClassId'].sort_index().unique().tolist())))

    def get_class_counts(self):
        return self.filtered['ClassId'].value_counts().sort_index()

    def get_class_members(self):
        classes = {}
        for clazz in self.get_classes():
            members = self.filtered[self.filtered["ClassId"] == int(clazz)]["ImageId"].tolist()
            members = [os.path.splitext(m)[0] for m in members]
            random.shuffle(members)
            eval_instances = members[:int(len(members) / 10) + 1]
            train_instances = [m for m in members if m not in eval_instances]
            classes[str(clazz)] = {
                "eval_instances": eval_instances,
                "train_instances": train_instances
            }
        return classes

    def iter_instances(self):
        for idx, instance in self.squashed.iterrows():
            yield instance

    def get_name(self, instance):
        return instance['ImageId'][:-4]

    def get_image(self, instance):
        return imread(os.path.join(self.path, 'Train', 'img', instance['ImageId']), as_gray=True).astype(np.float32)

    def get_masks(self, instance):
        mask = np.zeros((512, 512, 1), dtype=np.uint8)
        mask[:,:,0] = np.array(imageio.imread(os.path.join(self.path, 'Train', 'ann', instance['ImageId']), as_gray=True)).astype(np.float32)
        return mask

    def enhance_image(self, image):
        return contrast_stretch(image)