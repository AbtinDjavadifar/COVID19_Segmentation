from pathlib import Path
import os
import pandas as pd
import numpy as np
import imageio

img_dir = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Supervisely/Train/ann')
img_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
data = pd.DataFrame(columns=['ImageId','ClassId'])

for name in img_list:
    ImageId = name
    im = np.array(imageio.imread(os.path.join(img_dir, name)))
    if len(np.unique(im)) > 1:
        ClassId = 1
    else:
        ClassId = None
    data = data.append({'ImageId': ImageId, 'ClassId': ClassId}, ignore_index=True)

filtered = data.dropna(subset=['ClassId'], axis='rows')
data.to_csv('train.csv', index=False)
filtered.to_csv('filtered.csv', index=False)