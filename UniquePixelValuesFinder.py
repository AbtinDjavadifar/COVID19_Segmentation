import os
import numpy as np
import imageio
from pathlib import Path

name = '4094543-91.png'
labels_path = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Supervisely/Train/ann')
im = np.array(imageio.imread(os.path.join(labels_path, name)))
print(np.unique(im))