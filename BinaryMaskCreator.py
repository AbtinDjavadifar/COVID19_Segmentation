import os
from pathlib import Path
import SimpleITK as sitk
import nibabel
import numpy as np
import imageio
from multiprocessing import Pool

ct_dir = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/COVID')
img_dir = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Supervisely/Train/img')
fullylbl_dir = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Fully Labeled')
lbl_dir = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Labeled')
ann_dir = Path(r'C:/Users/djava/PycharmProjects/CoronaDetection/Data/Supervisely/Train/ann')

fullylbl_list = [f for f in os.listdir(fullylbl_dir) if f.endswith('.nii.gz')]
lbl_list = [f for f in os.listdir(lbl_dir) if f.endswith('.nii.gz')]

# for case in fullylbl_list:
def process_fullylbl(case):
    print('Processing case', case)
    ann = nibabel.load(os.path.join(fullylbl_dir, case))
    ann = np.asanyarray(ann.dataobj)
    ann = np.flip(np.rot90(ann), 0)
    case = case[:-7]
    for j in range(np.shape(ann)[2]):
        slice = ann[:,:,j] * 20
        imageio.imwrite(os.path.join(ann_dir, '{}-{}{}'.format(case, j+1, '.png')), slice.astype(np.uint8))

    dcm_list = [dcm for dcm in os.listdir(os.path.join(ct_dir, case)) if dcm.startswith('EXPORT')]
    for k, dcm in enumerate(dcm_list):
        img = sitk.ReadImage(os.path.join(ct_dir, case, dcm))
        img = sitk.IntensityWindowing(img, -1000, 1000, 0, 255) # rescale intensity range from [-1000,1000] to [0,255]
        img = sitk.Cast(img, sitk.sitkUInt8) # convert 16-bit pixels to 8-bit
        sitk.WriteImage(img, os.path.join(img_dir, '{}-{}{}'.format(case, len(dcm_list)-k, '.png')))

# for case in lbl_list:
def process_lbl(case):
    print('Processing case', case)
    ann = nibabel.load(os.path.join(lbl_dir, case))
    ann = np.asanyarray(ann.dataobj)
    ann = np.flip(np.rot90(ann), 0)
    case = case[:-7]

    dcm_list = [dcm for dcm in os.listdir(os.path.join(ct_dir, case)) if dcm.startswith('EXPORT')]

    with open(os.path.join(lbl_dir, '{}.txt'.format(case)), "r") as txt:
        lines = list(line for line in (l.strip() for l in txt) if line) #removing empty lines
        for x in lines:
            j = int(float(x))
            if j == 0:
                j = 1
            slice = ann[:,:,j-1] * 20
            imageio.imwrite(os.path.join(ann_dir, '{}-{}{}'.format(case, j, '.png')), slice.astype(np.uint8))
            # print(len(dcm_list),j)
            img = sitk.ReadImage(os.path.join(os.path.join(ct_dir, case), dcm_list[len(dcm_list)-j]))
            img = sitk.IntensityWindowing(img, -1000, 1000, 0, 255)  # rescale intensity range from [-1000,1000] to [0,255]
            img = sitk.Cast(img, sitk.sitkUInt8)  # convert 16-bit pixels to 8-bit
            sitk.WriteImage(img, os.path.join(img_dir, '{}-{}{}'.format(case, j, '.png')))

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(process_fullylbl, fullylbl_list)
    pool.map(process_lbl, lbl_list)