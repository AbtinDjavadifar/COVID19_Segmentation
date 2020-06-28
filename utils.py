
def create_binary_mask():

    import os
    from pathlib import Path
    import SimpleITK as sitk
    import nibabel
    import numpy as np
    import imageio
    from multiprocessing import Pool

    # for case in fullylbl_list:
    def process_fullylbl(case):
        print('Processing case', case)
        ann = nibabel.load(os.path.join(fullylbl_dir, case))
        ann = np.asanyarray(ann.dataobj)
        ann = np.flip(np.rot90(ann), 0)
        case = case[:-7]
        for j in range(np.shape(ann)[2]):
            slice = ann[:, :, j] * 20
            imageio.imwrite(os.path.join(ann_dir, '{}-{}{}'.format(case, j + 1, '.png')), slice.astype(np.uint8))

        dcm_list = [dcm for dcm in os.listdir(os.path.join(ct_dir, case)) if dcm.startswith('EXPORT')]
        for k, dcm in enumerate(dcm_list):
            img = sitk.ReadImage(os.path.join(ct_dir, case, dcm))
            img = sitk.IntensityWindowing(img, -1000, 1000, 0,
                                          255)  # rescale intensity range from [-1000,1000] to [0,255]
            img = sitk.Cast(img, sitk.sitkUInt8)  # convert 16-bit pixels to 8-bit
            sitk.WriteImage(img, os.path.join(img_dir, '{}-{}{}'.format(case, len(dcm_list) - k, '.png')))

    # for case in lbl_list:
    def process_lbl(case):
        print('Processing case', case)
        ann = nibabel.load(os.path.join(lbl_dir, case))
        ann = np.asanyarray(ann.dataobj)
        ann = np.flip(np.rot90(ann), 0)
        case = case[:-7]

        dcm_list = [dcm for dcm in os.listdir(os.path.join(ct_dir, case)) if dcm.startswith('EXPORT')]

        with open(os.path.join(lbl_dir, '{}.txt'.format(case)), "r") as txt:
            lines = list(line for line in (l.strip() for l in txt) if line)  # removing empty lines
            for x in lines:
                j = int(float(x))
                if j == 0:
                    j = 1
                slice = ann[:, :, j - 1] * 20
                imageio.imwrite(os.path.join(ann_dir, '{}-{}{}'.format(case, j, '.png')), slice.astype(np.uint8))
                # print(len(dcm_list),j)
                img = sitk.ReadImage(os.path.join(os.path.join(ct_dir, case), dcm_list[len(dcm_list) - j]))
                img = sitk.IntensityWindowing(img, -1000, 1000, 0,
                                              255)  # rescale intensity range from [-1000,1000] to [0,255]
                img = sitk.Cast(img, sitk.sitkUInt8)  # convert 16-bit pixels to 8-bit
                sitk.WriteImage(img, os.path.join(img_dir, '{}-{}{}'.format(case, j, '.png')))

    ct_dir = Path(r'./Data/COVID')
    img_dir = Path(r'./Data/Supervisely/Train/img')
    fullylbl_dir = Path(r'./Data/Fully Labeled')
    lbl_dir = Path(r'./Data/Labeled')
    ann_dir = Path(r'./Data/Supervisely/Train/ann')

    fullylbl_list = [f for f in os.listdir(fullylbl_dir) if f.endswith('.nii.gz')]
    lbl_list = [f for f in os.listdir(lbl_dir) if f.endswith('.nii.gz')]

    pool = Pool(os.cpu_count() - 2)
    pool.map(process_fullylbl, fullylbl_list)
    pool.map(process_lbl, lbl_list)

def create_csv():

    from pathlib import Path
    import os
    import pandas as pd
    import numpy as np
    import imageio

    img_dir = Path(r'./Data/Supervisely/Train/ann')
    img_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    data = pd.DataFrame(columns=['ImageId', 'ClassId'])

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

def devide_to_train_test():

    import os
    import shutil
    import random
    from pathlib import Path

    images_train = Path(r'./Data/Supervisely/Train/img')
    annotations_train = Path(r'./Data/Supervisely/Train/ann')

    images_test = Path(r'./Data/Supervisely/Test/img')
    annotations_test = Path(r'./Data/Supervisely/Test/ann')

    files = [file for file in os.listdir(images_train) if file.endswith(".png")]

    test = open("test.txt", "w")

    test_amount = round(0.15 * len(files))

    for x in range(test_amount):
        file = random.choice(files)
        files.remove(file)
        shutil.move(os.path.join(images_train, file), images_test)
        shutil.move(os.path.join(annotations_train, file), annotations_test)
        test.write(file + "\n")

def find_unique_pixel_values():

    import os
    import numpy as np
    import imageio
    from pathlib import Path

    print(np.unique(np.array(imageio.imread(os.path.join(Path(r'./Data/Supervisely/Train/ann'), '4094543-91.png')))))

def segmenter():

    from lungmask import lungmask, utils
    import SimpleITK as sitk
    from pathlib import Path
    import os
    import logging
    import numpy as np
    # import pdb

    images_path = Path("./Data/COVID")
    segmentation_path = Path("./Data/Segmentations")
    dict_path = Path("./Data/Dictionaries")
    images = [f for f in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, f))]

    # pdb.set_trace()

    logging.info(f'Load model')
    model = lungmask.get_model('unet', 'R231')
    for input in images:
        input_image = utils.get_input_image(os.path.join(images_path, input))
        logging.info(f'Infer lungmask')
        result = lungmask.apply(input_image, model, batch_size=1, volume_postprocessing=False)

        result_dict = np.zeros((np.shape(result)[0], 2))
        for i in range(np.shape(result)[0]):
            slice = result[i, :, :]
            area = np.size(np.where(slice > 0))
            result_dict[i, :] = [i + 1, area]
        result_dict = np.flip(result_dict[result_dict[:, 1].argsort()], axis=0)
        with open(os.path.join(dict_path, '{}.csv'.format(input)), "w") as outfile:
            np.savetxt(outfile, result_dict, delimiter=",")

        result_out = sitk.GetImageFromArray(result)
        result_out.CopyInformation(input_image)
        logging.info(f' %s results saved', input)
        sitk.WriteImage(result_out, os.path.join(segmentation_path, '{}.dcm'.format(input)))