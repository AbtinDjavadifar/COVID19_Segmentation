from lungmask import lungmask, utils
import SimpleITK as sitk
from pathlib import Path
import os
import logging
import numpy as np
# import pdb

images_path = Path("C:/Users/djava/PycharmProjects/CoronaDetection/Data/COVID")
segmentation_path = Path("C:/Users/djava/PycharmProjects/CoronaDetection/Data/Segmentations")
dict_path = Path("C:/Users/djava/PycharmProjects/CoronaDetection/Data/Dictionaries")
images = [f for f in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, f))]
# pdb.set_trace()

def main():

    logging.info(f'Load model')
    model = lungmask.get_model('unet', 'R231')
    for input in images:
        input_image = utils.get_input_image(os.path.join(images_path, input))
        logging.info(f'Infer lungmask')
        result = lungmask.apply(input_image, model, batch_size = 1, volume_postprocessing=False)

        result_dict = np.zeros((np.shape(result)[0],2))
        for i in range(np.shape(result)[0]):
            slice = result[i,:,:]
            area = np.size(np.where(slice > 0))
            result_dict[i,:] = [i+1, area]
        result_dict = np.flip(result_dict[result_dict[:, 1].argsort()], axis = 0)
        with open(os.path.join(dict_path,'{}.csv'.format(input)), "w") as outfile:
            np.savetxt(outfile, result_dict, delimiter=",")

        result_out= sitk.GetImageFromArray(result)
        result_out.CopyInformation(input_image)
        logging.info(f' %s results saved', input)
        sitk.WriteImage(result_out, os.path.join(segmentation_path, '{}.dcm'.format(input)))

if __name__ == "__main__":
    print('called as script')
    main()
