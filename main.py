from utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Covid19')
    parser.add_argument('--mode', help='It can be segmenter or supervisely', required=True)
    args = vars(parser.parse_args())

    if args['mode'] == 'segmenter':
        """
        segmenting the images using lungmask repo
        """
        segmenter()

    elif args['mode'] == 'supervisely':
        """
        preprocessing images for training via supervisely
        """
        create_binary_mask()
        create_csv()
        devide_to_train_test()
        find_unique_pixel_values()

    else:
        print("mode argument must be set to segmenter or supervisely")