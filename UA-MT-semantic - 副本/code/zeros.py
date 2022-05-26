import os
from PIL import Image
import numpy as np


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def tif_it(filename,output_dir):
    image = Image.open(filename)
    img_arr = np.asarray(image).astype(np.float32)

    img_shape = img_arr.shape
    new = np.zeros((img_shape[0], img_shape[1]), np.uint8)

    new_im = Image.fromarray(new)

    new_im.save(os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + '.png'))
    new_im.close()


if __name__ == '__main__':
    input_dir = "../data/ICA/train/unlabeled"
    output_dir = "../data/ICA/train/masks/0"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    img_path_list = all_files_under(input_dir, extension=".png")
    for filename in img_path_list:
        print("filename", filename)
        tif_it(filename, output_dir)
