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


if __name__ == '__main__':
    input_dir = "../semantic"
    output_dir = "../semantic"

    patient_path_list = all_files_under(input_dir)
    for patient in patient_path_list:

        img_path_list = all_files_under(patient+"/LAD/", extension=".png")
        video_count = len(all_files_under(patient+"/LAD/data/", extension=".vid"))
        seg_path = []
        print("patient", patient)
        for filename in img_path_list:

            if 'seg' in filename:
                seg_path.append(filename)
        count_1 = 2
        count_2 = 14
        for v in range(video_count):
            merge = np.zeros((512, 512), np.uint8)
            for filename_seg in seg_path[count_1+14*v:count_2+14*v]:
                image = Image.open(filename_seg)
                img_arr = np.asarray(image).astype(np.uint8)
                merge = image + merge
            for i in range(512):
                for j in range(512):
                    if merge[i][j] > 0:
                        merge[i][j] = 255
            new_im_1 = Image.fromarray(merge)

            output_dir = patient+"/LAD/4ch_merge/"
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            new_im_1.save(os.path.join(output_dir, os.path.splitext(os.path.basename(seg_path[count_1+14*v]))[0] + '.png'))
            new_im_1.close()
            merge = np.zeros((512, 512), np.uint8)