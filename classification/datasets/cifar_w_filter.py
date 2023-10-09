import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

FOLDER_ALIAS = {
    "sedan": "automobile",
    "suv": "automobile",
}


def rgba2rgb(src_path, dst_path):
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)

    # check if the image has an alpha channel
    if img.shape[2] == 4:
        # check if the alpha channel is fully opaque
        if np.all(img[:, :, 3] == 255):
            # extract the RGB channels and discard the alpha channel
            img_rgb = img[:, :, :3]
        else:
            # fill the transparent pixels with a background color
            bg_color = (255, 255, 255) # white
            alpha = img[:, :, 3].astype(float) / 255.0
            alpha = np.dstack([alpha, alpha, alpha])
            bg = np.ones_like(img[:, :, :3], dtype=float) * bg_color
            img_rgb = alpha * img[:, :, :3].astype(float) + (1 - alpha) * bg
            img_rgb = img_rgb.astype(np.uint8)
    else:
        # image does not have an alpha channel, just extract the RGB channels
        img_rgb = img[:, :, :3]

    # save the new image with three channels
    cv2.imwrite(dst_path, img_rgb)


def filter_single(img_path, dst_path):
    img = Image.open(img_path)
    if img.mode == "RGBA":
        rgba2rgb(img_path, dst_path)
        return
    img_arr = np.asarray(img)
    if img_arr.shape[-1] < 3:
        raise ValueError(f"{img_path} fewer than 3 channels")
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img.save(dst_path)


def filter_multiple(imgs_path, tar_path):
    # the imgs_path is a Path object
    assert isinstance(imgs_path, Path)
    assert isinstance(tar_path, Path)
    for i, file in enumerate(imgs_path.iterdir()):
        try:
            filter_single(file, os.path.join(tar_path, f"{str(i).zfill(4)}.jpg"))
        except Exception as e:
            print(f"File: {file} skipped due to processing failure {e}")


def main():
    base = "/home/uqzxwang/data/TTA/cifar-w/"
    tar = "/home/uqzxwang/data/TTA/cifar-w_filtered/"
    for meta_name in sorted(os.listdir(base)):
        base_dir = os.path.join(base, meta_name)
        tar_dir = os.path.join(tar, meta_name)
        if not os.path.exists(tar_dir):
            os.makedirs(tar_dir)
            # indent here for not repetitively filter dataset already been filtered
            for dataset_name in tqdm(os.listdir(base_dir)):
                src = Path(os.path.join(base_dir, dataset_name))
                dst = Path(os.path.join(tar_dir, dataset_name))
                for class_name in os.listdir(src):
                    src_cls = Path(os.path.join(src, class_name))
                    dst_cls = Path(os.path.join(dst, class_name))
                    if not os.path.exists(dst_cls):
                        os.makedirs(dst_cls)
                    filter_multiple(src_cls, dst_cls)


if __name__ == "__main__":
    main()
