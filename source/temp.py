import cv2
import random
import numpy as np
from PIL import Image
import albumentations as album

arg = 'all'


if arg in ['wfhad', 'all']:
    print("pass ")

# albumtransform = album.Compose([album.CoarseDropout(max_holes=4, max_height=60, max_width=60, min_holes=1, min_height=20, min_width=20, p=0.4)])
# # albumtransform = album.Compose([album.MaskDropout(mask_fill_value=1)])
#
# file_name = 'C:/Research/dataset/HO3D_v2/train/ABF10/rgb/0000.png'
# mask = np.zeros((416, 416))
# mask[:50, :] = 1
# mask[-50:, :] = 1
# mask[:, :50] = 1
# mask[:, -50:] = 1
#
# for i in range(100):
#     img = Image.open(file_name)
#     img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
#     if img.shape[-1] != 3:
#         img = img[:, :, :-1]
#
#     img = img / 255.
#
#     img_bef = img[:, :, [0, 1, 2]]
#     cv2.imshow("before augment", img_bef)
#     cv2.waitKey(1)
#
#     transformed = albumtransform(image=img)
#
#     img = transformed['image']
#     img_after = img[:, :, [0, 1, 2]]
#     cv2.imshow("after augment", img_after)
#     cv2.waitKey(0)