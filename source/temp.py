import cv2
import random
import numpy as np
from PIL import Image
import albumentations as album
from scipy.spatial import procrustes


true_xy_points = np.array([[0, 0], [1, 0]])
xy_points = np.array([[0, 1], [1, 1]])

mtx1, mtx2, disparity = procrustes(true_xy_points, xy_points)

print("mtx : ", mtx1, mtx2)