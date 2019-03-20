import random

import numpy as np

from src.models import get_unet_96
from src.samplers import seg_mask_with_ego_and_center_generator
from src.utils import visualize, visualize3

random.seed(7)

BATCH_SIZE = 1
SIZE = 128

t_gen = seg_mask_with_ego_and_center_generator('X:\\datasets\\rubik_ds\\rds_train', 'X:\\datasets\\egohands_all_frames', batch_size=BATCH_SIZE, channels=1, is_aug=True, size=SIZE, radius=3)
v_gen = seg_mask_with_ego_and_center_generator('X:\\datasets\\rubik_ds\\rds_valid', 'X:\\datasets\\egohands_all_frames', batch_size=3, channels=1, is_aug=False, size=SIZE, radius=3)

model = get_unet_96(input_shape=(SIZE, SIZE, 1), num_classes=2)
model.summary()

model.load_weights('E:\\DataSets\\rubik\\models\\seg-unet96-rds4-s128-e2-2-86.h5')

for images, points in v_gen:
    image = images[0]

    prediction = model.predict(images)
    mask = prediction[0][:, :, 0]
    mask_center = prediction[0][:, :, 1]

    visualize3(np.squeeze(image), np.squeeze(mask), np.squeeze(mask_center))