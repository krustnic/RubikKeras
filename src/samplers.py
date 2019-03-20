import os
import random

import cv2
import pandas as pd
import scipy.misc
import albumentations as A
import PIL
import numpy as np

from src.utils import make_gaussian


def ego_generator(path, size=128, is_random=True):
    random_number = 7
    folders = [f for f in os.listdir(path)]
    while True:
        if is_random:
            folder = random.choice(folders)
            frame_number = random.randint(0, 100)
        else:
            folder = folders[7]
            frame_number = random_number

        frame_name = 'frame_%s.jpg' % str(frame_number).zfill(4)

        image = scipy.misc.imread(os.path.join(path, folder, frame_name))
        image = cv2.resize(image, (640, 480))
        image = A.RandomCrop(height=size, width=size, p=1.0)(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = np.reshape(image, (size, size, 1))

        yield image


def seg_mask_with_ego_and_center_generator(rubik_path, ego_path, batch_size=32, size=256, channels=1, is_aug=True, radius=3):
    image_batch = []
    point_batch = []

    augs = A.Compose([
        A.Blur(blur_limit=5, p=0.5),
        A.RandomGamma(p=0.5),
        A.RandomBrightness(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RGBShift(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.5)
    ])

    image_names = [f for f in os.listdir(rubik_path) if f.find('csv') == -1]
    ego_gen = ego_generator(ego_path, size=size, is_random=is_aug)

    while True:
        random.shuffle(image_names)

        for image_name in image_names:
            name = image_name.split('.')[0]

            image = scipy.misc.imread(os.path.join(rubik_path, image_name))
            image = PIL.Image.fromarray(image)

            if channels == 1:
                result = np.array(image.convert('L'))
            else:
                result = np.array(image)

            df = pd.read_csv(os.path.join(rubik_path, '%s.csv' % name), header=None)
            row = df.iloc[0]

            k = size / result.shape[0]

            points = [
                (row[1], row[2]),
                (row[3], row[4]),
                (row[5], row[6]),
                (row[7], row[8]),
                (row[9], row[10]),
                (row[11], row[12]),
                # (row[13], row[14])
            ]

            points = [(p[0] *k, p[1] * k) for p in points]
            ctr = np.array(points).reshape((-1, 1, 2)).astype(np.int32)

            result = cv2.resize(result, (size, size))

            # Draw mask
            mask = np.zeros((result.shape[0], result.shape[0]), np.uint8)
            cv2.drawContours(mask, [ctr], 0, (255, 255, 255), -1)

            bg_image = next(ego_gen)

            # Draw center
            mask_with_center = np.zeros((mask.shape[0], mask.shape[1], 2))
            mask_with_center[:, :, 0] = mask / 255
            mask_with_center[:, :, 1] = make_gaussian(mask.shape[0], mask.shape[0], sigma=radius,
                                                      center=(row[13] * k, row[14] * k))

            if is_aug:
                augment = augs(image=result, mask=mask_with_center)
                result = augment['image']
                mask_with_center = augment['mask']

                bg_image = augs(image=bg_image)['image']

            if channels == 1:
                result = np.reshape(result, result.shape + (1,))
                bg_image = np.reshape(bg_image, bg_image.shape + (1,))

            image_batch.append(result)
            image_batch.append(bg_image)

            point_batch.append(mask_with_center)
            point_batch.append(np.zeros((size, size, 2)))

            if len(image_batch) // 2 == batch_size:
                image_batch = np.array(image_batch) / 255
                point_batch = np.array(point_batch)

                yield image_batch, point_batch
                image_batch = []
                point_batch = []
