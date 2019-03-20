from PIL import Image
from src.models import get_unet_96
import albumentations as A

from src.utils import *


OUTPUT = 'X:\\datasets\\rubikjs\\colors\\source'
SIZE = 128
MODEL_PATH = 'models\\seg-unet96-rds4-s128-e2-2-86.h5'

model = get_unet_96(input_shape=(SIZE, SIZE, 1), num_classes=2)
model.load_weights(MODEL_PATH)

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    frame = A.CenterCrop(height=480, width=480)(image=frame)['image']
    copy_frame = frame.copy()

    gray = Image.fromarray(frame, 'RGB').convert('L')

    im = gray.resize((SIZE, SIZE))
    img_array = np.array(im) / 255
    img_array = np.reshape(img_array, (SIZE, SIZE, 1))

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    mask = prediction[0][:, :, 0]
    mask = cv2.resize(mask, (480, 480))

    center = prediction[0][:, :, 1]
    center = cv2.resize(center, (480, 480))

    copy_frame[mask < 0.5, :] = 0
    copy_frame[center > 0.5] = 0

    mask[mask < 0.5] = 0
    mask[mask != 0] = 255
    mask = np.array(mask, np.uint8)

    center = np.array(center * 255, np.uint8)

    mask_copy = mask.copy()
    binary_mask = mask
    binary_center = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    approx = get_corners(binary_mask, alpha=0.03)

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)

    if approx is None:
        # print('Bad frame')
        continue

    center_point = get_center_point(binary_center)

    if center_point is None:
        # print('Bad center point')
        continue

    points = [(p[0][0], p[0][1]) for p in approx]

    rect_left = np.array([points[1], center_point, points[3], points[2]])
    rect_top = np.array([points[0], points[5], center_point, points[1]])
    rect_forward = np.array([center_point, points[5], points[4], points[3]])

    side_left = four_points_resize(frame, rect_left, 256)
    side_top = four_points_resize(frame, rect_top, 256)
    side_forward = four_points_resize(frame, rect_forward, 256)

    cv2.circle(mask, points[5], 5, (0, 255, 0), -1)

    cv2.drawContours(mask, [approx], 0, (147, 0, 255), 3)
    cv2.circle(mask, center_point, 5, (147, 0, 255), -1)

    cv2.drawContours(frame, [rect_left], 0, (147, 0, 255), 3)
    cv2.drawContours(frame, [rect_top], 0, (255, 0, 0), 3)
    cv2.drawContours(frame, [rect_forward], 0, (0, 255, 0), 3)

    cv2.line(mask, points[1], points[0], (0, 255, 0), thickness=3, lineType=8)
    cv2.line(mask, center_point, points[5], (0, 255, 0), thickness=3, lineType=8)

    if is_hexagon_valid(points, center_point, epsilon=10):
        mask[mask==0] = 45

        cv2.imshow('left', side_left)
        cv2.imshow('top', side_top)
        cv2.imshow('forward', side_forward)

        cv2.imshow("Capturing", frame)

    cv2.imshow("masks", binary_mask)

    key=cv2.waitKey(1)
    if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()