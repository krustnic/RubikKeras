import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from src.models import dice_coef, bce_dice_loss_double, dice_coef_loss_double, get_unet_96
from src.samplers import seg_mask_with_ego_and_center_generator
from src.utils import visualize


BATCH_SIZE = 16
EPOCH = 450
SIZE = 256


TRAIN = 'X:\\datasets\\rubik_ds\\rds_train'
VALID = 'X:\\datasets\\rubik_ds\\rds_valid'
EGO = 'X:\\datasets\\egohands_all_frames'

t_gen = seg_mask_with_ego_and_center_generator(TRAIN, EGO, batch_size=BATCH_SIZE, channels=1, is_aug=True, size=SIZE, radius=7) # r = 3
v_gen = seg_mask_with_ego_and_center_generator(VALID, EGO, batch_size=18, channels=1, is_aug=False, size=SIZE, radius=7)

# for images, points in t_gen:
#     print(images.shape, points.shape)
#     mask = points[0][:, :, 1]
#
#     visualize(np.squeeze(points[0][:, :, 0]), np.squeeze(points[0][:, :, 1]))

model = get_unet_96(input_shape=(SIZE, SIZE, 1), num_classes=2, dropout=None)
model.summary()

model.load_weights('E:\\DataSets\\rubik\\models\\seg-unet96-rds4-s256-3-13.h5')

model.compile(loss=dice_coef_loss_double, optimizer=keras.optimizers.Adam(lr=2e-3), metrics=[dice_coef]) # 2e-3
# model.compile(loss=bce_dice_loss_double, optimizer=keras.optimizers.Adam(lr=1e-4), metrics=[dice_coef])

model_checkpoint = ModelCheckpoint('E:\\DataSets\\rubik\\models\\seg-unet96-rds4-s256-4-{epoch:02d}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   period=1)

reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='min')

history_1 = model.fit_generator(
    t_gen,
    steps_per_epoch=450 // BATCH_SIZE,
    epochs=EPOCH,
    callbacks=[model_checkpoint, reduceLROnPlato],
    validation_data=v_gen,
    validation_steps=1,
    verbose=1
)