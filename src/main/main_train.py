import tensorflow as tf
import os
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from src.model.effnet_b7_model import define_model


cwd = os.getcwd()
TRAIN_DATA_PATH = pjoin(cwd, "data/datasets")
MODEL_CP_PATH = pjoin(cwd, "models/effnetb7/effnetb7")
# Training params
USE_CPU = False
BATCH_SIZE = 4
EPOCHS = 5

tf.keras.backend.clear_session()
# Preparing GPU device
if USE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

def main() -> None:
    model = define_model(input_shape=(512, 512, 3))
    # Loading weights
    if os.path.exists(MODEL_CP_PATH):
        model.load_weights(MODEL_CP_PATH)
    
    print(model.summary())
    X, y = load_data()
    print("Loaded data: X shape is {0}, y shape is {1}".format(X.shape, y.shape))

    save_cp = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_CP_PATH,
                                                 save_best_only=True,
                                                 save_weights_only=True,)

    model.fit(X, y, validation_split=0.25, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[save_cp])

def load_data() -> np.ndarray:
    X = np.load(pjoin(TRAIN_DATA_PATH, "X_train.npy"))
    y = np.load(pjoin(TRAIN_DATA_PATH, "y_train.npy"))
    return X, y

if __name__ == "__main__":
    main()