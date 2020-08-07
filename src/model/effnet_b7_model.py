from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def define_model(input_shape : tuple = (1024, 1024, 3)) -> Model:
    effnet = EfficientNetB7(include_top=False, input_shape=input_shape)
    effnet.trainable = False
    x = effnet.output
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation="sigmoid")(x)

    effnet_model = Model(effnet.input, x)

    optimizer = Adam()

    effnet_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return effnet_model