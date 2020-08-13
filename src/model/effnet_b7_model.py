from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def define_model(input_shape : tuple = (1024, 1024, 3)) -> Model:
    effnet = EfficientNetB7(include_top=False, input_shape=input_shape)

    layers_num = len(effnet.layers)
    percent_of_untrainable = 0.5
    first_trainable_layer = int(layers_num * percent_of_untrainable)
    i = 0
    for effnet_layer in effnet.layers:
        if i < first_trainable_layer:
            effnet_layer.trainable = False
        else:
            effnet_layer.trainable = True


    x = effnet.output
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation="sigmoid")(x)

    effnet_model = Model(effnet.input, x)

    optimizer = Adam()

    effnet_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["acc"])

    return effnet_model