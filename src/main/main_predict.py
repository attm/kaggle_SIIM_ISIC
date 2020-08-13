from os.path import join as pjoin
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from src.model.effnet_b7_model import define_model


cwd = os.getcwd()
SUBMISSION_EXAMPLE_PATH = pjoin(cwd, "data/datasets/submission.csv")
SUBMISSION_RESULT_PATH = pjoin(cwd, "data/datasets/result_submission.csv")
SUBMISSION_DATASET = pjoin(cwd, "data/datasets/X_submission.npy")
SAVED_MODEL_PATH = pjoin(cwd, "models/complete/effnetb7")
MODEL_CP_PATH = pjoin(cwd, "models/checkpoints/effnetb7.ckpt")
USE_CPU = False

tf.keras.backend.clear_session()
# Preparing GPU device
if USE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    model = define_model(input_shape=(512, 512, 3))
    # Loading weights
    try:
        model.load_weights(MODEL_CP_PATH)
        print("\nLoaded weights from {0}".format(MODEL_CP_PATH))
    except Exception:
        print("\nCan't load weight from {0}".format(MODEL_CP_PATH))

    X = np.load(SUBMISSION_DATASET)
    print("\nLoaded X of shape {0}\n".format(X.shape))
    start_time = time.time()
    predicted_labels = model.predict(X, batch_size=1)
    predicted_labels = np.rint(predicted_labels)
    predicted_labels = predicted_labels.astype("int32")
    end_time = time.time()
    print("\nPrediction process took {0} seconds".format(end_time-start_time))
    print("\nPredicted labels shape is {0}".format(predicted_labels.shape))
    submission = pd.read_csv(SUBMISSION_EXAMPLE_PATH)
    submission['target'] = predicted_labels
    submission.to_csv(SUBMISSION_RESULT_PATH, index=False)

if __name__ == "__main__":
    main()