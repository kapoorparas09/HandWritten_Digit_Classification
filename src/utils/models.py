import tensorflow as tf
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_model(LOSS_FUNCTION,OPTIMIZER, METRICS, Num_classes ):
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(Num_classes, activation="softmax", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=METRICS)

    return model_clf

# Creating unique names for models
def get_unique_name(filename):
    unique_filename = time.strftime(f"_%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_name(model_name)
    path_to_model= os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

def save_plot(history, plot_name, plot_dir):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    unique_filename = get_unique_name(plot_name)
    path_to_plot= os.path.join(plot_dir, unique_filename)
    plt.savefig(path_to_plot)