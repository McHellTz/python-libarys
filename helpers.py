
import matplotlib.pyplot as plt
import numpy as np

import keras.models
import tensorflow as tf
from tensorflow.keras.callbacks import History
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter



def plot_learning_curve(
    title: str, x: int, y: int, y_test: int, ylim: float = 0.6) -> None:
    plt.figure()
    plt.title(title)
    axes = plt.gca()
    axes.set_ylim([ylim, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    train_sizes = x
    train_scores = y
    test_scores = y_test

    plt.grid()

    plt.plot(
        train_sizes,
        train_scores,
        "o-",
        color=(177 / 255, 6 / 255, 58 / 255),
        label="Training accuracy",
    )
    plt.plot(
        train_sizes,
        test_scores,
        "o-",
        color=(246 / 255, 168 / 255, 0),
        label="Validation accuracy",
    )

    plt.legend(loc="best")

def plot_history(title: str, history: "History", ylim: float = 0.6) -> None:
    y = history.history["accuracy"]
    y_test = history.history["val_accuracy"]
    plot_learning_curve(title, np.arange(1, 1 + len(y)), y, y_test, ylim)


def plot_two_histories(history: "History", history_finetune: "History") -> None:
    y = history.history["accuracy"] + history_finetune.history["accuracy"]
    y_test = history.history["val_accuracy"] + history_finetune.history["val_accuracy"]
    plot_learning_curve("Transfer Learning", np.arange(1, 1 + len(y)), y, y_test, 0)