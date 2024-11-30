from utils.globalConst import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

def smooth(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(results, epoch, window_size=5):
    clear_output(wait=True)
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plotting loss
    axs[0].plot(range(epoch + 1), results['train_loss'], label="Train Loss", color="blue")
    axs[0].plot(range(epoch + 1), results['val_loss'], label="Validation Loss", color="orange")

    # Adding smoothed loss with horizontal shift
    smoothed_train_loss = smooth(results['train_loss'], window_size)
    smoothed_val_loss = smooth(results['val_loss'], window_size)
    offset = [None] * (window_size // 2)  # Padding for horizontal shift
    axs[0].plot(range(len(offset) + len(smoothed_train_loss)), offset + smoothed_train_loss.tolist(),
                label="Smoothed Train Loss", color="blue", linestyle="--")
    axs[0].plot(range(len(offset) + len(smoothed_val_loss)), offset + smoothed_val_loss.tolist(),
                label="Smoothed Validation Loss", color="orange", linestyle="--")

    axs[0].set_title("Loss over Epochs", fontsize=16)
    axs[0].set_xlabel("Epochs", fontsize=14)
    axs[0].set_ylabel("Loss", fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].set_ylim(0, 3)
    axs[0].grid()

    # Plotting accuracy
    axs[1].plot(range(epoch + 1), results['train_acc'], label="Train Accuracy", color="blue")
    axs[1].plot(range(epoch + 1), results['val_acc'], label="Validation Accuracy", color="orange")

    # Adding smoothed accuracy with horizontal shift
    smoothed_train_acc = smooth(results['train_acc'], window_size)
    smoothed_val_acc = smooth(results['val_acc'], window_size)
    axs[1].plot(range(len(offset) + len(smoothed_train_acc)), offset + smoothed_train_acc.tolist(),
                label="Smoothed Train Accuracy", color="blue", linestyle="--")
    axs[1].plot(range(len(offset) + len(smoothed_val_acc)), offset + smoothed_val_acc.tolist(),
                label="Smoothed Validation Accuracy", color="orange", linestyle="--")

    axs[1].set_title("Accuracy over Epochs", fontsize=16)
    axs[1].set_xlabel("Epochs", fontsize=14)
    axs[1].set_ylabel("Accuracy (%)", fontsize=14)
    axs[1].legend(fontsize=12)
    axs[1].grid()

    plt.tight_layout()
    plt.show()