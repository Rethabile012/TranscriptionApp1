import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history_path="training_history.npz"):
    # Load saved history
    history = np.load(history_path)

    train_ce = history["train_ce"]
    train_cer = history["train_cer"]
    train_wer = history["train_wer"]
    val_cer = history["val_cer"]
    val_wer = history["val_wer"]

    epochs = range(1, len(train_ce) + 1)

    # Plot Cross-Entropy loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ce, label="Train CE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Cross-Entropy)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot CER
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_cer, label="Train CER")
    plt.plot(epochs, val_cer, label="Validation CER")
    plt.xlabel("Epoch")
    plt.ylabel("CER")
    plt.title("Character Error Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot WER
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_wer, label="Train WER")
    plt.plot(epochs, val_wer, label="Validation WER")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.title("Word Error Rate")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_training_history()
