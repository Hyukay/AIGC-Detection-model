import matplotlib.pyplot as plt

def plot_training(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plots training and validation accuracy and loss over epochs.
    
    Args:
        train_losses (list): List of training loss values per epoch
        val_losses (list): List of validation loss values per epoch
        train_accuracies (list): List of training accuracy values per epoch
        val_accuracies (list): List of validation accuracy values per epoch
    """
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.show()
