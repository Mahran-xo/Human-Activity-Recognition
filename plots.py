import matplotlib.pyplot as plt
import os


def save_plots(train_accuracies, test_accuracies, train_losses, test_losses, out_dir='ActivityRecognition/metrics'):
    """
    saving the plots for train and test accuracies and losses
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_accuracies, color='tab:blue', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        test_accuracies, color='tab:red', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_losses, color='tab:blue', linestyle='-',
        label='train loss'
    )
    plt.plot(
        test_losses, color='tab:red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
