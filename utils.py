import torch
import torchvision
from dataset import HARDataset
from torch.utils.data import DataLoader
import config

from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)



def get_loaders(
        train_df,
        test_df,
        train_transform,
        test_transform,
        batch_size,
        num_workers=4,
        pin_memory=True,
):
    """

    :param train_df:
    :param test_df:
    :param train_transform:
    :param test_transform:
    :param batch_size:
    :param num_workers:
    :param pin_memory:
    :return:
    """


    # Create DataLoader with the WeightedRandomSampler
    # train_loader = DataLoader(dataset=hard_dataset, batch_size=your_batch_size, sampler=weighted_sampler)
    train_ds = HARDataset(
        df=train_df,
        transform=train_transform

    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )

    val_ds = HARDataset(
        df=test_df,
        transform=test_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=True
    )

    return train_loader, val_loader

def check_accuracy(test_loader, model, loss_fn, DEVICE="cuda"):
    test_loss = 0.0
    test_correct = 0
    cnt = 0
    all_true_labels = []
    all_predicted_labels = []

    prog_bar = tqdm(
        test_loader,
        total=len(test_loader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    model.eval()
    with torch.no_grad():
        for idx, test_batch in enumerate(prog_bar):
            cnt += 1
            features, labels = test_batch['image'].float().to(DEVICE), test_batch['label'].to(DEVICE)

            
            outputs = model(features)
            loss = loss_fn(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            test_loss += loss.item()

            test_correct += (predicted == labels).sum().item()

            # Append true and predicted labels to the lists
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

    test_loss /= cnt
    test_accuracy = 100 * (test_correct / len(test_loader.dataset))
    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, test_accuracy))

    # Convert lists to NumPy arrays
    true_labels_array = np.array(all_true_labels)
    predicted_labels_array = np.array(all_predicted_labels)

    # Generate and print the classification report
    report = classification_report(true_labels_array, predicted_labels_array)
    print("Classification Report:\n", report)
    with open('metrics/classification_report.txt', 'w') as file:
        file.write("Classification Report:\n")
        file.write(report)

    return test_loss, test_accuracy
