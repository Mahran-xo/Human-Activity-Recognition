import torch
import torchvision
from dataset import HARDataset
from torch.utils.data import DataLoader
import config
from tqdm import tqdm


def save_checkpoint(state, filename="../my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


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
    train_ds = HARDataset(
        df=train_df,
        transform=train_transform

    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
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
    )

    return train_loader, val_loader


def check_accuracy(test_loader, model, loss_fn, DEVICE="cuda"):
    """
    :param test_loader:
    :param DEVICE:
    :param loss_fn:
    :param loader:
    :param model:
    :param DEVICE:
    :return:
    """
    test_loss = 0.0
    test_correct = 0
    cnt=0
    prog_bar = tqdm(
        test_loader,
        total=len(test_loader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    model.eval()

    with torch.no_grad():
        for idx, test_batch in enumerate(prog_bar):
            cnt+=1
            features, labels = test_batch['frames'].to(DEVICE), test_batch['label'].to(DEVICE)

            outputs = model(features)
            loss = loss_fn(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            test_loss += loss.item()

            test_correct += (predicted == labels).sum().item()

    test_loss /= cnt
    test_accuracy = 100 * (test_correct / len(test_loader.dataset))
    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
    model.train()
    return test_loss, test_accuracy
