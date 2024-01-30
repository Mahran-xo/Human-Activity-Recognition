import torch
import torch.nn as nn
import torch.optim as optim
from model import build_model
import config
import albumentations as A
from albumentations.pytorch import ToTensorV2
from plots import save_plots
from tqdm import tqdm
from utils import (
    get_loaders,
    check_accuracy,
    save_checkpoint)


def train_fn(train_loader, DEVICE, model, optim, loss_fn, scaler):
    """
    :param train_loader:
    :param DEVICE:
    :param model:
    :param optim:
    :param loss_fn:
    :param scaler:
    :return:
    """
    prog_bar = tqdm(
        train_loader,
        total=len(train_loader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    train_loss = 0.0
    train_correct = 0
    cnt = 0
    # Training
    for idx, batch in enumerate(prog_bar):
        cnt += 1
        # Get a batch of 15 frames features and labels
        features, labels = batch['image'].to(DEVICE), batch['label'].to(DEVICE)

        optim.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(features)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        # Accumulate training loss and accuracy
        _, predicted = torch.max(outputs, 1)
        train_loss += loss.item()
        train_correct += (predicted == labels).sum().item()

    train_loss /= cnt  # Normalize the loss by the total number of examples
    train_accuracy = 100 * (train_correct / len(train_loader.dataset))
    print('Train Loss: {:.4f}, Train Accuracy: {:.2f}%'.format(train_loss, train_accuracy))
    return train_loss, train_accuracy


def main():
    model = build_model(True, 15).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.RandomRotate90(p=0.5),
        A.RandomContrast(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    train_loader, val_loader = get_loaders(
        config.TRAIN_DF,
        config.TEST_DF,
        config.TRAIN_DIR,
        config.TEST_DIR,
        train_transform,
        test_transform,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.PIN_MEMORY
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[7], gamma=0.1, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(config.NUM_EPOCHS):
        best_val_loss = 0.0
        train_loss, train_accuracy = train_fn(train_loader, config.DEVICE, model, optimizer, loss_fn, scaler)
        test_loss, test_accuracy = check_accuracy(val_loader, model, loss_fn,
                                                                                         config.DEVICE)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            print('Validation loss improved')
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'val_acc': test_loss,
            }

            save_checkpoint(checkpoint)
        else:
            print('Validation loss did not improve')

    scheduler.step()
    print('-'*50)

    save_plots(train_accuracies, test_accuracies, train_losses, test_losses)

if __name__ == "__main__":
    main()
