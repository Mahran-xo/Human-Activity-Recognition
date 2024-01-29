import torch
import torch.nn as nn
import torch.optim as optim
from model import build_model
import config
import albumentations as A
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import (
    get_loaders,
    check_accuracy,
    save_checkpoint,
    load_checkpoint
)


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
        features, labels = batch['frames'].to(DEVICE), batch['label'].to(DEVICE)

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
    model = build_model(True, 10).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_transform = A.Compose([
        A.ToPILImage(),
        A.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        A.RandomRotation(45),
        A.RandomAutocontrast(p=0.5),
        A.ToTensor(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = A.Compose([
        A.ToPILImage(),
        A.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        A.ToTensor(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_loader, val_loader = get_loaders(
        config.TRAIN_DF,
        config.TEST_DF,
        train_transform,
        test_transform,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    all_test_preds = []
    all_labels = []

    for epoch in range(config.NUM_EPOCHS):
        best_val_loss = 0.0
        train_loss, train_accuracy = train_fn(train_loader, config.DEVICE, model, optimizer, loss_fn, scaler)
        test_loss, test_accuracy, test_correct, test_preds, test_labels = check_accuracy(val_loader, model, loss_fn,
                                                                                         DEVICE)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        all_test_preds.extend(test_preds.cpu().detach().numpy())
        all_labels.extend(test_labels.cpu().detach().numpy())

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

    report = classification_report(all_labels, all_test_preds)
    report_filename = 'evaluation_report.txt'
    with open(report_filename, 'w') as file:
        file.write(report)

    # Plot training and testing losses and accuracies
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
