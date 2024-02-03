import os
import torch


TRAIN_DF = "Human Action Recognition/train.csv"
TEST_DF = "Human Action Recognition/test.csv"
# hyperparams
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 35
PIN_MEMORY = True
IMAGE_SIZE = 200 # Image size of resize when applying transforms.
NUM_WORKERS = 4 # Number of parallel processes for data preparation.