import os
import torch


TRAIN_DF = os.path.join('..', 'input', 'Human Action Recognition', 'test.csv')
TEST_DF = os.path.join('..', 'input', 'Human Action Recognition', 'test.csv')

# hyperparams
LEARNING_RATE = 1e-4
DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_EPOCHS = 200
PIN_MEMORY = True
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
NUM_WORKERS = 4 # Number of parallel processes for data preparation.