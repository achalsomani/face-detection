import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 100
STARTING_SHAPE = (275, 275)
INPUT_SHAPE = (225, 225)

LEARNING_RATE = 0.001
NUM_EPOCHS = 300