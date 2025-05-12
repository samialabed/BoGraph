import torch

# Class that will replace the project context for now only does the device flag

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
