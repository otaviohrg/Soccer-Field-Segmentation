import os
import torch


def save_checkpoint(state, checkpoint_path, filename):
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_path, filename))


def load_checkpoint(checkpoint_path, filename, model, optimizer):
    path = os.path.join(checkpoint_path, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]
