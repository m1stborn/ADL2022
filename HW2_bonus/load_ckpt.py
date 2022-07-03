import torch


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path)
    return ckpt


ckpt = load_checkpoint("../HW1/Slot-64a3079a.pt")
ckpt2 = load_checkpoint("../HW1/ckpt/intent/Intent-1ec4d12e.pt")
print(ckpt['acc'])
print(ckpt2['acc'])
