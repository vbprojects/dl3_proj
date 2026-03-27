import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 10)
        self.head = nn.Linear(10, 1000)
    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)

model = MyModel()
# we want to train `encoder` but avoid running `head`!
storage = []
class StopEx(Exception): pass
def hook(m, inp, out):
    storage.append(out)
    raise StopEx()

handle = model.encoder.register_forward_hook(hook)

try:
    model(torch.randn(2, 10))
except StopEx:
    pass

res = storage[0]
loss = res.sum()
loss.backward()
print("Grad:", model.encoder.weight.grad is not None)
print("Head Grad:", model.head.weight.grad is not None)
