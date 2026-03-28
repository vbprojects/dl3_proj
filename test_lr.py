import torch
import torch.nn as nn
from pytorch_metric_learning import losses

e_size = 100
loss_func = losses.SupConLoss(temperature=0.1)

# Ensure we optimize both model (PEFT params) AND the ProxyAnchor embeddings
cbm = losses.CrossBatchMemory(loss_func, e_size, memory_size=1024, miner=None)
print("Params:", list(cbm.parameters()))
