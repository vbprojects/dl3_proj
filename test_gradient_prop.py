import torch
import torch.nn as nn
from lvm_utils.model_helpers import load_model_id, first_stage, get_batch_conversation_embeddings_with_config, build_conversation_embedding_config
from PIL import Image
import numpy as np

model, processor = load_model_id(cache=False, load_peft=True)
model.gradient_checkpointing_enable()

# create a fake conversation
img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
conv = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe the image in great detail."},
        ],
    },
]

config = build_conversation_embedding_config(processor)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

vecs, lens = get_batch_conversation_embeddings_with_config(model, processor, [conv], config, normalize=True)
loss = vecs.sum()
loss.backward()

got_grad = False
for n, p in model.named_parameters():
    if p.requires_grad and p.grad is not None:
        got_grad = True
        break
print("Gradients propagating clearly:", got_grad)
