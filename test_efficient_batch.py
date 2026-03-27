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

def efficient_extract(m, p, cs, c):
    target = m.base_model.model if hasattr(m, 'base_model') else m
    orig_head = target.lm_head
    target.lm_head = nn.Identity()
    
    # We copy the get_batch code
    device = next(m.parameters()).device
    batch = p.apply_chat_template(cs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    prompt_lens = attention_mask.sum(dim=1).long()
    
    seqs = []
    for i in range(input_ids.size(0)):
        n = int(prompt_lens[i].item())
        seq_i = input_ids[i, :n]
        append_tok = seq_i.new_full((1,), c.append_token_id)
        seq_i = torch.cat([seq_i, append_tok], dim=0)
        seqs.append(seq_i)
        
    from torch.nn.utils.rnn import pad_sequence
    import torch.nn.functional as F
    
    input_ids2 = pad_sequence(seqs, batch_first=True, padding_value=c.pad_token_id)
    attention_mask2 = (input_ids2 != c.pad_token_id).long()
    extra = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")}
    
    out = m(
        input_ids=input_ids2,
        attention_mask=attention_mask2,
        return_dict=True,
        **extra,
    )
    
    target.lm_head = orig_head
    last_hidden = out.logits
    batch_idx = torch.arange(last_hidden.size(0), device=device)
    emb = last_hidden[batch_idx, prompt_lens, :]
    return F.normalize(emb, p=2, dim=-1)

import time
start = time.time()
res1 = get_batch_conversation_embeddings_with_config(model, processor, [conv], config)
t1 = time.time() - start

start = time.time()
res2 = efficient_extract(model, processor, [conv], config)
t2 = time.time() - start

print("Matches:", torch.allclose(res1[0], res2, atol=1e-5))
print("T1:", t1, "T2:", t2)
