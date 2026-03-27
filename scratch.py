# %%

import torch
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

# Load model and processor
model_id = "LiquidAI/LFM2-VL-450M"
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

model_id = "LiquidAI/LFM2-VL-450M"

# Option A (recommended first): 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # or torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    # quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,  # compute dtype
)
processor = AutoProcessor.from_pretrained(model_id)
#%%
model.save_pretrained("./LFM_VL-450-original")
# %%
# Load image and create conversation
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = load_image(url)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the image in great detail."},
        ],
    },
]

# Generate Answer
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1)
processor.batch_decode(outputs, skip_special_tokens=True)[0]

#%%
# 1) Generate

# %%
import peft
#%%
from peft import MissConfig, TaskType, get_peft_model

miss_cfg = MissConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    mini_r=8,
    miss_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, miss_cfg)
# %%
model.print_trainable_parameters()
#%%
# Load cifar10
from torchvision.datasets import CIFAR10

dataset = CIFAR10(root="./data", download=True)
#%%
from PIL import Image

# Load image
image = load_image(dataset[0][0])

# Upscale by a factor (example: 2x)
scale = 5
new_size = (image.width * scale, image.height * scale)

# Bicubic upscale
upscaled = image.resize(new_size, resample=Image.Resampling.BICUBIC)

upscaled
#%%
# Save result
# upscaled.save("output_bicubic.jpg", quality=95)


conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": upscaled},
            {"type": "text", "text": "Describe the image in great detail despite the low resolution. For some background this could be an image of an airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck."},
        ],
    },
]

# Generate Answer
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)

with torch.inference_mode():
    out_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False
    )

prompt_len = inputs["input_ids"].shape[1]
new_ids = out_ids[:, prompt_len:]
assistant_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

# %%
assistant_text
# %%
conversation_b = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": upscaled},
            {"type": "text", "text": "Describe the image in great detail despite the low resolution. For some background this could be an image of an airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck."}
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": assistant_text}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You must make a guess, make your best guess, even if you are not sure. What is the image of? One word"}
        ]
    }
]

inputs2 = processor.apply_chat_template(
    conversation_b,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)


gen_ids = model.generate(**inputs2, max_new_tokens=10)
#%%
# prompt length (where generated tokens start)
prompt_len = inputs2["input_ids"].shape[1]

# first generated token id
first_gen_token_id = gen_ids[:, prompt_len]  # shape: [batch]

# 2) Get hidden states for the full sequence
fwd = model(
    input_ids=gen_ids,
    attention_mask=torch.ones_like(gen_ids, device=gen_ids.device),
    output_hidden_states=True,
    return_dict=True,
)

# last layer hidden states: [batch, seq_len, hidden_size]
last_hidden = fwd.hidden_states[-1]

# embedding of first generated token: [batch, hidden_size]
first_gen_token_emb = last_hidden[:, prompt_len, :]
first_gen_token_emb.shape
first_gen_token_emb
#%%
assistant_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
assistant_text
#%%
from model_helpers import *

# %%
non_tunable_model, processor = load_model_id(load_peft=False)
#%%
from torchvision.datasets import CIFAR10

dataset = CIFAR10(root="./data", download=True)

# Get training data loader for CIFAR10
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

conversations = []
for i in range(10):
    image = load_image(dataset[i][0])
    conversation = first_stage(image, processor, non_tunable_model)
    conversations.append(conversation)
# %%
# simple train loop
model, processor = load_model_id()

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-4)
for epoch in range(1):
    for conversation in conversations:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(model.device)

        # 1) Forward on full conversation prompt
        out = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # logits at last prompt position predict the first generated token
        next_token_logits = out.logits[:, -1, :]                 # [B, vocab]
        first_gen_token_id = next_token_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

        # 2) Append that token and run a second forward pass
        input_ids_2 = torch.cat([inputs["input_ids"], first_gen_token_id], dim=1)
        attention_mask_2 = torch.cat(
            [inputs["attention_mask"], torch.ones_like(first_gen_token_id)],
            dim=1,
        )

        # pass through other multimodal fields too (pixel_values, etc.)
        extra = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        fwd2 = model(
            input_ids=input_ids_2,
            attention_mask=attention_mask_2,
            output_hidden_states=True,
            return_dict=True,
            **extra,
        )

        # embedding of first output token (last position after append)
        first_gen_token_emb = fwd2.hidden_states[-1][:, -1, :]   # [B, hidden]

        loss = first_gen_token_emb.norm(dim=-1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        break
    break
# %%
from utils import *
#%%
save_conversations(conversations, "conversations.zip")
#%%
loaded_conversations = load_conversations("conversations.zip")
#%%
loaded_conversations[0][0]["content"][0]["image"]