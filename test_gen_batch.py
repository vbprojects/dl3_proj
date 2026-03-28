import torch
from lvm_utils.model_helpers import load_model_id
from PIL import Image
import numpy as np

model, processor = load_model_id(cache=False, load_peft=False)
model.eval()

img1 = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
img2 = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))

conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "text", "text": "What is this?"},
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img2},
                {"type": "text", "text": "Describe the image in detail. Very very nicely."},
            ],
        },
    ],
]

processor.tokenizer.padding_side = "left"

inputs = processor.apply_chat_template(
    conversations,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
    padding=True,
).to(model.device)

with torch.no_grad():
    print("Generating...")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print("Shape of outputs:", outputs.shape)
    
    # decode only the newly generated tokens
    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    print("New tokens shape:", new_tokens.shape)
    texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
    print("Generated texts:", texts)
