#%%
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, NAdam
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
load_dotenv()
from lvm_utils.utils import materialize_conversation_images
from lvm_utils.model_helpers import load_model_id, build_conversation_embedding_config, get_batch_conversation_embeddings_with_config

import pandas as pd
from pytorch_metric_learning import losses
import json
import gc
# %%
# Build a proper Dataset and DataLoader for the Training Loop
class VLMDataset(Dataset):
    def __init__(self, index_df, cache_dir='./cache_data'):
        # Only keep successfully processed ones that have labels
        self.df = index_df[index_df['status'].isin(['done', 'STATUS_DONE'])].reset_index(drop=True)
        self.cache_dir = cache_dir
        
        if 'label' not in self.df.columns:
            raise KeyError("The index dataframe must contain a 'label' column.")
            
        # Convert string labels to 0-indexed integers for ProxyAnchorLoss
        self.labels, self.label_uniques = pd.factorize(self.df['label'])
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        conv_path = self.cache_dir + "/" + row['conversation_json_path']
        with open(conv_path, "r") as f:
            raw_conv = json.load(f)
        
        # Load the PIL images back into the dict
        # Ensure your materialization reads dict refs cleanly
        conv = materialize_conversation_images(raw_conv, self.cache_dir)
        return conv, self.labels[idx]

def collate_fn(batch):
    convs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return convs, labels

# %%
# Load index and initialize Dataset
index = pd.read_parquet("cache_data/index.parquet")
dataset = VLMDataset(index)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # NormalFloat4 is optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute dtype for the dequantization overhead
    bnb_4bit_use_double_quant=True, # Quantizes the quantization constants for extra VRAM savings
    llm_int8_skip_modules=["patch_embedding", "lm_head"] # Skip the patch embedding from integer quantization to prevent input cast errors!
)

batch_size = 5  # Keep this small to avoid VRAM OOM! (12GB -> ~4-8 depending on token length)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

model, processor = load_model_id(load_peft=True)
# model = PeftModel.from_pretrained(model, 
#     "test3poch",
#     is_trainable=True # 👈 here
#     )

#%%
llm_eos_config = build_conversation_embedding_config(processor)
progress = tqdm(trainloader, desc="Training")
import numpy as np

X, Y = np.empty((0, 1024)), np.empty((0,))

for i, (batch_convs, batch_labels) in enumerate(progress):
    batch_labels = batch_labels.to(model.device)
    
    # Mixed Precision Context for fast 4-bit/bfloat16 evaluation
    with torch.no_grad():
        # 1. Extract Contrastive Embeddings
        vecs, lens = get_batch_conversation_embeddings_with_config(
            model=model, 
            processor=processor, 
            conversations=batch_convs, 
            config=llm_eos_config,
            normalize=True # Usually desirable for metric learning
        )
        X = np.vstack((X, vecs.cpu().numpy()))
        Y = np.hstack((Y, batch_labels.cpu().numpy()))
np.savez("embeddings_nontrain.npz", X=X, Y=Y)