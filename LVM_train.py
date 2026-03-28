#%%
# Setup argparse, have it grab epoch
# import argparse
# parser = argparse.ArgumentParser(description="Train a model with PEFT.")
# parser.add_argument("--epoch", type=int, default=3, help="The number of epochs to train for.")
# args = parser.parse_args()
#%%
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, NAdam
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from lvm_utils.utils import materialize_conversation_images
from lvm_utils.model_helpers import load_model_id, build_conversation_embedding_config, get_batch_conversation_embeddings_with_config

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from pytorch_metric_learning import losses
from sklearn.neighbors import KNeighborsClassifier
import json
import gc

#%%
model, processor = load_model_id(load_peft=True)
model.gradient_checkpointing_enable()

# %%
# Build a proper Dataset and DataLoader for the Training Loop
class VLMDataset(Dataset):
    def __init__(self, index_df, cache_dir='./cache_data', label_mapping=None):
        # Only keep successfully processed ones that have labels
        self.df = index_df[index_df['status'].isin(['done', 'STATUS_DONE'])].reset_index(drop=True)
        self.cache_dir = cache_dir
        
        if 'label' not in self.df.columns:
            raise KeyError("The index dataframe must contain a 'label' column.")
            
        # Convert string labels to 0-indexed integers for ProxyAnchorLoss
        if label_mapping is None:
            self.labels, self.label_uniques = pd.factorize(self.df['label'])
            self.label_mapping = {val: i for i, val in enumerate(self.label_uniques)}
        else:
            self.label_mapping = label_mapping
            self.labels = self.df['label'].map(self.label_mapping).values
            self.label_uniques = pd.Series(list(self.label_mapping.keys()))
        
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

# Split index into train and validation (90% train, 10% val)
val_index = index.sample(frac=0.1, random_state=42)
train_index = index.drop(val_index.index)

train_dataset = VLMDataset(train_index)
val_dataset = VLMDataset(val_index, label_mapping=train_dataset.label_mapping)

batch_size = 5  # Keep this small to avoid VRAM OOM! (12GB -> ~4-8 depending on token length)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# %%
# Setup ProxyAnchorLoss & Optimizer
# LFM2-VL-450M text config has hidden_size=1024
try:
    embedding_size = model.config.text_config.hidden_size
except AttributeError:
    embedding_size = 1024 # Fallback

num_classes = len(train_dataset.label_uniques)
print(f"Initializing ProxyAnchorLoss: classes={num_classes}, embed_dim={embedding_size}")

loss_func = losses.ProxyAnchorLoss(
    num_classes=num_classes, 
    embedding_size=embedding_size, 
    margin=0.1, 
    alpha=32
).to(model.device)
# sup_con_loss = losses.SupConLoss(temperature=0.1)
# loss_func = losses.CrossBatchMemory(loss_func, embedding_size, memory_size=40, miner=None)

# Ensure we optimize both model (PEFT params) AND the ProxyAnchor embeddings
trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = NAdam([
    {"params": trainable_model_params, "lr": 1e-5}, 
    {"params": loss_func.parameters(), "lr": 1e-3}
])

# %%
# Training Loop
epochs = 20
accumulation_steps = 8 # Simulate a larger batch size (e.g. batch_size 4 * acc_steps 4 = effective batch 16)
llm_eos_config = build_conversation_embedding_config(processor)

model.train()
torch.cuda.empty_cache()

writer = SummaryWriter(log_dir="./runs/lvm_training")
global_step = 0

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1}/{epochs} ---")
    epoch_loss = 0.0
    
    progress = tqdm(trainloader, desc="Training")
    optimizer.zero_grad(set_to_none=True)
    
    for i, (batch_convs, batch_labels) in enumerate(progress):
        batch_labels = batch_labels.to(model.device)
        
        # Mixed Precision Context for fast 4-bit/bfloat16 evaluation
        with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
            # 1. Extract Contrastive Embeddings
            vecs, lens = get_batch_conversation_embeddings_with_config(
                model=model, 
                processor=processor, 
                conversations=batch_convs, 
                config=llm_eos_config,
                normalize=True # Usually desirable for metric learning
            )
            
            # 2. Compute Loss
            loss = loss_func(vecs, batch_labels)
            
        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # 3. Backprop
        loss.backward()
        
        # Step and reset gradients only after accumulating enough steps
        if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        real_loss = loss.item() * accumulation_steps
        epoch_loss += real_loss
        progress.set_postfix({"loss": f"{real_loss:.4f}"})
        
        writer.add_scalar('Loss/train_step', real_loss, global_step)
        global_step += 1
        
        # Free batch activations incrementally
        del vecs, loss, batch_convs, batch_labels
    
    avg_train_loss = epoch_loss / len(trainloader)
    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
    print(f"Epoch {epoch+1} Complete | Average Train Loss: {avg_train_loss:.4f}")

    # Validation Loop (using k-NN classifier instead of ProxyAnchorLoss)
    model.eval()
    
    train_embeddings = []
    train_targets = []
    val_embeddings = []
    val_targets = []
    
    with torch.no_grad():
        # 1. Extract Train Embeddings
        train_progress = tqdm(trainloader, desc="Extracting Train Embeddings")
        for batch_convs, batch_labels in train_progress:
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                vecs, lens = get_batch_conversation_embeddings_with_config(
                    model=model, 
                    processor=processor, 
                    conversations=batch_convs, 
                    config=llm_eos_config,
                    normalize=True
                )
            train_embeddings.append(vecs.float().cpu())
            train_targets.append(batch_labels.cpu())
            del vecs, batch_convs, batch_labels
            
        train_embeddings_tensor = torch.cat(train_embeddings)
        train_targets_tensor = torch.cat(train_targets)
        
        # 2. Extract Val Embeddings
        val_progress = tqdm(valloader, desc="Extracting Val Embeddings")
        for batch_convs, batch_labels in val_progress:
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                vecs, lens = get_batch_conversation_embeddings_with_config(
                    model=model, 
                    processor=processor, 
                    conversations=batch_convs, 
                    config=llm_eos_config,
                    normalize=True
                )
            val_embeddings.append(vecs.float().cpu())
            val_targets.append(batch_labels.cpu())
            del vecs, batch_convs, batch_labels
            
        val_embeddings_tensor = torch.cat(val_embeddings)
        val_targets_tensor = torch.cat(val_targets)
        
    # Fit k-NN and Score
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embeddings_tensor.numpy(), train_targets_tensor.numpy())
    val_acc = knn.score(val_embeddings_tensor.numpy(), val_targets_tensor.numpy())
    
    writer.add_scalar('Acc/val_knn', val_acc, epoch)
    print(f"Epoch {epoch+1} Complete | k-NN Val Accuracy: {val_acc:.4f}")
    
    model.train()

model.save_pretrained("./test20poch")
writer.close()

print("Training cycle complete!")
    
