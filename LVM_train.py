#%%
# Setup argparse, have it grab epoch
# import argparse
# parser = argparse.ArgumentParser(description="Train a model with PEFT.")
# parser.add_argument("--epoch", type=int, default=3, help="The number of epochs to train for.")
# args = parser.parse_args()
#%%

from lvm_utils.utils import *
from lvm_utils.model_helpers import *
import pandas as pd
#%%
model, processor = load_model_id(load_peft = True)
model.gradient_checkpointing_enable()
# %%
index = pd.read_parquet("cache_data/index.parquet")
json_path = index.head(1)['conversation_json_path'][0]
import json

conversations = [materialize_conversation_images(json.load(open("cache_data/" + json_path, "r")), './cache_data') for json_path in index.head(5)['conversation_json_path']]
#%%
llm_eos_config = build_conversation_embedding_config(processor)
# %%
# with torch.no_grad():
vecs = get_batch_conversation_embeddings_with_config(model = model, processor=processor, conversations = conversations, config = llm_eos_config)
# %%
npvec = vecs[0].detach().cpu().numpy()
# %%
import numpy as np
index.head(5)
#%%

#%%
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)
