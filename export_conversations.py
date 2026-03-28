
#%%
import pandas as pd
from tqdm import tqdm
from lvm_utils.utils import *

index = pd.read_parquet("cache_data/index.parquet")

rehydrated_conversations = []
for json_path in tqdm(index['conversation_json_path'], desc="Rehydrating conversations"):
    conv = materialize_conversation_images(json.load(open("cache_data/" + json_path, "r")), './cache_data')
    rehydrated_conversations.append(conv)
#%%
index.label
#%%
save_conversations(rehydrated_conversations, "rehydrated_conversations.zip", labels=index.label.to_list(), webp_quality=80)
#%%
conversations, labels = load_conversations("rehydrated_conversations.zip")
conversations[0]
labels[:5]
# %%
conversations[0][0]['content'][0]['image']