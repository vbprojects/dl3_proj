#%%
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from transformers.image_utils import load_image

from lvm_utils.cache_store import FirstStageCache
from lvm_utils.model_helpers import first_stage, load_model_id


def generate_cached_conversations(limit=10, split="train"):
    dataset = CIFAR10(root="./data", download=True, train=(split == "train"))
    model, processor = load_model_id(load_peft=False)
    cache = FirstStageCache(cache_root="./cache_data", webp_quality=80, running_ttl_seconds=1800)

    recovered = cache.recover_stale_running()
    if recovered:
        print(f"Recovered {recovered} stale running rows back to pending")

    conversations = []
    cache_hits = 0
    cache_misses = 0

    total = min(limit, len(dataset))
    for idx in tqdm(range(total), desc="Generating first_stage conversations"):
        image_raw, label = dataset[idx]
        image = load_image(image_raw)

        conv, hit, image_sha = cache.get_or_compute(
            image=image,
            compute_fn=lambda img: first_stage(img, processor, model),
            source_split=split,
            source_idx=idx,
            label=int(label),
        )

        if hit:
            cache_hits += 1
        else:
            cache_misses += 1

        conversations.append(conv)

        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            print(
                f"[{idx + 1}/{total}] image_sha={image_sha[:12]} "
                f"cache_hit={cache_hits} cache_miss={cache_misses}"
            )

    print("Final cache status:", cache.stats())
    return conversations


#%%
if __name__ == "__main__":
    cached_conversations = generate_cached_conversations(limit=1000, split="train")
    print(f"Collected {len(cached_conversations)} conversations")