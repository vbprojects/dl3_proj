#%%
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from transformers.image_utils import load_image

from lvm_utils.cache_store import FirstStageCache
from lvm_utils.model_helpers import first_stage, first_stage_batch, load_model_id


def generate_cached_conversations(limit=10, split="train", path = "./cache_data", batch_size=4):
    dataset = CIFAR10(root="./data", download=True, train=(split == "train"))
    model, processor = load_model_id(load_peft=False)
    cache = FirstStageCache(cache_root=path, webp_quality=80, running_ttl_seconds=1800)

    recovered = cache.recover_stale_running()
    if recovered:
        print(f"Recovered {recovered} stale running rows back to pending")

    conversations = []
    cache_hits = 0
    cache_misses = 0

    total = min(limit, len(dataset))
    for batch_start in tqdm(range(0, total, batch_size), desc="Generating first_stage conversations"):
        batch_end = min(batch_start + batch_size, total)
        batch_images = []
        batch_labels = []
        batch_idxs = []

        for idx in range(batch_start, batch_end):
            image_raw, label = dataset[idx]
            batch_images.append(load_image(image_raw))
            batch_labels.append(int(label))
            batch_idxs.append(idx)

        batch_convs, batch_hits, batch_shas = cache.get_or_compute_batch(
            images=batch_images,
            compute_batch_fn=lambda imgs: first_stage_batch(imgs, processor, model),
            source_splits=[split] * len(batch_images),
            source_idxs=batch_idxs,
            labels=batch_labels,
        )

        for i, hit in enumerate(batch_hits):
            if hit:
                cache_hits += 1
            else:
                cache_misses += 1
            conversations.append(batch_convs[i])

            idx = batch_idxs[i]
            image_sha = batch_shas[i]
            if (idx + 1) % 25 == 0 or (idx + 1) == total:
                print(
                    f"[{idx + 1}/{total}] image_sha={image_sha[:12]} "
                    f"cache_hit={cache_hits} cache_miss={cache_misses}"
                )

    print("Final cache status:", cache.stats())
    return conversations


#%%
if __name__ == "__main__":
    cached_conversations = generate_cached_conversations(limit=1400, split="train")
    print(f"Collected {len(cached_conversations)} conversations")