import lvm_utils.cache_store as cs

def get_or_compute_batch(
    self,
    images,
    compute_batch_fn,
    source_splits=None,
    source_idxs=None,
    labels=None,
):
    from lvm_utils.cache_store import hash_pil_image
    shas = [hash_pil_image(img) for img in images]
    
    results = [None] * len(images)
    hits = [False] * len(images)
    
    misses = []
    
    for i, sha in enumerate(shas):
        lbl = labels[i] if labels else None
        cached = self.load_done_conversation(sha)
        if cached is not None:
            if lbl is not None:
                self._upsert(sha, {"label": lbl})
            results[i] = cached
            hits[i] = True
        else:
            misses.append(i)
            self.mark_running(
                sha,
                source_split=source_splits[i] if source_splits else None,
                source_idx=source_idxs[i] if source_idxs else None,
                label=lbl,
            )
            
    if misses:
        try:
            miss_images = [images[i] for i in misses]
            miss_convs = compute_batch_fn(miss_images)
            for j, miss_idx in enumerate(misses):
                sha = shas[miss_idx]
                conv = miss_convs[j]
                self.save_done_conversation(
                    sha,
                    conv,
                    source_split=source_splits[miss_idx] if source_splits else None,
                    source_idx=source_idxs[miss_idx] if source_idxs else None,
                    label=labels[miss_idx] if labels else None,
                )
                results[miss_idx] = conv
        except Exception as exc:
            for j, miss_idx in enumerate(misses):
                self.mark_failed(shas[miss_idx], error=str(exc))
            raise exc

    return results, hits, shas

cs.FirstStageCache.get_or_compute_batch = get_or_compute_batch
print("Patched!")
