import io
import json
import hashlib
import zipfile
from pathlib import Path
from PIL import Image


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    """Encode image deterministically to PNG bytes for stable SHA keys."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def hash_pil_image(img: Image.Image) -> str:
    """Return SHA256 hash for a PIL image using deterministic PNG encoding."""
    return hashlib.sha256(_pil_to_png_bytes(img)).hexdigest()

def _pil_to_webp_bytes(img: Image.Image, quality=80):
    # WebP needs RGB/RGBA-like modes; convert if needed
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=6)
    return buf.getvalue()


def serialize_conversation_with_image_refs(
    conversation,
    cache_root,
    webp_quality=80,
):
    """
    Convert in-memory conversation to JSON-safe structure and persist images as WebP.

    Returns:
        tuple[serializable_conversation, first_image_rel_path]
    """
    cache_root = Path(cache_root)
    images_dir = cache_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    conv_out = []
    first_image_rel = None

    for msg in conversation:
        msg_out = {"role": msg["role"], "content": []}
        for item in msg["content"]:
            if item.get("type") == "image":
                img_sha = hash_pil_image(item["image"])
                img_rel = Path("images") / f"{img_sha}.webp"
                img_abs = cache_root / img_rel
                if not img_abs.exists():
                    img_abs.write_bytes(_pil_to_webp_bytes(item["image"], quality=webp_quality))
                if first_image_rel is None:
                    first_image_rel = img_rel.as_posix()
                msg_out["content"].append(
                    {
                        "type": "image_ref",
                        "sha256": img_sha,
                        "path": img_rel.as_posix(),
                    }
                )
            else:
                msg_out["content"].append(item)
        conv_out.append(msg_out)

    return conv_out, first_image_rel


def materialize_conversation_images(serialized_conversation, cache_root):
    """Restore a serialized conversation by replacing image_ref entries with PIL images."""
    cache_root = Path(cache_root)
    restored = []
    for msg in serialized_conversation:
        msg_out = {"role": msg["role"], "content": []}
        for item in msg["content"]:
            if item.get("type") == "image_ref":
                img_path = cache_root / item["path"]
                img = Image.open(img_path).copy()
                msg_out["content"].append({"type": "image", "image": img})
            else:
                msg_out["content"].append(item)
        restored.append(msg_out)
    return restored


def create_hashable_conversation(conversation, cache_root="./cache_data", webp_quality=80):
    """Compatibility helper used by scripts; returns JSON-safe conversation payload."""
    serial, _ = serialize_conversation_with_image_refs(
        conversation,
        cache_root=cache_root,
        webp_quality=webp_quality,
    )
    return serial

def save_conversations(conversations, out_path=None, labels=None, webp_quality=80):
    """
    conversations: list of conversation dicts
      each message content item is like:
      {"type": "image", "image": PIL.Image.Image}
      or {"type": "text", "text": "..."}
    out_path: e.g. "conversations.zip"
    labels: list of labels associated with each conversation (optional)
    """
    out_path = Path(out_path)
    manifest = []
    seen = {}  # hash -> archive path

    if labels is not None and len(labels) != len(conversations):
        raise ValueError("labels must have the same length as conversations")

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_LZMA) as zf:
        for i, conv in enumerate(conversations):
            conv_out = []
            for msg in conv:
                msg_out = {"role": msg["role"], "content": []}
                for item in msg["content"]:
                    if item["type"] == "image":
                        img_bytes = _pil_to_webp_bytes(item["image"], quality=webp_quality)
                        h = hashlib.sha256(img_bytes).hexdigest()
                        arc_img_path = f"images/{h}.webp"

                        if h not in seen:
                            zf.writestr(arc_img_path, img_bytes)
                            seen[h] = arc_img_path

                        msg_out["content"].append({
                            "type": "image_ref",
                            "path": seen[h],
                        })
                    else:
                        msg_out["content"].append(item)
                conv_out.append(msg_out)
            
            entry = {"conversation": conv_out}
            if labels is not None:
                entry["label"] = labels[i]
            manifest.append(entry)

        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))


def load_conversations(in_path):
    """
    Returns (conversations, labels) if labels were saved, else just returns conversations
    in original shape, restoring PIL images:
      {"type":"image", "image": PIL.Image.Image}
    """
    in_path = Path(in_path)
    conversations = []
    labels = []
    has_labels = False

    with zipfile.ZipFile(in_path, mode="r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

        for entry in manifest:
            # Handle backward compatibility where manifest is just a list of lists
            if isinstance(entry, list):
                conv_data = entry
            else:
                conv_data = entry.get("conversation", [])
                if "label" in entry:
                    labels.append(entry["label"])
                    has_labels = True

            conv_out = []
            for msg in conv_data:
                msg_out = {"role": msg["role"], "content": []}
                for item in msg["content"]:
                    if item["type"] == "image_ref":
                        data = zf.read(item["path"])
                        img = Image.open(io.BytesIO(data)).copy()
                        msg_out["content"].append({"type": "image", "image": img})
                    else:
                        msg_out["content"].append(item)
                conv_out.append(msg_out)
            conversations.append(conv_out)

    if has_labels:
        return conversations, labels
    return conversations

def upscale_image(image : Image.Image, scale : float) -> Image.Image:
    """Upscale the input image by the given scale factor using bicubic interpolation.
    
    Args:
        image (Image.Image): The input image.
        scale (float): The scale factor by which to upscale the image.
    
    Returns:
        upscaled (Image.Image): The upscaled image.
    """
    new_size = (image.width * scale, image.height * scale)
    upscaled = image.resize(new_size, resample=Image.Resampling.BICUBIC)
    return upscaled