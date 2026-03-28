from lvm_utils.utils import save_conversations, load_conversations
from PIL import Image
import numpy as np

img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
conv = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "hello"}]}]

save_conversations([conv], out_path="test_no_label.zip")
out = load_conversations("test_no_label.zip")
print("No label test:", type(out), len(out))

save_conversations([conv, conv], labels=["cat", "dog"], out_path="test_label.zip")
out_convs, out_labels = load_conversations("test_label.zip")
print("Label test:", out_labels)
